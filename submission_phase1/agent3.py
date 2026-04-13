import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM    = 18
STACK_SIZE = 4
INPUT_DIM  = OBS_DIM * STACK_SIZE
_device    = torch.device("cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

_model = DuelingDQN(INPUT_DIM, len(ACTIONS)).to(_device)
_weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
_model.load_state_dict(torch.load(_weights_path, map_location="cpu"))
_model.eval()

_obs_stack           = deque(maxlen=STACK_SIZE)
_step                = 0
_stuck_count         = 0
_prev_obs            = None
_was_attached        = False
_consecutive_contact = 0

def _is_new_episode(obs: np.ndarray) -> bool:
    global _prev_obs
    if _prev_obs is None:
        return True
    if _prev_obs[17] == 1 and obs[17] == 0:
        return True
    if np.sum(np.abs(obs - _prev_obs)) > 10:
        return True
    return False

def _reset_stack(obs: np.ndarray):
    _obs_stack.clear()
    for _ in range(STACK_SIZE):
        _obs_stack.append(obs.copy())

def _get_stack() -> np.ndarray:
    return np.concatenate(list(_obs_stack), axis=0).astype(np.float32)

def _get_fallback_action(obs: np.ndarray, stuck_count: int) -> str:
    if obs[17] == 1:
        if stuck_count % 2 == 1:
            return "L45" if (stuck_count // 2) % 2 == 0 else "R45"
        else:
            return "FW"
    return None

def policy(obs, rng=None) -> str:
    global _step, _stuck_count, _prev_obs
    global _was_attached, _consecutive_contact

    obs = np.array(obs, dtype=np.float32)

    if _is_new_episode(obs):
        _reset_stack(obs)
        _step                = 0
        _stuck_count         = 0
        _was_attached        = False
        _consecutive_contact = 0
    else:
        _obs_stack.append(obs.copy())

    _prev_obs = obs.copy()

    if obs[17] == 1:
        _stuck_count += 1
    else:
        _stuck_count = 0

    front_near = obs[6] + obs[8]
    ir         = obs[16]

    if not _was_attached:
        if ir == 1 and front_near >= 1:
            _consecutive_contact += 1
            if _consecutive_contact >= 3:
                _was_attached = True
        else:
            _consecutive_contact = 0

    if _was_attached:
        _step += 1
        return "L45"

    fallback = _get_fallback_action(obs, _stuck_count)
    if fallback is not None:
        _step += 1
        return fallback

    stacked = _get_stack()
    with torch.no_grad():
        s = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(_device)
        _ = _model(s)  

    _step += 1
    return "L45"