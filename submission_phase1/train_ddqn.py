# import os
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# import cv2
# cv2.imshow = lambda *args, **kwargs: None
# cv2.waitKey = lambda *args, **kwargs: None

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from obelix import OBELIX

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM   = 18
STACK_SIZE = 4
INPUT_DIM  = OBS_DIM * STACK_SIZE
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

class ObsStack:
    def __init__(self, obs_dim: int, size: int):
        self.size    = size
        self.obs_dim = obs_dim
        self.frames  = deque(maxlen=size)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        for _ in range(self.size):
            self.frames.append(obs.copy())
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32).to(device),
            torch.tensor(actions,                dtype=torch.long).to(device),
            torch.tensor(rewards,                dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states),  dtype=torch.float32).to(device),
            torch.tensor(dones,                  dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buffer)


def shape_reward(step_reward: float, obs: np.ndarray, env) -> float:

    shaped = step_reward

    if env.enable_push:
        if env.active_state == "P":
            shaped += 5.0
        else:
            shaped -= 2.0

    return shaped


GAMMA            = 0.99
LR               = 5e-4
BATCH_SIZE       = 128
MEMORY_SIZE      = 50_000
EPISODES         = 1000
TARGET_UPDATE_FREQ = 20
WARMUP_STEPS     = 1000

EPSILON_START    = 1.0
EPSILON_MIN      = 0.05
EPSILON_DECAY    = 0.997

def get_exploration_weights(obs: np.ndarray, step: int) -> list:
    """
    Obs-conditioned + step-aware action bias for random exploration.
    obs layout:
      [0:8]  sonar near, [8:16] sonar far, [16] IR, [17] stuck flag
    Note: obs[17] is STUCK not attached. Use env.enable_push for attachment.
    """
    near_bits = obs[0:8]
    far_bits  = obs[8:16]
    ir        = obs[16]

    if ir == 1:
        return [0.02, 0.06, 0.86, 0.04, 0.02]

    if near_bits.sum() > 0:
        left  = near_bits[:3].sum()
        right = near_bits[5:].sum()
        if left > right:
            return [0.05, 0.40, 0.40, 0.10, 0.05]
        elif right > left:
            return [0.05, 0.10, 0.40, 0.40, 0.05]
        return [0.04, 0.12, 0.68, 0.12, 0.04]

    if far_bits.sum() > 0:
        left  = far_bits[:3].sum()
        right = far_bits[5:].sum()
        if left > right:
            return [0.08, 0.32, 0.42, 0.12, 0.06]
        elif right > left:
            return [0.06, 0.12, 0.42, 0.32, 0.08]
        return [0.05, 0.12, 0.62, 0.12, 0.09]

    if step < 150:
        return [0.04, 0.06, 0.82, 0.04, 0.04]
    elif step < 400:
        return [0.08, 0.10, 0.64, 0.10, 0.08]
    else:
        return [0.30, 0.05, 0.30, 0.05, 0.30]



def get_episode_type(ep: int) -> str:

    if ep < 300:
        return "explore" if np.random.rand() < 0.8 else "mixed"
    elif ep < 600:
        r = np.random.rand()
        if r < 0.30:   return "explore"
        elif r < 0.80: return "mixed"
        else:          return "greedy"
    else:
        r = np.random.rand()
        if r < 0.10:   return "explore"
        elif r < 0.50: return "mixed"
        else:          return "greedy"

EPISODE_EPSILON = {
    "explore": 0.95,
    "mixed":   None,   
    "greedy":  0.05,
}

def select_action(stacked_state: np.ndarray, step: int,
                  ep_epsilon: float) -> int:
    if np.random.rand() < ep_epsilon:
        current_obs = stacked_state[-OBS_DIM:]       
        weights = get_exploration_weights(current_obs, step)
        return np.random.choice(len(ACTIONS), p=weights)
    with torch.no_grad():
        s = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0).to(device)
        return policy_net(s).argmax(dim=1).item()

def train_step():
    if len(buffer) < WARMUP_STEPS:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        
        best_actions = policy_net(next_states).argmax(dim=1)
        max_next_q   = target_net(next_states).gather(
            1, best_actions.unsqueeze(1)
        ).squeeze(1)
        target_q = rewards + GAMMA * max_next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


PHASE1_SEEDS = [42, 7, 13, 99, 256]
PHASE2_SEEDS = list(range(0, 20))

def get_seed_for_episode(ep: int):
    if ep < 300:
        return PHASE1_SEEDS[ep % len(PHASE1_SEEDS)]
    elif ep < 600:
        return PHASE2_SEEDS[ep % len(PHASE2_SEEDS)]
    return None   # fully random


obs_stack  = ObsStack(OBS_DIM, STACK_SIZE)
policy_net = DuelingDQN(INPUT_DIM, len(ACTIONS)).to(device)
target_net = DuelingDQN(INPUT_DIM, len(ACTIONS)).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer    = ReplayBuffer(MEMORY_SIZE)


reward_history   = []
best_mean_reward = -float('inf')
global_epsilon   = EPSILON_START

for ep in range(EPISODES):
    seed     = get_seed_for_episode(ep)
    ep_type  = get_episode_type(ep)
    ep_epsilon = EPISODE_EPSILON[ep_type] if EPISODE_EPSILON[ep_type] is not None \
                 else global_epsilon

    env = OBELIX(
        scaling_factor=1,
        max_steps=2000,
        difficulty=0,
        seed=seed
    )

    raw_obs = env.reset().astype(np.float32)
    state   = obs_stack.reset(raw_obs)

    done         = False
    total_reward = 0.0
    step         = 0
    got_sensor   = False

    while not done:
        action_idx = select_action(state, step, ep_epsilon)

        
        next_raw_obs, step_reward, done = env.step(
            ACTIONS[action_idx]
        )
        next_raw_obs = next_raw_obs.astype(np.float32)

        shaped = shape_reward(step_reward, next_raw_obs, env)

        next_state = obs_stack.step(next_raw_obs)

        if ep_type != "greedy":
            buffer.push(state, action_idx, shaped, next_state, done)

        if next_raw_obs[:17].sum() > 0:
            got_sensor = True

        state        = next_state
        total_reward += step_reward   
        step         += 1

        if ep_type != "greedy":
            train_step()

    if ep % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    global_epsilon = max(EPSILON_MIN, global_epsilon * EPSILON_DECAY)

    reward_history.append(total_reward)

    if ep % 10 == 0:
        phase   = 1 if ep < 300 else (2 if ep < 600 else 3)
        mean100 = np.mean(reward_history[-100:])
        print(
            f"Ep {ep+1:4d} | Ph{phase} | {ep_type:7s} | seed={str(seed):>4} | "
            f"gε={global_epsilon:.3f} | epε={ep_epsilon:.3f} | "
            f"steps={step:4d} | reward={total_reward:7.1f} | "
            f"mean100={mean100:7.1f} | "
            f"attached={env.enable_push} | sensor={got_sensor}"
        )

        if mean100 > best_mean_reward and ep_type == "greedy":
            best_mean_reward = mean100
            torch.save(policy_net.state_dict(), "weights.pth")
            print(f"           ↑ New best saved (mean100={mean100:.1f})")

torch.save(policy_net.state_dict(), "weights.pth")
print("\nTraining complete. weights.pth saved.")