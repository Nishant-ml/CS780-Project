import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SWEEP_TRIGGER         = 25
RECENT_WINDOW         = 40
SONAR_STABLE_LIMIT    = 12
FALSE_DETECT_COOLDOWN = 10


# ================== ESCAPE CONTROLLER ==================
class EscapeController:

    def __init__(self):
        self.reset()

    def reset(self):
        self.active            = False
        self.phase             = "idle"
        self.turn_steps_done   = 0
        self.turn_dir          = 1
        self.attempts          = 0
        self.consecutive_fails = 0
        self.last_obs          = None

    def _sensor_guided_turn(self, obs: np.ndarray) -> int:
        left_hits  = int(obs[0] + obs[1] + obs[2] + obs[3])
        right_hits = int(obs[12] + obs[13] + obs[14] + obs[15])
        if left_hits < right_hits:
            return 1
        elif right_hits < left_hits:
            return -1
        else:
            return 1 if self.attempts % 2 == 1 else -1

    def _turn_steps_needed(self) -> int:
        if self.last_obs is not None:
            left_hits  = int(self.last_obs[0] + self.last_obs[1] +
                             self.last_obs[2] + self.last_obs[3])
            right_hits = int(self.last_obs[12] + self.last_obs[13] +
                             self.last_obs[14] + self.last_obs[15])
            fwd_hits   = int(sum(self.last_obs[4:12]))
            if left_hits > 0 and right_hits > 0 and fwd_hits > 0:
                return 5  
        if self.consecutive_fails == 0:
            return 3   
        elif self.consecutive_fails == 1:
            return 3 
        else:
            return 5

    def update(self, stuck: bool, obs: np.ndarray) -> None:
        self.last_obs = obs[:18].copy()
        if self.active:
            if self.phase == "idle" and stuck:
                self.consecutive_fails += 1
                self.active          = True
                self.phase           = "turning"
                self.turn_steps_done = 0
                self.attempts       += 1
                self.turn_dir        = self._sensor_guided_turn(obs)
            elif self.phase == "idle" and not stuck:
                self.active            = False
                self.consecutive_fails = 0
                self.attempts          = 0
        else:
            if stuck:
                self.active          = True
                self.phase           = "turning"
                self.turn_steps_done = 0
                self.attempts       += 1
                self.turn_dir        = self._sensor_guided_turn(obs)

    def act(self):
        if not self.active:
            return None
        if self.phase == "turning":
            needed = self._turn_steps_needed()
            self.turn_steps_done += 1
            if self.turn_steps_done >= needed:
                self.phase           = "probing"
                self.turn_steps_done = 0
            return "L45" if self.turn_dir == 1 else "R45"
        if self.phase == "probing":
            self.phase = "idle"
            return "FW"
        return "FW"


# ================== SWEEP CONTROLLER ==================
class SweepController:
    TURN_STEPS = 2

    def __init__(self):
        self.reset()

    def reset(self):
        self.direction       = 1
        self.turning         = False
        self.turn_steps_done = 0

    def act(self, stuck: bool) -> str:
        if stuck and not self.turning:
            self.turning         = True
            self.turn_steps_done = 0
            self.direction      *= -1
        if self.turning:
            self.turn_steps_done += 1
            if self.turn_steps_done >= self.TURN_STEPS:
                self.turning = False
            return "L45" if self.direction == 1 else "R45"
        return "FW"


# ================== WALL CHASE DETECTOR ==================
class WallChaseDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_pattern       = None
        self.sonar_stable_steps = 0
        self.cooldown           = 0

    def update(self, obs: np.ndarray, box_detected: bool) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
            return True

        if not box_detected:
            self.sonar_stable_steps = 0
            self.last_pattern       = None
            return False

        current_pattern = tuple(obs[:16].astype(int))
        if current_pattern == self.last_pattern:
            self.sonar_stable_steps += 1
        else:
            self.sonar_stable_steps = 0
            self.last_pattern       = current_pattern

        if self.sonar_stable_steps >= SONAR_STABLE_LIMIT:
            self.cooldown           = FALSE_DETECT_COOLDOWN
            self.sonar_stable_steps = 0
            self.last_pattern       = None
            return True

        return False


# ================== MODEL ==================
class DRQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x, hidden=None):
        x = self.fc(x)
        x, hidden = self.lstm(x, hidden)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return v + a - a.mean(dim=2, keepdim=True), hidden


# ================== LOAD ==================
model = DRQN().to(device)
import os

model_path = os.path.join(os.path.dirname(__file__), "best_success2.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


hidden            = None
steps_blind       = 0
recently_detected = [0]
escape_ctrl       = EscapeController()
sweep_ctrl        = SweepController()
wall_detector     = WallChaseDetector()
step_count        = 0


def reset():
    global hidden, steps_blind, recently_detected, step_count
    hidden            = None
    steps_blind       = 0
    recently_detected = [0]
    step_count        = 0
    escape_ctrl.reset()
    sweep_ctrl.reset()
    wall_detector.reset()


def policy(obs,rng):
    global hidden, steps_blind, recently_detected, step_count

    obs   = obs.astype(np.float32)
    stuck = obs[17] == 1

    if not stuck:
        if obs[:16].sum() > 0:
            steps_blind = 0
        else:
            steps_blind += 1

    blind_feat = np.array([min(steps_blind, 100) / 100.0], dtype=np.float32)
    obs_aug    = np.concatenate([obs, blind_feat])

    sensors_active = obs_aug[:16].sum() > 0
    box_detected   = sensors_active and not stuck

    if hidden is not None:
        hidden = (hidden[0].detach(), hidden[1].detach())
    obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        q, hidden = model(obs_t, hidden)
    q_vals = q[0, 0].cpu().numpy()

    step_count += 1

    escape_ctrl.update(stuck, obs_aug)
    escape_action = escape_ctrl.act()
    if escape_action is not None:
        return escape_action

    is_wall_chase = wall_detector.update(obs_aug, box_detected)
    if is_wall_chase:
        box_detected         = False
        recently_detected[0] = 0

    if box_detected:
        recently_detected[0] = RECENT_WINDOW
    elif recently_detected[0] > 0:
        recently_detected[0] -= 1

    if box_detected or recently_detected[0] > 0:
        return ACTIONS[np.argmax(q_vals)]

    if steps_blind > SWEEP_TRIGGER:
        return sweep_ctrl.act(stuck)

    return ACTIONS[np.argmax(q_vals)]

