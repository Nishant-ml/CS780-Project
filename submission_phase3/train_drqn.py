# ================== IMPORTS ==================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from obelix import OBELIX

# ================== CONFIG ==================
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN   = 32
BURNIN    = 8
LEARN_LEN = SEQ_LEN - BURNIN  # 24

BATCH_SIZE   = 32
GAMMA        = 0.97
LR           = 5e-4

EPS_START    = 0.75
EPS_END      = 0.05
EPS_DECAY    = 0.998

EPISODES      = 2000
TARGET_UPDATE = 20

SWEEP_TRIGGER         = 15   
RECENT_WINDOW         = 30
SONAR_STABLE_LIMIT    = 20
FALSE_DETECT_COOLDOWN = 40



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
                return 4   
        if self.consecutive_fails == 0:
            return 2   
        elif self.consecutive_fails == 1:
            return 3 
        else:
            return 4

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


# ================== CURIOSITY ==================
class SensorMemory:
    def __init__(self):
        self.seen: set = set()

    def reset(self):
        self.seen.clear()

    def intrinsic(self, obs: np.ndarray) -> float:
        if obs[17] == 1:
            return 0.0
        key = tuple(obs[:16].astype(int))
        if key not in self.seen:
            self.seen.add(key)
            return 1.0
        return 0.0


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


# ================== REPLAY ==================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, episode):
        self.buffer.append(episode)

    def sample(self):
        batch = []
        while len(batch) < BATCH_SIZE:
            ep = random.choice(self.buffer)
            if len(ep) < SEQ_LEN:
                continue
            if random.random() < 0.3:
                start = max(0, len(ep) - SEQ_LEN)
            else:
                start = random.randint(0, len(ep) - SEQ_LEN)
            batch.append(ep[start: start + SEQ_LEN])
        return batch

    def __len__(self):
        return len(self.buffer)


# ================== OBS AUGMENTATION ==================
def augment_obs(obs: np.ndarray, steps_blind: int) -> np.ndarray:
    blind_feat = np.array([min(steps_blind, 100) / 100.0], dtype=np.float32)
    return np.concatenate([obs, blind_feat])  # 18 → 19


# ================== ACTION SELECTION ==================
def select_action(obs_aug, hidden, epsilon, model,
                  sweep_ctrl, recently_detected,
                  escape_ctrl, wall_detector):

    sensors_active = obs_aug[:16].sum() > 0
    steps_blind    = int(obs_aug[18] * 100)
    stuck          = obs_aug[17] == 1

    escape_ctrl.update(stuck, obs_aug)

    # ---- PRIORITY 0: ESCAPE ----
    escape_action = escape_ctrl.act()
    if escape_action is not None:
        obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden = model(obs_t, hidden)
        return ACTIONS.index(escape_action), hidden

    box_detected = sensors_active and not stuck

    is_wall_chase = wall_detector.update(obs_aug, box_detected)
    if is_wall_chase:
        box_detected         = False
        recently_detected[0] = 0

    if box_detected:
        recently_detected[0] = RECENT_WINDOW
    elif recently_detected[0] > 0:
        recently_detected[0] -= 1

    # ---- PHASE 1: DRQN ----
    if box_detected or recently_detected[0] > 0:
        if random.random() < epsilon:
            return random.randint(0, 4), hidden
        obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q, hidden = model(obs_t, hidden)
        return q.argmax().item(), hidden

    # ---- PHASE 2: SWEEP ----
    if steps_blind > SWEEP_TRIGGER:
        action_str = sweep_ctrl.act(stuck)
        return ACTIONS.index(action_str), hidden

    # ---- PHASE 3: STRUCTURED RANDOM ----
    if random.random() < epsilon:
        if random.random() < 0.6:
            return 2, hidden
        return random.randint(0, 4), hidden

    obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        q, hidden = model(obs_t, hidden)
    return q.argmax().item(), hidden


# ================== REWARD SHAPING ==================
def shape_reward(obs,action_idx,raw_reward,env,done,
                 intrinsic:float,step_t:int,max_steps:int):
    shaped=0.0
    shaped-=0.01

    if obs[17]==1:
        shaped-=0.3


    if done and env.enable_push and env._box_touches_boundary(
        env.box_center_x,env.box_center_y
    ):
        efficiency=40*((max_steps-step_t)/max_steps)
        shaped+=50.0+efficiency

    if done and not env.enable_push:
        shaped-=5.0

    return float(shaped)


# ================== TRAIN STEP ==================
def train_step(model, target, buffer, optimizer):
    batch = buffer.sample()

    states, actions, rewards, next_states, dones = [], [], [], [], []
    for seq in batch:
        s, a, r, ns, d = zip(*seq)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(ns)
        dones.append(d)

    states      = torch.FloatTensor(np.array(states)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    actions     = torch.LongTensor(np.array(actions)).to(device)
    rewards     = torch.FloatTensor(np.array(rewards)).to(device)
    dones       = torch.FloatTensor(np.array(dones)).to(device)

    with torch.no_grad():
        _, h_burn     = model(states[:, :BURNIN])
        _, h_burn_tgt = target(next_states[:, :BURNIN])

    s_l  = states[:, BURNIN:]
    ns_l = next_states[:, BURNIN:]
    a_l  = actions[:, BURNIN:]
    r_l  = rewards[:, BURNIN:]
    d_l  = dones[:, BURNIN:]

    q_vals,        _ = model(s_l, h_burn)
    next_q_online, _ = model(ns_l, h_burn)
    next_q_target, _ = target(ns_l, h_burn_tgt)

    q_vals = q_vals.gather(2, a_l.unsqueeze(-1)).squeeze(-1)

    best_a     = next_q_online.argmax(dim=2)
    max_next_q = next_q_target.gather(2, best_a.unsqueeze(-1)).squeeze(-1)

    target_q = r_l + GAMMA * max_next_q * (1 - d_l)

    loss = nn.functional.smooth_l1_loss(q_vals, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return loss.item()


# ================== CURRICULUM ==================
def get_seed(ep):
    if ep < 500:
        return ep % 50
    elif ep < 1000:
        return ep % 100
    return None


def get_max_steps(ep):
    if ep < 100:
        return 1000
    return 2000


# ================== INIT ==================
model  = DRQN().to(device)
target = DRQN().to(device)

ckpt_path = "new_success.pth"
try:
    ckpt       = torch.load(ckpt_path, map_location="cpu")
    input_size = ckpt["fc.0.weight"].shape[1]
    if input_size == 19:
        model.load_state_dict(ckpt)
        target.load_state_dict(model.state_dict())
        print(f"Resumed from {ckpt_path} (19-input confirmed)")
    else:
        print(f"Checkpoint is {input_size}-input — training from scratch")
except Exception as e:
    print(f"No checkpoint ({e}) — training from scratch")

optimizer     = optim.Adam(model.parameters(), lr=LR)
buffer        = ReplayBuffer()
escape_ctrl   = EscapeController()
sweep_ctrl    = SweepController()
sensor_mem    = SensorMemory()
wall_detector = WallChaseDetector()

epsilon      = EPS_START
best_success = 0.0

reward_history  = []
success_history = []
attach_history  = []

print(f"Config | LR={LR} | EPS={EPS_START}→{EPS_END} | GAMMA={GAMMA} | SEQ={SEQ_LEN}")


# ================== TRAIN LOOP ==================
for ep in range(EPISODES):

    max_steps = get_max_steps(ep)
    env = OBELIX(
        scaling_factor=5,
        max_steps=max_steps,
        difficulty=3,
        wall_obstacles=True,
        seed=get_seed(ep)
    )

    obs               = env.reset().astype(np.float32)
    hidden            = None
    steps_blind       = 0
    recently_detected = [0]

    escape_ctrl.reset()
    sweep_ctrl.reset()
    sensor_mem.reset()
    wall_detector.reset()

    episode             = []
    total_raw_reward    = 0.0
    total_shaped_reward = 0.0
    episode_success     = False
    episode_attach      = False
    steps_taken         = 0

    for t in range(max_steps):

        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())

        obs_aug = augment_obs(obs, steps_blind)

        action_idx, hidden = select_action(
            obs_aug, hidden, epsilon, model,
            sweep_ctrl, recently_detected,
            escape_ctrl, wall_detector
        )

        next_obs, raw_reward, done = env.step(ACTIONS[action_idx], render=True)
        next_obs    = next_obs.astype(np.float32)
        steps_taken = t + 1

        if next_obs[17] == 0:
            if next_obs[:16].sum() > 0:
                steps_blind = 0
            else:
                steps_blind += 1

        next_obs_aug = augment_obs(next_obs, steps_blind)
        intrinsic    = sensor_mem.intrinsic(next_obs)

        if env.enable_push:
            episode_attach = True
        if done and env.enable_push and env._box_touches_boundary(
            env.box_center_x, env.box_center_y
        ):
            episode_success = True

        shaped = shape_reward(
            next_obs, action_idx, raw_reward, env, done,
            intrinsic, t, max_steps
        )

        episode.append((obs_aug, action_idx, shaped, next_obs_aug, float(done)))

        obs = next_obs
        total_raw_reward    += raw_reward
        total_shaped_reward += shaped

        if done:
            break

    buffer.push(episode)

    loss_val = 0.0
    if len(buffer) > 20:
        loss_val = train_step(model, target, buffer, optimizer)

    if ep % TARGET_UPDATE == 0:
        target.load_state_dict(model.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    reward_history.append(total_raw_reward)
    success_history.append(episode_success)
    attach_history.append(episode_attach)

    torch.save(model.state_dict(), "latest.pth")

    if len(success_history) >= 50:
        success_rate = np.mean(success_history[-50:])
        if success_rate > best_success:
            best_success = success_rate
            torch.save(model.state_dict(), "best_success.pth")
            print(f"  >> Saved BEST SUCCESS: {success_rate*100:.1f}%")

    if ep % 100 == 0:
        torch.save(model.state_dict(), f"checkpoint_{ep}.pth")

    if ep % 10 == 0:
        s_rate = np.mean(success_history[-50:]) * 100 if len(success_history) >= 50 else 0.0
        a_rate = np.mean(attach_history[-50:])  * 100 if len(attach_history)  >= 50 else 0.0
        shaped_per_step = total_shaped_reward / steps_taken if steps_taken > 0 else 0.0
        print(
            f"Ep {ep:4d} | Steps: {steps_taken:4d} | "
            f"RawR: {total_raw_reward:10.1f} | ShapedR: {total_shaped_reward:8.2f} | "
            f"S/step: {shaped_per_step:6.3f} | "
            f"Loss: {loss_val:.4f} | Eps: {epsilon:.3f} | "
            f"Success(50): {s_rate:.1f}% | Attach(50): {a_rate:.1f}%"
        )