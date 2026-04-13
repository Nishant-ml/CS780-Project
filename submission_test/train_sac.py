import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from obelix import OBELIX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

GAMMA = 0.99
LR = 3e-4
ALPHA = 0.2
TAU = 0.01

BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_BUFFER = 50
EPISODES = 2000

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

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

class EscapeController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.phase = "idle"
        self.turn_steps_done = 0
        self.turn_dir = 1
        self.attempts = 0
        self.consecutive_fails = 0
        self.last_obs = None

    def _sensor_guided_turn(self, obs):
        left_hits = int(obs[0]+obs[1]+obs[2]+obs[3])
        right_hits = int(obs[12]+obs[13]+obs[14]+obs[15])
        if left_hits < right_hits:
            return 1
        elif right_hits < left_hits:
            return -1
        else:
            return 1 if self.attempts % 2 == 1 else -1

    def _turn_steps_needed(self):
        if self.last_obs is not None:
            left_hits = int(self.last_obs[0]+self.last_obs[1]+self.last_obs[2]+self.last_obs[3])
            right_hits = int(self.last_obs[12]+self.last_obs[13]+self.last_obs[14]+self.last_obs[15])
            fwd_hits = int(sum(self.last_obs[4:12]))
            if left_hits > 0 and right_hits > 0 and fwd_hits > 0:
                return 4
        if self.consecutive_fails == 0:
            return 2
        elif self.consecutive_fails == 1:
            return 3
        else:
            return 4

    def update(self, stuck, obs):
        self.last_obs = obs[:18].copy()
        if self.active:
            if self.phase == "idle" and stuck:
                self.consecutive_fails += 1
                self.active = True
                self.phase = "turning"
                self.turn_steps_done = 0
                self.attempts += 1
                self.turn_dir = self._sensor_guided_turn(obs)
            elif self.phase == "idle" and not stuck:
                self.active = False
                self.consecutive_fails = 0
                self.attempts = 0
        else:
            if stuck:
                self.active = True
                self.phase = "turning"
                self.turn_steps_done = 0
                self.attempts += 1
                self.turn_dir = self._sensor_guided_turn(obs)

    def act(self):
        if not self.active:
            return None
        if self.phase == "turning":
            needed = self._turn_steps_needed()
            self.turn_steps_done += 1
            if self.turn_steps_done >= needed:
                self.phase = "probing"
                self.turn_steps_done = 0
            return "L45" if self.turn_dir == 1 else "R45"
        if self.phase == "probing":
            self.phase = "idle"
            return "FW"
        return "FW"

class SweepController:
    TURN_STEPS = 2
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = 1
        self.turning = False
        self.turn_steps_done = 0

    def act(self, stuck):
        if stuck and not self.turning:
            self.turning = True
            self.turn_steps_done = 0
            self.direction *= -1
        if self.turning:
            self.turn_steps_done += 1
            if self.turn_steps_done >= self.TURN_STEPS:
                self.turning = False
            return "L45" if self.direction == 1 else "R45"
        return "FW"

def shape_reward(obs, action_idx, raw_reward, done, env, step_t, max_steps):
    shaped = -0.01
    if action_idx == 2:
        shaped += 0.05
    if obs[:16].sum() > 0:
        shaped += 0.1
    if obs[17] == 1:
        shaped -= 0.5
    if done and env.enable_push and env._box_touches_boundary(env.box_center_x, env.box_center_y):
        efficiency = (max_steps - step_t) / max_steps
        shaped += 1000.0 + 200.0 * efficiency
    return shaped

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, episode):
        self.buffer.append(episode)

    def sample(self):
        batch = []
        while len(batch) < BATCH_SIZE:
            ep = random.choice(self.buffer)
            if len(ep) < 2:
                continue
            idx = random.randint(0, len(ep)-2)
            batch.append(ep[idx])
        s,a,r,ns,d = zip(*batch)
        return (
            torch.FloatTensor(s).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(ns).to(device),
            torch.FloatTensor(d).to(device),
        )
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,N_ACTIONS)
        )
    def forward(self,x):
        logits = self.net(x)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,N_ACTIONS)
        )
    def forward(self,x):
        return self.net(x)

actor = Actor().to(device)
critic1 = Critic().to(device)
critic2 = Critic().to(device)
target_critic1 = Critic().to(device)
target_critic2 = Critic().to(device)

target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=LR)
critic1_opt = optim.Adam(critic1.parameters(), lr=LR)
critic2_opt = optim.Adam(critic2.parameters(), lr=LR)

buffer = ReplayBuffer()

def select_action(state, escape_ctrl, sweep_ctrl, eval=False):
    stuck = state[17] == 1
    sensors_active = state[:16].sum() > 0

    escape_ctrl.update(stuck, state)
    esc = escape_ctrl.act()
    if esc is not None:
        return ACTIONS.index(esc)

    if not sensors_active:
        return ACTIONS.index(sweep_ctrl.act(stuck))

    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    probs = actor(state_t).detach().cpu().numpy()[0]

    if eval:
        return np.argmax(probs)
    return np.random.choice(N_ACTIONS, p=probs)

def train_step():
    s,a,r,ns,d = buffer.sample()

    with torch.no_grad():
        next_probs = actor(ns)
        next_log_probs = torch.log(next_probs.clamp(min=1e-8))
        q_next = torch.min(target_critic1(ns), target_critic2(ns))
        v_next = (next_probs*(q_next - ALPHA*next_log_probs)).sum(dim=1)
        target_q = r + GAMMA*(1-d)*v_next

    q1 = critic1(s).gather(1,a.unsqueeze(1)).squeeze()
    q2 = critic2(s).gather(1,a.unsqueeze(1)).squeeze()

    loss_q1 = nn.functional.mse_loss(q1, target_q)
    loss_q2 = nn.functional.mse_loss(q2, target_q)

    critic1_opt.zero_grad(); loss_q1.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(),5.0)
    critic1_opt.step()

    critic2_opt.zero_grad(); loss_q2.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(),5.0)
    critic2_opt.step()

    probs = actor(s)
    log_probs = torch.log(probs.clamp(min=1e-8))
    q_vals = torch.min(critic1(s), critic2(s))
    actor_loss = (probs*(ALPHA*log_probs - q_vals)).sum(dim=1).mean()

    actor_opt.zero_grad(); actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),5.0)
    actor_opt.step()

    for tp,p in zip(target_critic1.parameters(), critic1.parameters()):
        tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)
    for tp,p in zip(target_critic2.parameters(), critic2.parameters()):
        tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)

reward_history=[]
success_history=[]
best_success=0.0

for ep in range(EPISODES):

    env = OBELIX(max_steps=get_max_steps(ep), difficulty=2, wall_obstacles=False, seed=get_seed(ep)
                 ,scaling_factor=5)

    state = env.reset().astype(np.float32)

    escape_ctrl = EscapeController()
    sweep_ctrl = SweepController()

    episode=[]
    total_reward=0
    episode_success=False

    for t in range(env.max_steps):

        action_idx = select_action(state, escape_ctrl, sweep_ctrl)

        next_state, raw_reward, done = env.step(ACTIONS[action_idx], render=True)
        next_state = next_state.astype(np.float32)

        reward = shape_reward(next_state, action_idx, raw_reward, done, env, t, env.max_steps)

        episode.append((state, action_idx, reward, next_state, float(done)))

        if done and env.enable_push and env._box_touches_boundary(env.box_center_x, env.box_center_y):
            episode_success = True

        state = next_state
        total_reward += reward

        if done:
            break

    buffer.push(episode)

    if len(buffer) > MIN_BUFFER:
        train_step()

    reward_history.append(total_reward)
    success_history.append(episode_success)

    avg_success = np.mean(success_history[-50:]) if len(success_history)>=50 else 0

    torch.save(actor.state_dict(),"actor_latest.pth")

    if avg_success > best_success:
        best_success = avg_success
        torch.save(actor.state_dict(),"actor_best.pth")
        print(f"Saved best model: {avg_success*100:.1f}%")

    if ep % 10 == 0:
        print(f"Ep {ep} | Reward: {total_reward:.2f} | Success(50): {avg_success*100:.1f}%")