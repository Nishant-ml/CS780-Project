# OBELIX: Autonomous Warehouse Robot under Partial Observability

An end-to-end Reinforcement Learning (RL) framework designed for the **OBELIX warehouse robot** to autonomously locate, attach to, and push target boxes to boundary zones under severe partial observability (POMDP)[cite: 3]. Developed as a capstone project for **CS780 at IIT Kanpur**[cite: 3].

---

## 📹 Demonstration & Competition

- **Video Demonstration:** [YouTube Project Walkthrough](https://youtu.be/Fz3YwjtmdYO)[cite: 3]
- **Codabench Benchmark Results:**
  - **Final Leaderboard Rank (Test Phase):** 87[cite: 3]
  - **Level 1 (Static Target):** 61[cite: 3]
  - **Level 2 (Blinking Target):** 97[cite: 3]
  - **Level 3 (Moving + Blinking Target):** 82[cite: 3]

---

## 📌 Problem Formulation

The robot operates in a continuous arena under a **Partially Observable Markov Decision Process (POMDP)** with wall obstacles and target ambiguity[cite: 3]:

- **Observation Space:** 18-dimensional binary vector received from local sonar and infrared sensor arrays[cite: 3].
- **Action Space:** 5 discrete navigation and manipulation actions[cite: 3].
- **Task Hierarchy:** `Find` $\to$ `Attach` $\to$ `Push` $\to$ `Unwedge`[cite: 3].
- **Environmental Complexity:**
  - **Level 1:** Static box[cite: 3].
  - **Level 2:** Intermittently visible / blinking box (temporal inconsistency)[cite: 3].
  - **Level 3:** Moving + blinking box (target interception challenge)[cite: 3].
  - **Obstacles:** Static walls sharing identical sensor signature patterns with the target box[cite: 3].

---

## 💡 Key Challenges & Algorithm Evolution

### 1. The Degenerate Policy Problem (Baselines)
Traditional value-based approaches (**Q-learning**, **SARSA**, **Double DQN**) failed to generalize, converging to degenerate **rotation-dominated policies**[cite: 3]. Due to severe collision penalties near boundaries and walls, unguided exploration caused agents to spin in place rather than move forward[cite: 3].

### 2. Deep Recurrent Q-Networks (DRQN)
Integrated an LSTM recurrent memory layer ($h_t = \text{LSTM}(x_t, h_{t-1})$) to maintain belief states over past observations during blinking intervals[cite: 3]. While effective with walls, training stability remained sensitive to sequence lengths and burn-in periods[cite: 3].

### 3. Soft Actor-Critic (SAC) + Hybrid Controller (Final Approach)
Adopted an entropy-regularized Actor-Critic framework:

$$\mathcal{J}(\pi) = \mathbb{E}_{(s, a) \sim \pi} \left[ Q(s, a) - \alpha \log \pi(a\vert{}s) \right]$$

Combined with structured heuristic controllers for deterministic edge-case recovery[cite: 3]:
- **Sweep Exploration:** Activates systematic rotational sweeping when all sensory inputs are zero ($o_t = \mathbf{0}$)[cite: 3].
- **Anti-Stuck / Unwedging:** Triggers recovery maneuvers when persistent non-progress states are detected[cite: 3].

---

## 🎯 Reward Shaping Design

To prevent policy collapse and guide dense learning without altering the optimal goal policy, the shaped reward $r'$ is defined as[cite: 3]:

$$r' = -0.01 + 0.05 \cdot \mathbb{I}_{\text{forward}} + 0.1 \cdot \mathbb{I}_{\text{sensor\_active}} - 0.5 \cdot \mathbb{I}_{\text{stuck}} + R_{\text{goal}}$$

where the terminal completion bonus encourages fast execution[cite: 3]:

$$R_{\text{goal}} = 1000 + 200 \cdot \left( \frac{T - t}{T} \right)$$

---

## 📊 Benchmark & Evaluation Results

Evaluated across **25 unseen randomized evaluation seeds** across all difficulty configurations[cite: 3]:

### Final Test Phase Performance (Cumulative Reward)

| Model | Difficulty Level | With Walls | No Walls |
| :--- | :--- | :---: | :---: |
| **DRQN** | Level 1 (Static)[cite: 3] | -17,661.0[cite: 3] | -2,085.7[cite: 3] |
| | Level 2 (Blinking)[cite: 3] | -17,909.3[cite: 3] | -3,612.5[cite: 3] |
| | Level 3 (Moving + Blinking)[cite: 3] | -6,308.9[cite: 3] | -4,042.1[cite: 3] |
| **SAC (Ours)** | Level 1 (Static)[cite: 3] | **-16,153.6**[cite: 3] | **-1,110.9**[cite: 3] |
| | Level 2 (Blinking)[cite: 3] | **-17,232.4**[cite: 3] | -3,642.9[cite: 3] |
| | Level 3 (Moving + Blinking)[cite: 3] | -10,567.7[cite: 3] | **-3,606.8**[cite: 3] |

### Overall Averages

| Model | With Walls (Mean) | No Walls (Mean) | Weighted Reward |
| :--- | :---: | :---: | :---: |
| **DRQN** | **-13,959.7**[cite: 3] | -3,246.8[cite: 3] | **-9,674.5**[cite: 3] |
| **SAC** | -14,651.2[cite: 3] | **-2,786.9**[cite: 3] | -9,905.5[cite: 3] |

- **Empirical Success Rates:** SAC achieves **~100%** task success in static environments, **~80%** in blinking scenarios (Level 2), and **~100%** in moving + blinking settings (Level 3)[cite: 3].
- **Core Takeaway:** Structured exploration and reward design have a significantly larger impact on policy stability in POMDPs than neural network memory complexity alone[cite: 3].

---

## ⚙️ Hyperparameters

| Parameter | SAC[cite: 3] | DRQN[cite: 3] |
| :--- | :---: | :---: |
| **Learning Rate** | $3 \times 10^{-4}$[cite: 3] | $5 \times 10^{-4}$[cite: 3] |
| **Batch Size** | 64[cite: 3] | 32[cite: 3] |
| **Discount Factor ($\gamma$)** | 0.99[cite: 3] | 0.97[cite: 3] |
| **Replay Buffer Size** | 10,000[cite: 3] | 10,000[cite: 3] |
| **Target Smoothing Coefficient ($\tau$)** | 0.01[cite: 3] | Periodic (20 steps)[cite: 3] |
| **Entropy Coefficient ($\alpha$)** | 0.2[cite: 3] | — |
| **Exploration Strategy** | Maximum Entropy Stochastic[cite: 3] | $\epsilon$-greedy ($0.75 \to 0.05$)[cite: 3] |
| **Sequence Length / Burn-in** | — | 32 / 8[cite: 3] |

---

## 📁 Repository Structure

```text
├── models/
│   ├── sac_agent.py          # Soft Actor-Critic actor & critic network implementations
│   ├── drqn_agent.py         # Deep Recurrent Q-Network with LSTM cell
│   └── baselines.py          # Tabular Q-learning, SARSA, and DDQN baselines
├── controllers/
│   └── exploration.py        # Sweep exploration and anti-stuck heuristic controllers
├── utils/
│   ├── reward_shaping.py     # Custom reward shaping wrappers
│   └── replay_buffer.py      # Standard and sequential replay buffers
├── train.py                  # Phased curriculum training pipeline
├── evaluation.py             # Multi-seed test evaluation benchmark
├── report.pdf                # Full capstone project report
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
