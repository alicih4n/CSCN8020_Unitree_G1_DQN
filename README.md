# Deep Q-Network Control of the Unitree G1 Left Elbow

**Student Full Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Assignment:** Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow (Assignment 3)  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College, Ontario, Canada  
**GitHub Repository URL:** [https://github.com/alicih4n/CSCN8020_Assignment3](https://github.com/alicih4n/CSCN8020_Assignment3)  
**Cloneable .git URL:** `https://github.com/alicih4n/CSCN8020_Assignment3.git`  
**Operating Environment:** macOS (Apple Silicon M1) | Python 3.13 | PyTorch 2.10.0 with MPS GPU acceleration

---

## 1. Short Project Summary

This project implements a complete, student-written PyTorch Deep Q-Network (DQN) agent to control the left elbow joint (`left_elbow_joint`) of the fixed-base Unitree G1 humanoid robot in MuJoCo and Gymnasium. Rather than relying on turnkey libraries like Stable-Baselines3, all core RL components—Q-Network, Replay Buffer, epsilon-greedy action selection, target network synchronization, Huber loss TD optimization, and greedy evaluation—were implemented from scratch. The agent underwent a controlled parameter study on exploration decay ($\text{decay}=0.995$ vs $0.985$) and achieved a **100% success rate (20/20 episodes)** across four benchmark goals (`-0.8, -0.4, +0.4, +0.8` rad), outperforming the rule-based baseline in step speed (**19.8 steps** vs 24.0) and goal precision (**0.0039 rad** vs 0.0122 rad error).

---

## 2. Environment & Dependency Setup

### Step 1: Create and Activate Virtual Environment
```bash
# Clone repository
git clone https://github.com/alicih4n/CSCN8020_Assignment3.git
cd CSCN8020_Assignment3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Fetch External Unitree Assets & Generate Fixed-Base Model
```bash
# Fetch external Unitree MuJoCo models
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco

# Generate fixed-base G1 XML scene assets
python src/create_fixed_base_g1.py
```

---

## 3. Execution Commands

### Run the Jupyter Notebook
```bash
jupyter notebook Unitree_MuJoCo_G1_Primer_Workshop.ipynb
```

### Run Self-Validation Smoke Test
```bash
PYTHONPATH=src python src/dqn/smoke_test.py
```

### Train the DQN Policy (Headless Parameter Study)
```bash
PYTHONPATH=src python src/dqn/train_dqn.py
```
*Trains Configuration A (decay=0.995) and Configuration B (decay=0.985) for 600 episodes headlessly, saving checkpoints to `models/` and logging plots to `results/`.*

### Evaluate Policy & Load Checkpoint
```bash
PYTHONPATH=src python src/dqn/evaluate_dqn.py
```
*Evaluates both DQN configurations and the rule-based baseline greedily ($\epsilon=0.0$) over 20 benchmark episodes, prints evaluation comparison tables, and saves the best model to `models/selected_dqn.pt`.*

### Render Policy in 3D MuJoCo Viewer
* **macOS (Apple Silicon M1/M2/M3)**:
  ```bash
  # Step-by-step evaluation renderer (as recommended in assignment specifications):
  PYTHONPATH=src mjpython src/dqn/render_dqn_policy.py

  # Optional continuous back-and-forth swing demo:
  PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py
  ```
* **Linux / Windows WSL**:
  ```bash
  PYTHONPATH=src python src/dqn/render_dqn_policy.py
  ```

---

## 4. Student-Written DQN Implementation Details

The DQN implementation is located in `src/dqn/`:
* **`q_network.py` (`QNetwork`)**: PyTorch `nn.Module` with 4 inputs ($[\theta, \dot{\theta}, g, g - \theta]$), two hidden layers of 64 units with ReLU activations, Kaiming Normal weight initialization, and 3 unconstrained linear outputs corresponding to actions (Decrease, Hold, Increase target).
* **`replay_buffer.py` (`ReplayBuffer`)**: 50,000 capacity circular transition memory using `collections.deque`, sampling mini-batches of size 64 as PyTorch tensors on the active device.
* **`agent.py` (`DQNAgent`)**: Online and Target Q-networks with hard updates every 250 optimization steps, Huber loss (Smooth L1), gradient clipping at 1.0, $\epsilon$-greedy action selection, and checkpoint saving/loading.
* **`train_dqn.py`**: Headless training script recording rewards, success rates, loss, and epsilon decay over 600 episodes.
* **`evaluate_dqn.py`**: Benchmark evaluation script running 20 episodes with $\epsilon=0.0$ across goals `-0.8, -0.4, +0.4, +0.8` rad.

---

## 5. Major Repository Files Overview

```text
├── README.md                          # Main project documentation & execution guide
├── SUBMISSION_INFO.md                 # One-page portal submission reference
├── MAC_M1_RUN_GUIDE.md                # Native macOS execution notes
├── requirements.txt                   # Dependency list (torch, gymnasium, mujoco, numpy, etc.)
├── .gitignore                         # Git exclusion rules per Section 16.5
├── Unitree_MuJoCo_G1_Primer_Workshop.ipynb # Completed assignment Jupyter Notebook
├── models/                            # Saved PyTorch model checkpoints
│   ├── dqn_config_a.pt                # Trained weights for Config A (decay=0.995)
│   ├── dqn_config_b.pt                # Trained weights for Config B (decay=0.985)
│   └── selected_dqn.pt                # Best model checkpoint for evaluation
├── report/                            # Technical report documentation
│   ├── DQN_Assignment_Report.md       # 13-section Academic Technical Report
│   └── DQN_Assignment_Report.pdf      # PDF version of technical report
├── results/                           # Evaluation CSVs and training visualization plots
│   ├── config_a/                      # Training plots and metrics for Config A
│   ├── config_b/                      # Training plots and metrics for Config B
│   ├── epsilon_decay_reward_comparison.png
│   ├── epsilon_decay_success_comparison.png
│   ├── evaluation_success_by_angle.png
│   └── rule_based_evaluation_metrics.csv
└── src/
    ├── g1_rl/                         # Gymnasium environment wrapper
    │   └── g1_elbow_env.py
    └── dqn/                           # Student-written PyTorch DQN implementation
        ├── __init__.py
        ├── q_network.py
        ├── replay_buffer.py
        ├── agent.py
        ├── train_dqn.py
        ├── evaluate_dqn.py
        ├── render_dqn_policy.py
        ├── render_dqn_continuous.py
        └── smoke_test.py
```
