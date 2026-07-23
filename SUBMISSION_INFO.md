# Assignment Submission Information

**Student Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Assignment:** Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College, Ontario, Canada  
**GitHub Repository:** [https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN](https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN)

---

## Executive Summary of Completed Work

This project implements a complete, course-compliant Deep Q-Network (DQN) agent in PyTorch to control the left elbow joint of the Unitree G1 humanoid robot in MuJoCo and Gymnasium.

Key technical achievements:
1. **PyTorch Q-Network Implementation**: Designed a 4-input, 3-output neural network (`QNetwork`), Replay Buffer (`ReplayBuffer`), and DQN agent (`DQNAgent`) with Kaiming initialization, target network synchronization (hard update every 250 steps), gradient clipping (1.0), and Huber loss.
2. **Apple Silicon Hardware Acceleration**: Configured device selection to automatically utilize **Apple Silicon MPS (Metal Performance Shaders)** on macOS M1/M2/M3, completing 600 training episodes in under 3 minutes.
3. **Parameter Study**: Conducted a controlled comparison between **Configuration A** ($\text{epsilon decay} = 0.995$) and **Configuration B** ($\text{epsilon decay} = 0.985$).
4. **Greedy Benchmark Evaluation**: Evaluated both trained models greedily ($\epsilon = 0.0$) against the hand-written rule-based baseline across 20 benchmark episodes (4 target angles: `-0.8, -0.4, +0.4, +0.8` rad, 5 episodes each).
5. **Outstanding Performance**: Reached a **100% Success Rate (20/20 episodes)**, exceeding the required 80% threshold. The DQN agent outperformed the rule-based baseline in step efficiency (**19.8 steps** vs **24.0 steps**) and joint goal precision (**0.0039 rad** vs **0.0122 rad** error).
6. **Documentation & Deliverables**: Created full CSV metrics, visualization plots, an academic technical report ([`report/DQN_Assignment_Report.md`](file:///Users/alicihanozdemir/Documents/SpringClassAssignments/CSCN8020/Unitree_MuJoCo_G1_Primer_Workshop/report/DQN_Assignment_Report.md)), and continuous 3D rendering scripts.

---

## Step-by-Step Instructions: How to Run the Project

### 1. Environment Setup

Clone the repository and install the dependencies:
```bash
git clone https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN.git
cd CSCN8020_Unitree_G1_DQN

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required Python dependencies
pip install -r requirements.txt
```

Fetch external Unitree MuJoCo models (if not already present):
```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
```

Generate the fixed-base G1 robot XML model:
```bash
python src/create_fixed_base_g1.py
```

---

### 2. Run the Self-Validation Smoke Test

To verify that PyTorch, device selection (MPS/CUDA/CPU), Replay Buffer sampling, and Q-Network forward/backward passes are working:
```bash
PYTHONPATH=src python src/dqn/smoke_test.py
```
**Expected Output:** `=== ALL SMOKE TESTS PASSED SUCCESSFULLY ===`

---

### 3. Train the DQN Agent (Parameter Study)

To train both Configuration A ($\text{decay}=0.995$) and Configuration B ($\text{decay}=0.985$) headlessly for 600 episodes:
```bash
PYTHONPATH=src python src/dqn/train_dqn.py
```
**What to Expect:**
* Headless training completes in under 3 minutes on Apple Silicon M1 (MPS).
* Trained model checkpoints saved to `models/dqn_config_a.pt` and `models/dqn_config_b.pt`.
* CSV metric logs saved to `results/config_a/` and `results/config_b/`.
* Reward, rolling success rate, loss, and epsilon decay plots generated in `results/`.

---

### 4. Evaluate the Trained Policy against the Baseline

To run greedy evaluation ($\epsilon = 0.0$) over 20 benchmark episodes:
```bash
PYTHONPATH=src python src/dqn/evaluate_dqn.py
```
**What to Expect:**
* Prints formatted Markdown comparison tables showing Success Count (20/20), Success Rate (100%), Mean Reward, Episode Steps, and Final Absolute Error for the Rule-Based policy, Config A, and Config B.
* Generates `results/evaluation_success_by_angle.png`.
* Automatically copies the superior model (Config A) to `models/selected_dqn.pt`.

---

### 5. View the Live 3D Simulation (Rendered Policy)

To visually demonstrate the trained DQN agent controlling the G1 left elbow in 3D MuJoCo:

* **macOS (Mac M1/M2/M3)**:
  ```bash
  PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py
  ```
* **Linux / Windows WSL**:
  ```bash
  PYTHONPATH=src python src/dqn/render_dqn_continuous.py
  ```

**What to Expect:**
* A 3D MuJoCo window opens showing the fixed-base Unitree G1 robot.
* The agent automatically flexes the left elbow joint (located on the right side of your screen when viewing frontally) to `-0.8` rad, holds it with high precision, prints `[Goal Reached!]`, and alternates to `+0.8` rad in a continuous loop.

---

## File Structure Overview

```text
├── SUBMISSION_INFO.md                 # Student & project submission info
├── README.md                          # Repository documentation & instructions
├── MAC_M1_RUN_GUIDE.md                # Native macOS M1 setup & execution guide
├── requirements.txt                   # Python package dependencies
├── models/
│   ├── dqn_config_a.pt                # Checkpoint for Configuration A (decay=0.995)
│   ├── dqn_config_b.pt                # Checkpoint for Configuration B (decay=0.985)
│   └── selected_dqn.pt                # Selected best model checkpoint
├── report/
│   └── DQN_Assignment_Report.md       # Academic Technical Report (13 Chapters)
├── results/
│   ├── config_a/                      # Metrics and training plots for Config A
│   ├── config_b/                      # Metrics and training plots for Config B
│   ├── epsilon_decay_reward_comparison.png
│   ├── epsilon_decay_success_comparison.png
│   ├── evaluation_success_by_angle.png
│   └── rule_based_evaluation_metrics.csv
└── src/
    ├── g1_rl/
    │   └── g1_elbow_env.py            # Gymnasium environment wrapper
    └── dqn/
        ├── __init__.py                # Package initializer
        ├── q_network.py               # PyTorch Q-Network module
        ├── replay_buffer.py           # Replay Memory buffer
        ├── agent.py                   # DQNAgent class
        ├── train_dqn.py               # Training loop & plot generator
        ├── evaluate_dqn.py            # Benchmark evaluation script
        ├── render_dqn_policy.py       # Step-by-step rendering script
        ├── render_dqn_continuous.py   # Continuous back-and-forth 3D demo
        └── smoke_test.py              # Compilation verification script
```
