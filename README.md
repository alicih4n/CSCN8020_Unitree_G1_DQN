# Deep Q-Network Control of the Unitree G1 Left Elbow

**Developer / Student Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College, Ontario, Canada  
**Repository:** [https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN.git](https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN.git)

---

### Project Purpose & Objective

The primary objective of this project is to build and train a **student-written PyTorch Deep Q-Network (DQN)** agent to control the left elbow joint (`left_elbow_joint`) of the **Unitree G1 humanoid robot** in a physics simulation environment (MuJoCo and Gymnasium). 

Rather than relying on hand-crafted rules or turnkey libraries (such as Stable-Baselines3), this project implements every reinforcement learning component from scratch:
- **Neural Q-Network (`QNetwork`)**: A 4-input observation to 3-output action-value estimate architecture with Kaiming Normal weight initialization.
- **Experience Replay Buffer (`ReplayBuffer`)**: A 50,000-transition circular memory buffer for off-policy mini-batch sampling.
- **Target Network & Bellman Target Optimization**: Online and target Q-networks synchronized every 250 steps, utilizing Huber loss and gradient clipping.
- **Parameter Study**: Controlled comparison between **Configuration A** ($\text{epsilon decay} = 0.995$) and **Configuration B** ($\text{epsilon decay} = 0.985$).
- **Benchmark Evaluation**: Greedy policy evaluation ($\epsilon = 0.0$) across 20 benchmark episodes (`-0.8, -0.4, +0.4, +0.8` rad target angles), proving a 100% success rate and outperforming the rule-based baseline.

---

## Project Overview

This project develops a reproducible instructional workflow for working with the Unitree G1 humanoid robot in MuJoCo.

The workshop begins with environment preparation and model inspection, then progresses through:

1. MuJoCo installation and viewer validation
2. Unitree G1 model inspection
3. Joint, actuator, sensor, `qpos`, and `qvel` analysis
4. Single-joint proportional-derivative control
5. Whole-body joint stabilization
6. Gravity and bias-force compensation
7. Creation of a course-owned fixed-base G1 model
8. CSV logging and deterministic validation
9. Construction of a custom Gymnasium environment
10. Rule-based environment validation
11. Optional interactive visualization before reinforcement learning
12. Deep Q-Network (DQN) policy training, parameter study, and evaluation

---

## Educational Purpose

The workshop is intended for college-level students studying:

- Reinforcement learning
- Robotics
- Machine learning
- Simulation
- Control systems
- Artificial intelligence
- Python programming

Students are expected to understand the relationship between:

```text
High-level discrete action
        ↓
Internal joint-position target
        ↓
PD controller
        ↓
Bias-force compensation
        ↓
Actuator torque
        ↓
Simulated physical movement
```

The workshop separates conventional low-level control from high-level reinforcement-learning decisions. This allows students to focus on the reinforcement-learning problem without first needing to solve full humanoid balance, locomotion, inverse kinematics, and whole-body torque control.

---

## Learning Outcomes

After completing the workshop, students should be able to:

1. Explain the role of MuJoCo in robot simulation.
2. Distinguish between bodies, joints, actuators, sensors, and degrees of freedom.
3. Explain the purpose of `qpos` and `qvel`.
4. Load and inspect the Unitree G1 29-DOF model.
5. Identify a joint and actuator by name.
6. Read joint position and velocity data.
7. Apply bounded actuator torque.
8. Implement a proportional-derivative controller.
9. Explain the effect of gravity and bias forces.
10. Create a fixed-base instructional robot model.
11. Record simulation results in CSV format.
12. Build a Gymnasium-compatible environment.
13. Explain the difference between `terminated` and `truncated`.
14. Define observations, actions, rewards, and success conditions.
15. Validate an environment with a rule-based policy.
16. Confirm deterministic simulation behaviour.
17. Train a DQN agent using PyTorch to solve the multi-goal control problem.
18. Compare the learned policy to the rule-based baseline and perform hyperparameter tuning.

---

## Current Project Status

| Milestone | Status |
|---|---|
| WSL 2 and Ubuntu setup | Complete |
| macOS M1 Native Setup | Complete |
| MuJoCo installation | Complete |
| MuJoCo viewer test | Complete |
| Unitree G1 repository integration | Complete |
| G1 model inspection | Complete |
| Fixed-base G1 generation | Complete |
| Left-elbow PD control | Complete |
| Whole-body joint stabilization | Complete |
| Bias-force compensation | Complete |
| CSV logging | Complete |
| Deterministic controller validation | Complete |
| Gymnasium environment | Complete |
| Gymnasium environment checker | Complete |
| Rule-based validation policy | Complete |
| Five-run determinism test | Complete |
| Optional rendered validation | Complete |
| Interactive camera-preparation demo | Complete |
| Student-written DQN | Complete |
| Physical G1 deployment | Future work |

---

## Requirements

- macOS (Apple Silicon native) or Windows 11 with WSL 2 (Ubuntu 24.04)
- Python 3.12 or 3.13
- PyTorch (with MPS support for Apple Silicon GPU acceleration)

## Setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The workshop also uses the official Unitree MuJoCo repository as an external dependency:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
```

Ensure you run the fixed-base model generator once:
```bash
python src/create_fixed_base_g1.py
```

---

## Deep Q-Network (DQN) Control

The DQN implementation is located in the `src/dqn/` package. It contains a PyTorch-based Q-Network, Replay Buffer, and DQNAgent, as well as scripts for training, evaluation, and rendering.

### 1. Training the Agent
To train the agent on both Configuration A (decay = 0.995) and Configuration B (decay = 0.985):
```bash
PYTHONPATH=src python src/dqn/train_dqn.py
```
This script runs training headlessly and logs metrics to `results/config_a/` and `results/config_b/`. Checkpoints are saved under `models/`.

### 2. Evaluating the Policy
To run greedy evaluation ($\epsilon = 0.0$) of both configurations and the rule-based policy over the 20 benchmark episodes:
```bash
PYTHONPATH=src python src/dqn/evaluate_dqn.py
```
This generates comparison tables in the terminal, saves metrics CSV logs, and copies the best-performing model to `models/selected_dqn.pt`.

### 3. Rendering and Recording Video
To visually demonstrate the trained DQN policy:
* **macOS (Mac M1/M2/M3):**
  ```bash
  PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py
  ```
* **Linux / WSL:**
  ```bash
  PYTHONPATH=src python src/dqn/render_dqn_continuous.py
  ```
*(When the window opens, the robot will automatically alternate between goal targets in 3D MuJoCo).*
