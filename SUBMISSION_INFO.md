# CSCN8020 Assignment Submission Info

**Student Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Assignment:** Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow  
**Instructor:** Prof. Enrique Espinosa  
**GitHub Repository:** [https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN.git](https://github.com/alicih4n/CSCN8020_Unitree_G1_DQN.git)

---

### Project Summary
* **Objective**: Train a PyTorch Deep Q-Network (DQN) agent from scratch to control the Unitree G1 robot's left elbow joint across multi-goal targets in MuJoCo and Gymnasium.
* **Methodology**: Built a 4-input $\to$ 3-output Q-Network, Replay Buffer (50,000 capacity), and target network synchronization. Accelerated on Apple Silicon M1 (MPS). Conducted an exploration parameter study ($\text{decay}=0.995$ vs $0.985$).
* **Results**: Achieved a **100% Success Rate (20/20 episodes)** on greedy evaluation ($\epsilon=0.0$), outperforming the rule-based baseline in both step efficiency (**19.8 steps** vs 24.0) and goal precision (**0.0039 rad** vs 0.0122 rad error).

---

### Quick Run Commands

```bash
# 1. Run Self-Validation Smoke Test
PYTHONPATH=src python src/dqn/smoke_test.py

# 2. Run Greedy Benchmark Evaluation (Prints comparison table)
PYTHONPATH=src python src/dqn/evaluate_dqn.py

# 3. Launch Interactive 3D MuJoCo Simulation (Continuous elbow movement)
PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py
```
