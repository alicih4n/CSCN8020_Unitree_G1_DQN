# CSCN8020 Assignment 3 Submission Info

**Student Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Assignment Title:** Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College, Ontario, Canada  
**GitHub Repository:** [https://github.com/alicih4n/CSCN8020_Assignment3.git](https://github.com/alicih4n/CSCN8020_Assignment3.git)  
**YouTube Video Demonstration:** [https://www.youtube.com/watch?v=KJPv8v0fPEY](https://www.youtube.com/watch?v=KJPv8v0fPEY)

---

### Project Summary (~100 words)
This project implements a complete, student-written PyTorch Deep Q-Network (DQN) agent to control the left elbow joint of the Unitree G1 humanoid robot in MuJoCo and Gymnasium. Rather than using turnkey libraries, all RL components—Q-Network (4-input, 3-output), Replay Buffer (50,000 capacity), target network synchronization, Huber loss TD optimization, and greedy evaluation—were built from scratch. Hardware accelerated on Apple Silicon M1 (MPS), the agent completed an exploration parameter study ($\text{decay}=0.995$ vs $0.985$) and achieved a **100% success rate (20/20 episodes)** across four benchmark goals (`-0.8, -0.4, +0.4, +0.8` rad), outperforming the rule-based baseline in step speed (**19.8 steps** vs 24.0) and goal precision (**0.0039 rad** vs 0.0122 rad error).

---

### Quick Run Commands

```bash
# 1. Run Self-Validation Smoke Test
PYTHONPATH=src python src/dqn/smoke_test.py

# 2. Run Greedy Benchmark Evaluation (Prints comparison table)
PYTHONPATH=src python src/dqn/evaluate_dqn.py

# 3. Launch Interactive 3D MuJoCo Simulation (Step-by-step or continuous demo)
PYTHONPATH=src mjpython src/dqn/render_dqn_policy.py
PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py
```
