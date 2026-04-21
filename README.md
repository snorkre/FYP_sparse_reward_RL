Reinforcement Learning for Sparse Reward Environments

Overview:
This project investigates the performance of Deep Q-Network (DQN) variants in sparse reward environments, using LunarLander-v3 from Gymnasium.

Three approaches are implemented and compared:
1) Baseline: DQN with Double DQN (DDQN)
2) Curriculum Learning (CL)
3) Hindsight Experience Replay (HER)

The goal is to evaluate how these methods improve learning efficiency, stability, and final performance.

Methods:
1) Baseline (DQN + DDQN)
    A standard Deep Q-Network enhanced with Double DQN to reduce overestimation bias.
2) Curriculum Learning
    The environment difficulty is progressively increased:
        Stage 1: Low gravity (easy)
        Stage 2: Medium gravity
        Stage 3: High gravity with wind (hard)
    Progression is performance-based, using a rolling average reward threshold.

3) Hindsight Experience Replay (HER)
    Transitions are relabelled with alternative goals to convert failed episodes into useful learning signals.
        i) Goal-conditioned state: [state, goal]
        ii) Future strategy used for relabelling

Project Structure
FYP/
│
├── src/
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── her_replay_buffer.py
│   ├── curriculum_wrapper.py
│   ├── train.py
│   ├── train_cl.py
│   ├── train_her.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── models/
│   ├── rewards/
│   └── plots/
│
└── README.md


Installation:
Install required dependencies:
pip install numpy torch gymnasium matplotlib pandas


Training:
Baseline (DQN + DDQN)
python src/train.py --env LunarLander-v3 --episodes 600

Curriculum Learning:
python src/train_cl.py --env LunarLander-v3 --episodes 1000

HER:
python src/train_her.py --env LunarLander-v3 --episodes 1000


Evaluation:
    Run evaluation across multiple seeds:
    python src/evaluate.py

This will:
    Load reward logs from all methods
    Compute mean and standard deviation
    Generate comparison plots
    Output a summary table

Results:
Results are saved in:
    results/rewards/   # CSV logs
    results/plots/     # Training curves
    results/models/    # Trained models

Evaluation includes:
    Mean ± standard deviation across seeds
    Moving average smoothing
    Sample efficiency comparison (episodes to reach threshold)

Key Findings:
    Curriculum Learning improves early training stability and learning speed
    HER improves sample efficiency by learning from failed trajectories
    Baseline DQN provides a strong reference but struggles in sparse settings

Notes:
    Experiments are run with multiple random seeds for robustness
    All methods use the same network architecture and hyperparameters for fairness
    Results may vary slightly due to stochastic training

Author
Saniska Dangol

