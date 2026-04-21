# Overcoming Sparse Rewards in Reinforcement Learning: A Comparative Analysis of Curriculum Learning and Hindsight Experience Replay

## Project Overview

This project investigates and compares reinforcement learning approaches for solving the `LunarLander-v3` environment from the Gymnasium library. Specifically, it evaluates:
* **Deep Q-Network (DQN) / Double DQN (DDQN)** (Baseline)
* **Curriculum Learning** (Progressive difficulty)
* **Hindsight Experience Replay (HER)**

The goal is to analyse performance under sparse reward conditions and assess learning stability and sample efficiency across multiple random seeds.

## Methods Summary
* **DQN/DDQN:** The baseline deep reinforcement learning approach using experience replay and target networks.
* **Curriculum Learning:** A manual approach that gradually increases task difficulty (modifying gravity and wind) to stabilise learning.
* **HER:** An algorithmic approach that re-labels failed experiences as successful hindsight goals to densify the reward signal.

## Installation

Ensure you have Python 3.8+ installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/snorkre/FYP_sparse_reward_RL 
cd FYP_sparse_reward_RL
pip install -r requirements.txt
```

## How to Run Experiments

All experiments are reproducible using fixed random seeds (0, 1, and 2). This project is designed to run locally without reliance on Google Colab. 

### 1. DQN + DDQN Baseline
```bash
python src/train.py --env LunarLander-v3 --episodes 1000 --seed 0
python src/train.py --env LunarLander-v3 --episodes 1000 --seed 1
python src/train.py --env LunarLander-v3 --episodes 1000 --seed 2
```

### 2. Curriculum Learning
```bash
python src/train_curriculum.py --episodes 1000 --seed 0
python src/train_curriculum.py --episodes 1000 --seed 1
python src/train_curriculum.py --episodes 1000 --seed 2
```

### 3. Hindsight Experience Replay (HER)
```bash
python src/train_her.py --episodes 1000 --seed 0
python src/train_her.py --episodes 1000 --seed 1
python src/train_her.py --episodes 1000 --seed 2
```

## Evaluation

The project includes evaluation scripts for statistical comparison, reward curve analysis, and statistical significance testing (Welch's t-test).

To generate the comparison plots and compute summary statistics:
```bash
python src/evaluate.py
python src/t_test.py
```

## Repository Structure

All outputs are generated directly into the local `results/` directory.

```text
FYP_sparse_reward_RL/
├── src/  
|   ├── curriculum_wrapper.py
|   ├── dqn.py
|   ├── evaluate.py
|   ├── her_replay_buffer.py
|   ├── replay_buffer.py             
│   ├── t_test.py
│   ├── train_curriculum.py
│   ├── train_her.py
│   ├── train.py
│   └── utils.py
├── results/              # Auto-generated during training
│   ├── plots/            # Training curves and comparison graphs
│   ├── rewards/              # Episode rewards and logged metrics
│   └── models/           # Saved trained agent weights (.pt files)
├── requirements.txt      # Project dependencies
└── README.md
```
## Author
    Saniska Dangol