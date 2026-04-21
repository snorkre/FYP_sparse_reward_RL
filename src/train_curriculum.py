from __future__ import annotations
import argparse 
import os
import csv

import gymnasium as gym
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from dqn import DQNAgent, DQNConfig
from curriculum_wrapper import CurriculumWrapper
from utils import save_reward_plot, ensure_dir

def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """
    Linearly decay epsilon over time.
    Higher epsilon encourages exploration in early training, while lower epsilon promotes exploitation as training progresses."""
    if step >= decay_steps:
        return eps_end
    t = step/ float(decay_steps)
    return eps_start + t * (eps_end - eps_start)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    # Environment setup
    # CurriculumWrapper dynamically adjusts task difficulty
    env = CurriculumWrapper(env_id=args.env, start_stage=args.start_stage, window=10, verbose=True)

    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # Ensure compatibility with DQN assumptions
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance (env.action_space, gym.spaces.Discrete)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Agent and configs
    # Shared configuration across all curriculum stages
    cfg = DQNConfig(
        gamma= 0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer=2_000,
        target_update_every=1_000,
        grad_clip_norm=10.0,
    )

    # Single agent trained across all stages (knowledge transfer via weights)
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        cfg=cfg,
        seed=args.seed,
    )                 

    # Replay buffer shared across curriculum stages
    buffer = ReplayBuffer(capacity=cfg.buffer_size, seed=args.seed)

    # Exploration setup
    eps_start, eps_end = 1.0, 0.05
    decay_steps = 20_000
    

    rewards = []
    stages_log = [] # track which stage each episode was in 
    global_step = 0 # used for epsilon decay and target updates

    # Training loop
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        # Record current stage at start of episode
        ep_stage = env.current_stage

        while not done:
            # Epsilon-greedy action selection
            eps = linear_epsilon(global_step, eps_start, eps_end, decay_steps)
            action = agent.act(obs.astype(np.float32), eps=eps)

            # Environment interaction
            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Store transition for off-policy learning
            buffer.push(
                obs.astype(np.float32),
                int(action),
                float(reward),
                obs2.astype(np.float32),
                done,
            )

            obs = obs2
            ep_reward += float(reward)

            # Global step count for epsilon decay and target network updates
            global_step += 1
            agent.step_count = global_step
            
            # Train only after sufficient exploration data is collected
            if len(buffer) >= cfg.min_buffer:
                batch = buffer.sample(cfg.batch_size)
                agent.train_step(batch)
                agent.update_target_if_needed()

        rewards.append(ep_reward)
        stages_log.append(ep_stage)
        
        # Logging training progress and curriculum stage
        if (ep + 1) % 10 == 0:
            avg10 = np.mean(rewards[-10:])
            print(f"ep={ep+1:4d} stage={env.current_stage} reward={ep_reward:7.2f} avg10={avg10:7.2f}")

    env.close()
    
    # Save model
    model_path = f"results/models/{args.env}_curriculum_seed{args.seed}.pt"
    ensure_dir(os.path.dirname(model_path))
    torch.save(agent.q.state_dict(), model_path)
    print("Saved model:", model_path)

    # Save rewards
    csv_path = f"results/rewards/{args.env}_curriculum_seed{args.seed}.csv"
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "stages"])
        for i, (r, s) in enumerate(zip(rewards, stages_log)):
            writer.writerow([i, r, s])
    print("saved rewards:", csv_path)

    # Save plot
    out_plot = f"results/plots/{args.env}_curriculum_seed{args.seed}.png"
    save_reward_plot(rewards, out_plot)
    print("saved plot:", out_plot)


if __name__ == "__main__":
    main()
 