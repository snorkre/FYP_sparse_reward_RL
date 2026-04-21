from __future__ import annotations
import argparse

import os
import csv
import gymnasium as gym
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from dqn import DQNAgent, DQNConfig
from utils import save_reward_plot, ensure_dir 


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """
    Linearly decay epsilon for epsilon-greedy exploration.
    
    Encourages exploration in early training and gradually shifts towards exploitation as training progresses."""
    if step >= decay_steps:
        return eps_end
    t = step / float(decay_steps)
    return eps_start + t * (eps_end - eps_start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Environment setup
    env = gym.make(args.env)
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # Ensure compatibility with DQN (continuous states, discrete actions)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    # Use GPU if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize hyperparameters
    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer=2_000,
        target_update_every=1_000,
        grad_clip_norm=10.0,
    )

    # Initialize DQN agent (with Double DQN) 
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        cfg=cfg,
        seed=args.seed,
    )

    # Experience replay buffer 
    buffer = ReplayBuffer(capacity=cfg.buffer_size, seed=args.seed)

    # Epsilon-greedy exploration schedule
    eps_start, eps_end = 1.0, 0.05
    decay_steps = 20_000

    rewards = []
    global_step = 0 # Track total steps for epsilon decay and target network updates

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        # Run one full episode
        while not done:
            # Compute exploration rate
            eps = linear_epsilon(global_step, eps_start, eps_end, decay_steps)

            # Select action using epsilon-greedy policy
            action = agent.act(obs.astype(np.float32), eps=eps)

            # Interact with environment
            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Store transition in replay buffer
            buffer.push(
                obs.astype(np.float32),
                int(action),
                float(reward),
                obs2.astype(np.float32),
                done,
            )
            
            # Move to next state
            obs = obs2
            ep_reward += float(reward)

            # Update global step counter
            global_step += 1
            agent.step_count = global_step # Used for periodic target network updates

            # Start training only after sufficient data is collected
            if len(buffer) >= cfg.min_buffer:
                batch = buffer.sample(cfg.batch_size)
                agent.train_step(batch)
                agent.update_target_if_needed()

        rewards.append(ep_reward)

        # Logging for monitoring Learning progress
        if (ep + 1) % 10 == 0:
            avg10 = np.mean(rewards[-10:])
            print(f"ep={ep+1:4d} reward={ep_reward:7.2f} avg10={avg10:7.2f}")

    env.close()

    # Save trained model
    os.makedirs("results/models", exist_ok = True)
    model_path = f"results/models/{args.env}_dqn_ddqn_seed{args.seed}.pt"
    ensure_dir(os.path.dirname(model_path))
    torch.save(agent.q.state_dict(), model_path)
    print("saved model:", model_path)

    # Save reward history for analysis
    csv_path = f"results/rewards/{args.env}_dqn_ddqn_seed{args.seed}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(rewards))
    print("saved rewards:", csv_path)

    # Save reward curve plot (used in evaluation section)
    out_plot = f"results/plots/{args.env}_dqn_ddqn_seed{args.seed}.png"
    save_reward_plot(rewards, out_plot)
    print("Saved plot:", out_plot)


if __name__ == "__main__":
    main()
