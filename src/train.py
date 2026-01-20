from __future__ import annotations
import argparse

import gymnasium as gym
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from dqn import DQNAgent, DQNConfig
from utils import save_reward_plot


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
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

    env = gym.make(args.env)
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer=2_000,
        target_update_every=1_000,
        grad_clip_norm=10.0,
    )

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        cfg=cfg,
        seed=args.seed,
    )

    buffer = ReplayBuffer(capacity=cfg.buffer_size, seed=args.seed)

    eps_start, eps_end = 1.0, 0.05
    decay_steps = 20_000

    rewards = []
    global_step = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            eps = linear_epsilon(global_step, eps_start, eps_end, decay_steps)
            action = agent.act(obs.astype(np.float32), eps=eps)

            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            buffer.push(
                obs.astype(np.float32),
                int(action),
                float(reward),
                obs2.astype(np.float32),
                done,
            )

            obs = obs2
            ep_reward += float(reward)
            global_step += 1
            agent.step_count = global_step

            if len(buffer) >= cfg.min_buffer:
                batch = buffer.sample(cfg.batch_size)
                agent.train_step(batch)
                agent.update_target_if_needed()

        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg10 = np.mean(rewards[-10:])
            print(f"ep={ep+1:4d} reward={ep_reward:7.2f} avg10={avg10:7.2f}")

    env.close()

    out_plot = f"results/plots/{args.env}_dqn_ddqn_seed{args.seed}.png"
    save_reward_plot(rewards, out_plot)
    print("Saved plot:", out_plot)


if __name__ == "__main__":
    main()
