from __future__ import annotations
import argparse
import os 
import csv

import gymnasium as gym
import numpy as np
import torch

from dqn import DQNAgent, DQNConfig
from her_replay_buffer import HERReplayBuffer
from utils import save_reward_plot, ensure_dir

def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    t = step / float(decay_steps)
    return eps_start + t* (eps_end - eps_start)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--her_ratio", type=int, default=4,
                        help="HER hindsight transitions per real transition")
    args = parser.parse_args()

    # environment 
    env = gym.make(args.env)
    obs, info = env.reset(seed = args.seed)
    env.action_space.seed(args.seed)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

     # goal conditioned obs: original 8-dims obs + 2-dim goal = 10-dim
    base_obs_dim = int(np.prod(env.observation_space.shape))
    goal_dim = 2
    obs_dim = base_obs_dim + goal_dim
    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"obs dim (goal-conditioned): {obs_dim}, Actions: {n_actions}")

    # agent (same DQN + DDQN as arm A, just wider input)
    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=100_000,
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

    # HER replay buffer
    buffer = HERReplayBuffer(
        capacity=cfg.buffer_size,
        her_ratio=args.her_ratio,
        goal=[0.0, 0.0], # landing pad centre
        seed=args.seed,
    )

    # fixed goal (landing pad)
    goal = np.array([0.0, 0.0], dtype=np.float32)

    eps_start, eps_end = 1.0, 0.05
    decay_steps = 20_000

    rewards = []
    global_step = 0
    
    # tranning loop
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # append goal to observation
            obs_gc = np.concatenate([obs.astype(np.float32), goal])

            eps = linear_epsilon(global_step, eps_start, eps_end, decay_steps)
            action = agent.act(obs_gc, eps=eps)

            obs2, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # store raw transition in episode buffer (hHER relables at episode end)
            buffer.store_transition(
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

            # train if enough samples in buffer
            if len(buffer) >= cfg.min_buffer:
                batch = buffer.sample(cfg.batch_size)
                agent.train_step(batch)
                agent.update_target_if_needed()
                

        # end of episode: apply HER relabelling
        buffer.finish_episode()
        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg10 = np.mean(rewards[-10:])
            print(f"ep={ep+1:4d} reward={ep_reward:7.2f} avg10={avg10:7.2f} "
                  f"buffer={len(buffer):6d}")
            
    env.close()

    # save model 
    model_path = f"results/models/{args.env}_her_seed{args.seed}.pt"
    ensure_dir(os.path.dirname(model_path))
    torch.save(agent.q.state_dict(), model_path)
    print("saved model:", model_path)

    #save rewards CSV
    csv_path = f"results/rewards/{args.env}_her_seed{args.seed}.csv"
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(rewards))
    print("saved rewards:", csv_path)

    # save plot 
    out_plot = f"results/plots/{args.env}_her_seed{args.seed}.png"
    save_reward_plot(rewards, out_plot)
    print("Saved plot:", out_plot)
 
 
if __name__ == "__main__":
    main()
 

    

                        

