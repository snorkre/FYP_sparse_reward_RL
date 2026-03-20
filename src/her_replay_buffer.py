from __future__ import annotations
import numpy as np
from collections import deque
import random

# HER REPLAY BUFFER

class HERReplayBuffer:
    
    def __init__(
            self,
            capacity:int,
            her_ratio: int = 4,
            goal: list[float] | None = None,
            seed: int = 0,
    ) -> None:
            self.capacity = capacity
            self.her_ratio = her_ratio
            self.goal = np.array(goal if goal is not None else [ 0.0, 0.0], dtype=np.float32)
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

            # main buffer stores goal- conditioned transition
            self.buffer: deque[tuple] = deque (maxlen=capacity)
            
            # episode buffer stores raw transitions for current episode
            self._episode: list[tuple] = []

    def __len__(self) -> int:
        return len(self.buffer)
        
        # episode management

    def store_transition(self, s, a, r, s2, done) -> None:
        # store a raw transition during an episode (before HER relabeling)
        self._episode.append((
            np.array(s, dtype=np.float32),
            int(a),
            float(r),
            np.array(s2, dtype=np.float32),
            bool(done),
        ))

    def finish_episode(self) -> None:
        # called at episode end. Applies HER and pushes all transition to buffer
        if len(self._episode) == 0:
            return
            
        # achieved goal = lander position at the end of the episode 
        final_obs = self._episode[-1][3]
        achieved_goal = final_obs[:2].copy()

        for i, (s, a, r, s2, done) in enumerate(self._episode):
            # original transition with real goal 
            s_gc = self._concat_goal(s, self.goal)
            s2_gc = self._concat_goal(s2, self.goal)
            self.buffer.append((s_gc, a, r, s2_gc, done))

            # HER relablled transitions
            # use "future" strategy: pick random future states as hindsight goals
            future_indices = list(range(i, len(self._episode)))
            sampled = self.np_rng.choice(
                future_indices,
                size=min(self.her_ratio, len(future_indices)),
                replace=False,
            )

            for idx in sampled:
                hindsight_goal = self._episode[idx][3][:2].copy() # future x, y

                s_h = self._concat_goal(s, hindsight_goal)
                s2_h = self._concat_goal(s2, hindsight_goal)

                # recompute reward: +1 if lander is close to hindsight goal

                r_h = self._compute_her_reward(s2, hindsight_goal)
                done_h = bool(r_h > 0)

                self.buffer.append((s_h, a, r_h, s2_h, done_h))
                    
        self._episode.clear()

    # sampling

    def sample(self, batch_size: int):
        # sample a random batch of goal conditioned transitions
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )
        
        # helper

    def _concat_goal(self, obs: np.ndarray, goal:np.ndarray) -> np.ndarray:
        # concatenate observation with goal to create goal conditioned obs
        return np.concatenate([obs, goal], axis=0)
        
    def _compute_her_reward(self, obs: np.ndarray, goal:np.ndarray) -> float:
        # sparse reward: +1.0 if lander is within threshold of hindsight goal, else 0.
        pos = obs[:2]
        dist = float(np.linalg.norm(pos - goal))
        return -dist
        


                



