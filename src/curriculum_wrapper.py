from __future__ import annotations
import numpy as np
import gymnasium as gym

# REWARD SHAPING HELPERS

def dense_reward(base_reward: float, obs: np.ndarray, terminated: bool) -> float:
    # stage 1 : standard LunarLander dense reward, unchanged.
    return base_reward

def partial_reward(base_reward: float, obs:np.ndarray, terminated: bool) -> float:
    if terminated and abs(base_reward) >= 90:
        return base_reward
    return 0.0

def sparse_reward(base_reward: float, obs:np.ndarray, terminated: bool) -> float:
    #stage 3: fully sparse: +100 land, -100 crash 0 otherwise.
    if terminated and abs(base_reward) >= 90:
        return base_reward
    return 0.0

SHAPERS =   {
    1: dense_reward,
    2: partial_reward,
    3: sparse_reward,
}

STAGE_THRESHOLDS = {
    1: 0.0, 
    2: 100.0,
}

# CURRICULUM WRAPPER

class CurriculumWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            start_stage: int = 1,
            window: int = 10,
            verbose: bool = True,
    ) -> None:
        super().__init__(env)
        assert start_stage in (1,2,3), "start_stage must be 1, 2 or 3"
        self.stage = start_stage
        self.window = window
        self.verbose = verbose

        self._recent_rewards: list[float] = []
        self._current_ep_reward: float = 0.0


    # CORE GYM INTERFACE

    def reset (self, **kwargs):
        self._current_ep_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # apply stage- appropriate reward shaping
        shaped = SHAPERS[self.stage](float(reward), obs, bool(terminated))

        self._current_ep_reward += shaped

        # at episode end, update rolling avg and maybe advance stage
        if terminated or truncated:
            self._recent_rewards.append(self._current_ep_reward)
            if len(self._recent_rewards) > self.window:
                self._recent_rewards.pop(0)
            self._maybe_advance_stage()

        info["original_reward"] = float(reward)
        info["stage"] = self.stage

        return obs, shaped, terminated, truncated, info
    
    # STAGE MANAGEMENT
    
    def _maybe_advance_stage(self) -> None:
        # advance to the next stage if the rolling avf exceeds the threshold.
        if self.stage >= 3:
            return
        if len(self._recent_rewards) < self.window:
            return # not enough data yet
        
        avg = float(np.mean(self._recent_rewards))
        threshold = STAGE_THRESHOLDS[self.stage]

        if avg > threshold:
            self.stage += 1
            self._recent_rewards.clear() # reset window for fresh tracking
            if self.verbose:
                print(f"[Curriculum] Advanced to Stage {self.stage}"
                      f"(avg{self.window}={avg:.1f} > {threshold:.1f})")
                
    # UTILITY
    @property
    def current_stage(self) -> int:
        return self.stage
    
    def rolling_avg(self) -> float | None:
        # returns current rolling avg reward, or None if window not fully yet.
        if len(self._recent_rewards) < self.window:
            return None
        return float(np.mean(self._recent_rewards))


