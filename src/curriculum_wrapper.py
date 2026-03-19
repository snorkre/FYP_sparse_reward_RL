from __future__ import annotations
import numpy as np
import gymnasium as gym

# STAGE DEFINITION- EACH STAGE IS DIFFERENT LUNARLANDER ENVIRONMENT

STAGE_CONFIGS = {
    1: dict(gravity=-4.0, enable_wind=False, wind_power=0.0, turbulence_power=0.0),
    2: dict(gravity=-7.0, enable_wind=False, wind_power=0.0, turbulence_power=0.0),
    3: dict(gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5),


}
STAGE_THRESHOLDS = {
    1: 0.0, 
    2: 50.0,
}

# CURRICULUM WRAPPER

class CurriculumWrapper(gym.Wrapper):
    def __init__(
            self,
            env_id: str = "LunarLander-v3",
            start_stage: int = 1,
            window: int = 10,
            verbose: bool = True,
    ) -> None:
        env = self._make_env(env_id, start_stage)
        super().__init_(env)
        
        self.env_id = env_id
        self.stage = start_stage
        self.window = window
        self.verbose = verbose

        self._recent_rewards: list[float] = []
        self._current_ep_reward: float = 0.0

    # INTERNAL HELPER
    def _make_env(self, env_id:str, stage: int) -> gym.Env:
        #create a LunarLander environment with stage approprirate parameters.
        cfg = STAGE_CONFIGS[stage]
        return gym.make(env_id, **cfg)
    
    def _advance_stage(self) -> None:
        # swap the underlying environment for the nest stage.
        self.stage += 1
        new_env = self._make_env(self.env_id, self.stage)
        self.env.close()
        self.env = new_env
        self._recent_rewards.clear()
        if self.verbose:
            cfg = STAGE_CONFIGS[self.stage]
            print(f"\n[Curriculum] *** Advanced to Stage {self.stage} ***")
            print(f"[Curriculum] gravity = {cfg['gravity']},"
                  f"wind={cfg['enable_wind']},"
                  f"wind_power={cfg['wind_power']}\n")
            
    def _maybe_advance_stage(self) -> None:
        # check if rolling avg exceeds threshold and advance if so
        if self.stage >= 3:
            return
        if len(self._recent_rewards) < self.window:
            return
        
        avg = float(np.mean(self._recent_rewards))
        threshold = STAGE_THRESHOLDS[self.stage]
        
        if avg > threshold:
            self._advance_stage()



    # CORE GYM INTERFACE

    def reset (self, **kwargs):
        self._current_ep_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._current_ep_reward += float(reward)

        # at episode end, update rolling avg and maybe advance stage
        if terminated or truncated:
            self._recent_rewards.append(self._current_ep_reward)
            if len(self._recent_rewards) > self.window:
                self._recent_rewards.pop(0)
            self._maybe_advance_stage()

        info["stage"] = self.stage

        return obs, reward, terminated, truncated, info
    
    # UTILITY
    @property
    def current_stage(self) -> int:
        return self.stage
    
    def rolling_avg(self) -> float | None:
        # returns current rolling avg reward, or None if window not fully yet.
        if len(self._recent_rewards) < self.window:
            return None
        return float(np.mean(self._recent_rewards))


