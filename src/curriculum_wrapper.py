from __future__ import annotations
import numpy as np
import gymnasium as gym

# Stage definitions
# Each stage increases environment difficulty by adjusting gravity and wind parameters.

STAGE_CONFIGS = {
    1: dict(gravity=-4.0, enable_wind=False, wind_power=0.0, turbulence_power=0.0),
    2: dict(gravity=-7.0, enable_wind=False, wind_power=0.0, turbulence_power=0.0),
    3: dict(gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5),


}

# Performance thresholds required to adcance stages
STAGE_THRESHOLDS = {
    1: 0.0, # advance from stage 1 to 2, need to avg any positive reward (i.e. land at all)
    2: 50.0, # advance from stage 2 to 3, need to avg 50 reward (i.e. land softly and near the pad)
}

class CurriculumWrapper(gym.Wrapper):
    """
    Implement manual curriculum learning by progressively increasing task difficulty based on agent performance.
    The agent is trained continuously, and knowledge is transferred implicitly via learned network weights"""
    def __init__(
            self,
            env_id: str = "LunarLander-v3",
            start_stage: int = 1,
            window: int = 10,
            verbose: bool = True,
    ) -> None:
        env = self._make_env(env_id, start_stage)
        super().__init__(env)
        
        self.env_id = env_id
        self.stage = start_stage
        self.window = window
        self.verbose = verbose

        # Track recent rewards for performance-based progression
        self._recent_rewards: list[float] = []
        self._current_ep_reward: float = 0.0

    # INTERNAL HELPER
    def _make_env(self, env_id:str, stage: int) -> gym.Env:
        """ 
        Create a LunarLander environment configured for a given stage.
        """
        cfg = STAGE_CONFIGS[stage]
        return gym.make(env_id, **cfg)
    
    def _advance_stage(self) -> None:
        """
        Advances to the next curriculum stage by creating a new environment with increased difficulty."""
        self.stage += 1
        new_env = self._make_env(self.env_id, self.stage)
        self.env.close()
        self.env = new_env

        # Reset performance tracking for new stage
        self._recent_rewards.clear()

        if self.verbose:
            cfg = STAGE_CONFIGS[self.stage]
            print(f"\n[Curriculum] *** Advanced to Stage {self.stage} ***")
            print(f"[Curriculum] gravity = {cfg['gravity']},"
                  f"wind={cfg['enable_wind']},"
                  f"wind_power={cfg['wind_power']}\n")
            
    def _maybe_advance_stage(self) -> None:
        """ 
        Check if rolling avg exceeds threshold and advance stage if met.
         Uses rolling average to ensure stable progression decisions """
        if self.stage >= 3:
            return
        if len(self._recent_rewards) < self.window:
            return
        
        avg = float(np.mean(self._recent_rewards))
        threshold = STAGE_THRESHOLDS[self.stage]
        
        if avg > threshold:
            self._advance_stage()

    def reset (self, **kwargs):
        """
        Resets the environment and initializes episode reward tracking."""
        self._current_ep_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Executes one step in the environment, updates reward tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._current_ep_reward += float(reward)

        # At episode end, update rolling avg and maybe advance stage
        if terminated or truncated:
            self._recent_rewards.append(self._current_ep_reward)
            if len(self._recent_rewards) > self.window:
                self._recent_rewards.pop(0)
            self._maybe_advance_stage()
        
        # Log current stage in info for analysis
        info["stage"] = self.stage

        return obs, reward, terminated, truncated, info

    @property
    def current_stage(self) -> int:
        return self.stage
    
    def rolling_avg(self) -> float | None:
        """
        Returns current rolling avg reward, or None if window not fully yet.
        """
        if len(self._recent_rewards) < self.window:
            return None
        return float(np.mean(self._recent_rewards))


