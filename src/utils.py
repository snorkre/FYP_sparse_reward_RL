from __future__ import annotations
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window: int = 20) -> np.ndarray:
    """
    Computes a moving average over a sequence of rewards.
    Used to smooth out reward signals in RL, making overall learning trends easier to interpret."""
    arr = np.array(x, dtype=np.float32)

    # If not enough data points, return raw values
    if len(arr) < window:
        return arr
    
    # Create uniform averaging kernel 
    kernel = np.ones(window, dtype=np.float32) / window

    # Apply convolution to compute moving average
    return np.convolve(arr, kernel, mode="valid")



def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists before saving files.
    Prevents errors when trying to save models, rewards, or plots to non-existent directories."""
    os.makedirs(path, exist_ok=True)

def save_reward_plot(rewards: List[float], out_path: str, window: int = 20) -> None:
    """
    Saves a plot of episode rewards over time, with an optional moving average for trend visualization.
    The moving average helps to smooth out the reward curve, making it easier to see overall learning progress."""
    ensure_dir(os.path.dirname(out_path))

    # Compute moving average of rewards for smoother visualization
    ma = moving_average(rewards, window=window)

    plt.figure()

    # Plot moving average (smoothed reward trend)
    plt.plot(rewards, label = "reward")
    if len(ma) > 0:
        x = list(range(window - 1, window -1 +len(ma)))
        plt.plot(x, ma, label = "moving avg")
    plt.title("Episode rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()

    # Save figure for later analysis/report inclusion
    plt.savefig(out_path, dpi =160)
    plt.close()
    



 