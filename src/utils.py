from __future__ import annotations
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window: int = 20) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="valid")



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_reward_plot(rewards: List[float], out_path: str, window: int = 20) -> None:
    ensure_dir(os.path.dirname(out_path))
    ma = moving_average(rewards, window=window)

    plt.figure()
    plt.plot(rewards, label = "reward")
    if len(ma) > 0:
        x = list(range(window - 1, window -1 +len(ma)))
        plt.plot(x, ma, label = "moving avg")
    plt.title("Episode rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi =160)
    plt.close()
    



 