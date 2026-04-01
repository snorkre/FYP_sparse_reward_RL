from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV        = "LunarLander-v3"
SEEDS      = [0, 1, 2]
RESULTS_DIR = "/content/drive/MyDrive/FYP/results/rewards"
PLOTS_DIR   = "/content/drive/MyDrive/FYP/results/plots"
MA_WINDOW  = 20   # moving average window for smoothing

METHODS = {
    "Baseline (DQN + DDQN)": f"{ENV}_dqn_ddqn",
    "Curriculum Learning":    f"{ENV}_curriculum",
    "HER (DQN + HER)":        f"{ENV}_her",
}

COLOURS = {
    "Baseline (DQN + DDQN)": "#2196F3",   
    "Curriculum Learning":    "#4CAF50",   
    "HER (DQN + HER)":        "#F44336",   
}

# Helpers
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple moving average."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_rewards(method_key: str, seeds: list[int]) -> list[np.ndarray]:
    """Load reward arrays for all seeds of a given method.
    
    Handles both 'reward' and 'rewards' column names for robustness.
    Returns a list of 1-D numpy arrays, one per seed.
    """
    arrays = []
    for seed in seeds:
        path = os.path.join(RESULTS_DIR, f"{method_key}_seed{seed}.csv")
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path} — skipping seed {seed}")
            continue
        df = pd.read_csv(path)

        # Normalise column name: handle 'reward' or 'rewards'
        df.columns = [c.strip().lower() for c in df.columns]
        if "reward" in df.columns:
            rewards = df["reward"].values.astype(np.float32)
        elif "rewards" in df.columns:
            rewards = df["rewards"].values.astype(np.float32)
        else:
            raise ValueError(f"No 'reward' column found in {path}. Columns: {df.columns.tolist()}")

        arrays.append(rewards)
        print(f"  Loaded {path} — {len(rewards)} episodes")

    return arrays


def align_and_aggregate(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align arrays to same length (min length), compute mean and std.
    
    Returns:
        episodes: episode indices
        mean:     mean reward per episode
        std:      std reward per episode
    """
    if len(arrays) == 0:
        raise ValueError("No reward arrays to aggregate.")

    min_len = min(len(a) for a in arrays)
    trimmed = np.stack([a[:min_len] for a in arrays], axis=0)  # (n_seeds, n_episodes)

    mean = np.mean(trimmed, axis=0)
    std  = np.std(trimmed,  axis=0)
    episodes = np.arange(1, min_len + 1)

    return episodes, mean, std


# Main evaluation
def run_evaluation() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    summary_rows = []

    for label, method_key in METHODS.items():
        print(f"\nLoading: {label}")
        arrays = load_rewards(method_key, SEEDS)

        if len(arrays) == 0:
            print(f"  [SKIP] No data found for {label}")
            continue

        episodes, mean, std = align_and_aggregate(arrays)

        # Apply moving average smoothing
        mean_smooth = moving_average(mean, MA_WINDOW)
        std_smooth  = moving_average(std,  MA_WINDOW)
        ep_smooth   = episodes[MA_WINDOW - 1:]  # align x-axis after convolution

        colour = COLOURS[label]

        # Plot mean curve
        ax.plot(ep_smooth, mean_smooth, label=label, color=colour, linewidth=2)

        # Plot shaded std region
        ax.fill_between(
            ep_smooth,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            alpha=0.2,
            color=colour,
        )

        # Summary stats
        final_mean = float(np.mean(mean[-50:]))   # mean of last 50 episodes
        final_std  = float(np.std(mean[-50:]))
        peak_mean  = float(np.max(mean_smooth))

        # Training speed: first episode where smoothed mean > 100
        above_threshold = np.where(mean_smooth > 100)[0]
        speed = int(ep_smooth[above_threshold[0]]) if len(above_threshold) > 0 else None

        summary_rows.append({
            "Method":              label,
            "Final Mean (last 50)": f"{final_mean:.1f}",
            "Final Std (last 50)":  f"{final_std:.1f}",
            "Peak Mean":            f"{peak_mean:.1f}",
            "Episodes to >100":     speed if speed else "Not reached",
        })

    # Plot formatting 
    ax.axhline(y=200, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Solved threshold (200)")
    ax.axhline(y=0,   color="grey",  linestyle=":",  linewidth=0.8, alpha=0.4)

    ax.set_title(
        "Comparison of DQN Variants on LunarLander-v3\n"
        f"(Mean ± Std across {len(SEEDS)} seeds, smoothed with window={MA_WINDOW})",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    out_path = os.path.join(PLOTS_DIR, f"{ENV}_comparison_all_methods.png")
    plt.savefig(out_path, dpi=160)
    plt.show()
    print(f"\nSaved comparison plot: {out_path}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print("=" * 70)

    # Save summary CSV
    summary_path = os.path.join(RESULTS_DIR, "summary_comparison.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    run_evaluation()