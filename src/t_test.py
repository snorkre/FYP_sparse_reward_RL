"""
Statistical significance testing for FYP results.
Welch's t-test (unequal variance) on final mean reward (last 50 episodes).
"""

import numpy as np
from scipy import stats

# Final mean reward (last 50 episodes) per seed — from evaluate.py output
baseline   = np.array([209.4, 198.4, 130.6])
curriculum = np.array([80.2, 91.3, -97.8])
her        = np.array([-89.0, 17.3, -131.1])

comparisons = [
    ("Baseline vs Curriculum", baseline, curriculum),
    ("Baseline vs HER", baseline, her),
    ("Curriculum vs HER", curriculum, her),
]

print("=" * 70)
print("Welch's t-test (two-sample, unequal variance)")
print(f"n = 3 seeds per group")
print("=" * 70)
print(f"{'Comparison':<28} {'t':>8} {'p-value':>10} {'Cohen d':>10} {'Sig?':>6}")
print("-" * 70)

for name, a, b in comparisons:
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    cohens_d = (a.mean() - b.mean()) / pooled_std
    sig = "*" if p_val < 0.05 else ""
    print(f"{name:<28} {t_stat:>8.3f} {p_val:>10.4f} {cohens_d:>10.2f} {sig:>6}")

print("-" * 70)
print("* Significant at p < 0.05")
print()
print("Interpretation:")
print("- Cohen's d > 0.8 = large effect size")
print("- All comparisons show large to very large effects (d > 1.2)")
print("- Baseline vs HER reaches significance despite n=3 (massive effect)")
print("- Baseline vs Curriculum does not reach significance (insufficient power)")