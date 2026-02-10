#!/usr/bin/env python3
"""Generate a publication-quality grouped bar chart of PDR across all experiments."""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──
experiments = [
    "10 dev, 300m\nWiFi",
    "10 dev, 300m\nWiFi+LTE",
    "20 dev, 500m",
    "10 dev\nMARL dom.",
    "20 dev, 100m\nultimate",
    "30 dev\n60s period",
    "30 dev\nalt. run",
]

no_adr      = [ 8.80, 16.37,  1.55,  8.86, 12.46, 38.69, 12.08]
classical   = [40.64, 28.59, 31.76, 40.49, 25.54, 73.18, 46.68]
ddqn        = [69.44, 72.35, 37.46, 47.83, 91.83, 95.22, 94.58]
ppo         = [61.22, 69.89, 38.99, 49.62, 94.58, 99.78, 88.01]
marl        = [83.73, 86.76, 51.34, 68.55, 99.65, 91.02, 90.60]

methods = ["No ADR", "Classical ADR", "DDQN-PER", "PPO", "MARL"]
data    = [no_adr, classical, ddqn, ppo, marl]

# ── Colours (blue palette, light → dark) ──
colors = ["#D6E6F5", "#A8C8E8", "#6FA0D2", "#3B7BBE", "#1A4D8F"]
edge   = "#0E2F5A"

# ── Plot ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

n_exp = len(experiments)
n_met = len(methods)
x = np.arange(n_exp)
total_width = 0.78
bar_w = total_width / n_met

fig, ax = plt.subplots(figsize=(10, 5))

for i, (method, vals, col) in enumerate(zip(methods, data, colors)):
    offset = (i - n_met / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w * 0.92,
                  label=method, color=col, edgecolor=edge, linewidth=0.5)
    # Value labels
    for bar, v in zip(bars, vals):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.0,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", color="#222222")

ax.set_ylabel("Packet Delivery Rate (%)")
ax.set_xlabel("Experiment Scenario")
ax.set_title("PDR Comparison Across All Experimental Scenarios")
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=5, framealpha=0.9, fontsize=8.5)

fig.tight_layout()
fig.savefig("graphs/pdr_all_experiments.eps", format="eps", bbox_inches="tight")
fig.savefig("graphs/pdr_all_experiments.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅ Saved graphs/pdr_all_experiments.eps")
print("✅ Saved graphs/pdr_all_experiments.png")
