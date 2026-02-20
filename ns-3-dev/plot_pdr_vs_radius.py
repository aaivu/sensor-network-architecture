#!/usr/bin/env python3
"""PDR vs Radius comparison plot."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ADR_KEYS    = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS  = {"no_adr": "No ADR", "classical_adr": "Classical",
               "ddqn_adr": "DDQN-PER", "ppo_adr": "PPO", "marl_adr": "MARL"}
ADR_COLORS  = {"no_adr": "#e74c3c", "classical_adr": "#e67e22",
               "ddqn_adr": "#2980b9", "ppo_adr": "#27ae60", "marl_adr": "#8e44ad"}
ADR_MARKERS = {"no_adr": "o", "classical_adr": "s",
               "ddqn_adr": "^", "ppo_adr": "D", "marl_adr": "P"}

# TC groupings to plot side by side
GROUPS = {
    "Baseline — no interference\n(TC 1: 30m, TC 2: 100m, TC 3: 500m)": [1, 2, 3],
    "Building-size sweep — WiFi scales with radius\n(TC 4–8: 50→500m)": [4, 5, 6, 7, 8],
}


def load_pdr(tc_ids):
    rows = []
    for tc_id in tc_ids:
        df = pd.read_csv(f"test_case_results/tc{tc_id:02d}_summary.csv")
        pdr = df[df["metric"] == "pdr_pct"]
        for _, r in pdr.iterrows():
            rows.append({
                "tc_id":   tc_id,
                "radius":  int(r["radius"]),
                "adr_key": r["adr_key"],
                "mean":    float(r["mean"]),
                "ci_lo":   float(r["ci95_low"]),
                "ci_hi":   float(r["ci95_high"]),
                "wifi":    int(r["wifiInterferers"]),
                "nDevices":int(r["nDevices"]),
            })
    return pd.DataFrame(rows)


plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("PDR vs Indoor Environment Radius — ADR Method Comparison",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (group_name, tc_ids) in zip(axes, GROUPS.items()):
    df = load_pdr(tc_ids)
    radii = sorted(df["radius"].unique())

    for key in ADR_KEYS:
        sub = df[df["adr_key"] == key].sort_values("radius")
        x   = sub["radius"].values
        y   = sub["mean"].values
        lo  = sub["ci_lo"].values
        hi  = sub["ci_hi"].values
        ax.plot(x, y,
                color=ADR_COLORS[key], marker=ADR_MARKERS[key],
                linewidth=2.2, markersize=8,
                label=ADR_LABELS[key], zorder=4)
        ax.fill_between(x, lo, hi,
                        color=ADR_COLORS[key], alpha=0.12, zorder=2)

    # TC annotations below x-axis ticks
    y_floor = max(0, df["mean"].min() - 12)
    for tc_id in tc_ids:
        subrow = df[(df["tc_id"] == tc_id) & (df["adr_key"] == "ddqn_adr")]
        if subrow.empty:
            continue
        r = subrow.iloc[0]
        wifi_txt = f"WiFi={r['wifi']}" if r["wifi"] else "no WiFi"
        note = f"TC{tc_id}\n{wifi_txt}\nn={r['nDevices']}"
        ax.text(r["radius"], y_floor, note,
                ha="center", va="top", fontsize=6.5, color="#555555")

    ax.set_title(group_name, fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel("Gateway Radius (m)", fontsize=10)
    ax.set_ylabel("Packet Delivery Ratio (%)", fontsize=10)
    ax.set_xticks(radii)
    ax.set_xticklabels([f"{r}m" for r in radii])
    ymin = max(0, df[["mean", "ci_lo"]].min().min() - 8)
    ymax = min(102, df[["mean", "ci_hi"]].max().max() + 3)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

fig.tight_layout()
out = Path("test_case_results/plots/pdr_vs_radius.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved → {out}")
