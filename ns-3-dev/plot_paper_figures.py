#!/usr/bin/env python3
"""
Paper figures (4 plots).
Run: venv/bin/python3 plot_paper_figures.py
"""

import csv, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# ── output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("graphs/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ADR order / labels ────────────────────────────────────────────────────────
ADR_KEYS = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS = {
    "no_adr":        "No ADR",
    "classical_adr": "Classical ADR",
    "ddqn_adr":      "DDQN-PER",
    "ppo_adr":       "PPO",
    "marl_adr":      "MARL",
}

# ── Sequential blues (bar charts) ─────────────────────────────────────────────
BAR_BLUES = {
    "no_adr":        "#c6dbef",
    "classical_adr": "#6baed6",
    "ddqn_adr":      "#3182bd",
    "ppo_adr":       "#1a588f",
    "marl_adr":      "#08306b",
}

# ── Blue-family palette for line charts ───────────────────────────────────────
# Five visually distinct shades + different line styles / markers
LINE_BLUES = {
    "no_adr":        "#9ecae1",   # light sky blue
    "classical_adr": "#4292c6",   # cornflower blue
    "ddqn_adr":      "#2171b5",   # medium blue
    "ppo_adr":       "#08519c",   # dark blue
    "marl_adr":      "#08306b",   # dark navy
}
LINE_LS = {
    "no_adr":        (0, (4, 2)),          # long dash
    "classical_adr": (0, (2, 1, 2, 1)),   # dash-dot-dash
    "ddqn_adr":      "-",                  # solid
    "ppo_adr":       "-.",                 # dash-dot
    "marl_adr":      "--",                 # dashed
}
LINE_MARKERS = {
    "no_adr":        "o",
    "classical_adr": "s",
    "ddqn_adr":      "^",
    "ppo_adr":       "D",
    "marl_adr":      "P",
}
LINE_LW = {
    "no_adr":        1.8,
    "classical_adr": 1.8,
    "ddqn_adr":      2.4,   # bold — top performer
    "ppo_adr":       2.0,
    "marl_adr":      2.4,   # bold — top performer
}

# ── global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "legend.framealpha":    0.95,
    "legend.edgecolor":     "#cccccc",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.alpha":           0.35,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.6,
    "axes.axisbelow":       True,
    "axes.facecolor":       "white",
    "figure.facecolor":     "white",
    "figure.dpi":           150,
    "lines.solid_capstyle": "round",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    for ext in ("png", "eps"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None,
                    format=ext)
        print(f"  saved {p}")
    plt.close(fig)


def load_grand():
    data = {}
    with open("test_case_results/grand_summary_pdr.csv") as f:
        for row in csv.DictReader(f):
            if row["metric"] != "pdr_pct":
                continue
            tc = int(row["tc_id"])
            data.setdefault(tc, {})[row["adr_key"]] = float(row["mean"])
    return data


def load_tc_metrics(tc_id: int):
    data = {}
    with open(f"test_case_results/tc{tc_id:02d}_summary.csv") as f:
        for row in csv.DictReader(f):
            data[(row["adr_key"], row["metric"])] = {
                "mean":   float(row["mean"]),
                "std":    float(row["std"]),
                "ci_low": float(row["ci95_low"]),
                "ci_hi":  float(row["ci95_high"]),
            }
    return data


def smooth_dense(x, y, n_pts=400):
    """Smooth spline for dense data (≥ 10 points)."""
    rng = max(float(np.ptp(y)), 2.0)
    s = len(x) * (rng ** 2) * 0.40
    try:
        spl = UnivariateSpline(x, y, k=3, s=s, ext=3)
    except Exception:
        spl = UnivariateSpline(x, y, k=1, s=s, ext=3)
    xs = np.linspace(x[0], x[-1], n_pts)
    ys = np.clip(spl(xs), 0, 102)
    return xs, ys


def load_sweep_pdr(n):
    frames = []
    for fp in sorted(glob.glob(f"sweep_results/n{n:03d}_r*_summary.csv")):
        df = pd.read_csv(fp)
        frames.append(df[df["metric"] == "pdr_pct"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hgrid_only(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.xaxis.grid(False)


def label_bars_smart(ax, bars, vals, base_pad=1.0, min_gap=7.5, fontsize=8.5):
    """Place value labels above bars; push up to avoid vertical collisions."""
    items = []
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        cx = bar.get_x() + bar.get_width() / 2
        items.append((cx, bar.get_height(), v))

    placed = []
    for cx, bh, v in items:
        y = bh + base_pad
        changed = True
        while changed:
            changed = False
            for pcx, py in placed:
                if abs(cx - pcx) < 0.50 and abs(y - py) < min_gap:
                    y = py + min_gap
                    changed = True
        placed.append((cx, y))
        ax.text(cx, y, f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color="#1a1a1a")


def bar_legend(ax, loc="upper left", ncol=5, bbox=None):
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor=BAR_BLUES[k], edgecolor="white",
                       linewidth=0.5, label=ADR_LABELS[k])
        for k in ADR_KEYS
    ]
    kw = dict(handles=handles, ncol=ncol, fontsize=10, frameon=True,
              framealpha=0.95, edgecolor="#cccccc",
              handlelength=1.4, handletextpad=0.5, columnspacing=0.9)
    if bbox:
        ax.legend(loc=loc, bbox_to_anchor=bbox, **kw)
    else:
        ax.legend(loc=loc, **kw)


def line_legend_blue(fig, y=0.0):
    handles = [
        mlines.Line2D([], [], color=LINE_BLUES[k],
                      linestyle=LINE_LS[k], linewidth=LINE_LW[k],
                      marker=LINE_MARKERS[k], markersize=7,
                      label=ADR_LABELS[k])
        for k in ADR_KEYS
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, y),
               fontsize=10, frameon=True,
               handlelength=2.6, handletextpad=0.6, columnspacing=1.2)


grand = load_grand()


# =============================================================================
# FIGURE 1 — PDR for 30 devices at 100 m (TC5, single scenario)
# =============================================================================
print("\n[Fig 1] PDR at 30 devices, 100 m — TC5")

tc5_pdr = [grand[5].get(a, np.nan) for a in ADR_KEYS]

fig, ax = plt.subplots(figsize=(7, 4.8))
bars = ax.bar(range(len(ADR_KEYS)), tc5_pdr,
              color=[BAR_BLUES[a] for a in ADR_KEYS],
              edgecolor="white", linewidth=0.4, width=0.55, zorder=3)
label_bars_smart(ax, bars, tc5_pdr, base_pad=1.0, min_gap=7.0, fontsize=10)

hgrid_only(ax)
ax.set_xticks(range(len(ADR_KEYS)))
ax.set_xticklabels([ADR_LABELS[a] for a in ADR_KEYS], fontsize=10.5)
ax.set_ylabel("Packet Delivery Rate (%)", labelpad=6)
ax.set_title("PDR at 30 Devices, 100 m Radius  (8 WiFi APs)")
ax.set_ylim(0, 115)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis="both", direction="out", length=3, width=0.7)
fig.tight_layout()
save_fig(fig, "fig1_pdr_30dev_100m")


# =============================================================================
# FIGURE 2 — Energy + PDR dual-axis (TC5)
# =============================================================================
print("\n[Fig 2] Energy vs PDR — TC5")

tc5m = load_tc_metrics(5)

energy_vals = [tc5m[(a, "total_energy_mJ")]["mean"] / 1000 for a in ADR_KEYS]
energy_errs = [tc5m[(a, "total_energy_mJ")]["std"]  / 1000 for a in ADR_KEYS]
energy_lo   = [tc5m[(a, "total_energy_mJ")]["ci_low"] / 1000 for a in ADR_KEYS]
energy_hi   = [tc5m[(a, "total_energy_mJ")]["ci_hi"]  / 1000 for a in ADR_KEYS]
pdr_vals    = [tc5m[(a, "pdr_pct")]["mean"]              for a in ADR_KEYS]

x2 = np.arange(len(ADR_KEYS))

fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
ax2 = ax1.twinx()

bars2 = ax1.bar(x2, energy_vals,
                color=[BAR_BLUES[a] for a in ADR_KEYS],
                edgecolor="white", linewidth=0.4, zorder=3)
ax1.errorbar(x2, energy_vals, yerr=energy_errs, fmt="none",
             ecolor="#555555", capsize=3, linewidth=0.9, zorder=4)
for xi, (lo, hi) in enumerate(zip(energy_lo, energy_hi)):
    ax1.plot([xi - 0.18, xi + 0.18], [lo, lo], color="#555555",
             linewidth=1.0, zorder=5)
    ax1.plot([xi - 0.18, xi + 0.18], [hi, hi], color="#555555",
             linewidth=1.0, zorder=5)
label_bars_smart(ax1, bars2, energy_vals, base_pad=0.02, min_gap=0.8, fontsize=8.5)

ax2.plot(x2, pdr_vals, color="#333333", linewidth=2.0, linestyle="--",
         marker="D", markersize=7, zorder=6)
for xi, pdr in enumerate(pdr_vals):
    ax2.annotate(f"{pdr:.1f}%",
                 xy=(xi, pdr), xytext=(0, 9), textcoords="offset points",
                 ha="center", fontsize=9, color="#333333")

hgrid_only(ax1)
ax1.set_xticks(x2)
ax1.set_xticklabels([ADR_LABELS[a] for a in ADR_KEYS], rotation=18, ha="right")
ax1.set_ylabel("Total Energy (kJ)", labelpad=6)
ax1.set_title("Energy Consumption vs. PDR  (30 dev, 100 m, 8 WiFi APs)")
ax2.set_ylabel("Packet Delivery Ratio (%)", labelpad=6)
ax2.set_ylim(0, 130)
ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
ax1.tick_params(axis="both", direction="out", length=3, width=0.7)
ax2.tick_params(axis="y",   direction="out", length=3, width=0.7)

energy_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=BAR_BLUES[k],
                  edgecolor="white", label=ADR_LABELS[k])
    for k in ADR_KEYS
]
pdr_handle = mlines.Line2D([], [], color="#333333", linewidth=2.0,
                            linestyle="--", marker="D", markersize=7,
                            label="PDR (%)")
ax1.legend(handles=energy_handles + [pdr_handle],
           loc="upper left", fontsize=9, framealpha=0.95,
           edgecolor="#cccccc", handlelength=1.4, ncol=2)
fig.tight_layout()
save_fig(fig, "fig2_energy_pdr_tc5")


# =============================================================================
# FIGURE 3 — PDR vs. Gateway Radius, 1x2 panels  (N=10 and N=50)
# =============================================================================
print("\n[Fig 3] PDR vs. Gateway Radius")

N_VALS   = [10, 50]
N_TITLES = {10: "$N = 10$  devices", 50: "$N = 50$  devices"}

plt.rcParams["axes.facecolor"] = "#f5f8fc"
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.alpha"]     = 0.40

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.suptitle("PDR vs. Gateway Radius", fontsize=14, fontweight="bold", y=1.02)

for idx, n in enumerate(N_VALS):
    ax = axes[idx]
    df = load_sweep_pdr(n)
    if df.empty:
        continue

    for adr in ADR_KEYS:
        sub = df[df["adr_key"] == adr].sort_values("radius")
        xr  = sub["radius"].values.astype(float)
        ym  = sub["mean"].values
        ylo = sub["ci95_low"].values
        yhi = sub["ci95_high"].values

        xs, ys = smooth_dense(xr, ym)
        _, yls = smooth_dense(xr, ylo)
        _, yhs = smooth_dense(xr, yhi)

        ax.plot(xs, ys,
                color=LINE_BLUES[adr], linewidth=LINE_LW[adr],
                linestyle=LINE_LS[adr], zorder=4)
        ax.fill_between(xs, yls, yhs,
                        color=LINE_BLUES[adr], alpha=0.10, zorder=2, lw=0)
        step = max(1, len(xr) // 8)
        ax.scatter(xr[::step], ym[::step],
                   color=LINE_BLUES[adr], marker=LINE_MARKERS[adr],
                   s=40, zorder=5, edgecolors="white", linewidths=0.7)

    sub0   = df[df["adr_key"] == ADR_KEYS[0]].sort_values("radius")
    radii  = sub0["radius"].values.astype(int)
    tick_step = 200 if radii[-1] >= 800 else 100
    xticks = [r for r in radii if r % tick_step == 0 or r == radii[0]]

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(r) for r in xticks], fontsize=9)
    ax.set_title(N_TITLES[n], fontsize=12)
    ax.set_xlabel("Gateway Radius (m)", labelpad=4)
    ax.set_ylabel("PDR (%)", labelpad=4)
    ax.set_ylim(-2, 108)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_xlim(left=0, right=radii[-1] * 1.05)
    ax.tick_params(axis="both", direction="in", length=3, width=0.7)

line_legend_blue(fig, y=-0.06)
fig.tight_layout(rect=[0, 0.12, 1, 1])
save_fig(fig, "fig3_pdr_vs_radius_sweep")

# restore bar-chart rcParams
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"]     = 0.35


# =============================================================================
# FIGURE 4 — Density scalability  (20 / 30 / 50 / 100 devices, 100 m)
#            TC16 (5 devices) removed
# =============================================================================
print("\n[Fig 4] Density scalability")

density_tcs = [(17, 20), (5, 30), (18, 50), (19, 100)]
n_sc  = len(density_tcs)
n_adr = len(ADR_KEYS)
bw    = 0.16
x4    = np.arange(n_sc)
offs  = np.linspace(-(n_adr - 1) / 2, (n_adr - 1) / 2, n_adr) * bw

fig, ax = plt.subplots(figsize=(9, 5.2))
for i, adr in enumerate(ADR_KEYS):
    vals = [grand[tc].get(adr, np.nan) for tc, _ in density_tcs]
    bars = ax.bar(x4 + offs[i], vals, bw,
                  color=BAR_BLUES[adr], edgecolor="white",
                  linewidth=0.4, zorder=3)
    label_bars_smart(ax, bars, vals, base_pad=1.0, min_gap=7.5, fontsize=8.5)

hgrid_only(ax)
ax.set_xticks(x4)
ax.set_xticklabels([f"{d} devices" for _, d in density_tcs], fontsize=11)
ax.set_xlabel("Number of Devices  (radius = 100 m, 8 WiFi APs)", labelpad=6)
ax.set_ylabel("Packet Delivery Rate (%)", labelpad=6)
ax.set_title("PDR Scalability with Increasing Device Density")
ax.set_ylim(0, 122)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis="both", direction="out", length=3, width=0.7)
bar_legend(ax, loc="lower left")
fig.tight_layout()
save_fig(fig, "fig4_density_scalability")

print(f"\nAll figures saved to {OUT_DIR}/")
