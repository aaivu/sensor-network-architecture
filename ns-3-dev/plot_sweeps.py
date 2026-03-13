#!/usr/bin/env python3
"""
Sweep Plot Generator  —  clean publication-quality output
=========================================================
Produces 2×2 grid figures saved as .png + .eps:
  • sweep_a_{metric}  — metric vs Number of Devices (4 radii)
  • sweep_b_{metric}  — metric vs Radius            (4 device counts)

Usage:
    python3 plot_sweeps.py
    python3 plot_sweeps.py --metric avg_sf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# ── style ─────────────────────────────────────────────────────────────────────
# Colorblind-safe, high-contrast palette (Wong 2011)
ADR_KEYS   = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS = {
    "no_adr":        "No ADR",
    "classical_adr": "Classical ADR",
    "ddqn_adr":      "DDQN-PER",
    "ppo_adr":       "PPO",
    "marl_adr":      "MARL",
}
ADR_COLORS = {
    "no_adr":        "#D55E00",   # vermillion
    "classical_adr": "#E69F00",   # amber
    "ddqn_adr":      "#0072B2",   # blue
    "ppo_adr":       "#009E73",   # green
    "marl_adr":      "#CC79A7",   # pink-purple
}
ADR_MARKERS = {
    "no_adr":        "o",
    "classical_adr": "s",
    "ddqn_adr":      "^",
    "ppo_adr":       "D",
    "marl_adr":      "P",
}
ADR_LS = {
    "no_adr":        "-",
    "classical_adr": "--",
    "ddqn_adr":      "-",
    "ppo_adr":       "-.",
    "marl_adr":      (0, (3, 1, 1, 1)),  # dash-dot-dot
}

METRIC_LABELS = {
    "pdr_pct":               "Packet Delivery Ratio (%)",
    "avg_snr_db":            "Average SNR (dB)",
    "avg_sf":                "Average Spreading Factor",
    "total_energy_mJ":       "Total Energy (mJ)",
    "energy_per_success_mJ": "Energy per Delivered Packet (mJ)",
    "avg_toa_ms":            "Avg Time-on-Air (ms)",
    "snr_margin_median_db":  "Median SNR Margin (dB)",
}
HIGHER_IS_BETTER = {
    "pdr_pct": True,  "avg_snr_db": True,  "snr_margin_median_db": True,
    "avg_sf":  False, "total_energy_mJ": False,
    "energy_per_success_mJ": False, "avg_toa_ms": False,
}

RESULTS_DIR = Path("sweep_results")
PLOTS_DIR   = Path("sweep_results/plots")

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    10,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     ":",
    "grid.linewidth":     0.7,
    "axes.axisbelow":     True,
    "axes.facecolor":     "#fafafa",
    "figure.facecolor":   "white",
    "figure.dpi":         150,
    "lines.linewidth":    2.2,
    "lines.solid_capstyle": "round",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(results_dir.glob("n???_r????_summary.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    if not frames:
        raise FileNotFoundError(
            f"No summary CSVs in {results_dir}. Run run_sweeps.py first.")
    return pd.concat(frames, ignore_index=True)


def smooth(x: np.ndarray, y: np.ndarray, n_pts: int = 400):
    """Smoothing spline with s scaled to the data range so all lines look clean.
    Noisy lines (large ptp) get more smoothing; flat lines stay exact."""
    if len(x) < 4:
        xs = np.linspace(x[0], x[-1], n_pts)
        return xs, np.interp(xs, x, y)
    # s proportional to range^2 per point — aggressive enough for Classic ADR
    rng = max(float(np.ptp(y)), 2.0)
    s   = len(x) * (rng ** 2) * 0.40
    try:
        spl = UnivariateSpline(x, y, k=3, s=s, ext=3)
    except Exception:
        spl = UnivariateSpline(x, y, k=1, s=s, ext=3)
    xs = np.linspace(x[0], x[-1], n_pts)
    ys = np.clip(spl(xs), float(np.min(y)) - 2, float(np.max(y)) + 2)
    return xs, ys


def pivot(df, metric, adr_key, x_col, fixed_col, fixed_val):
    sub = df[
        (df["metric"]  == metric) &
        (df["adr_key"] == adr_key) &
        (df[fixed_col] == fixed_val)
    ].sort_values(x_col)
    return sub[[x_col, "mean", "ci95_low", "ci95_high"]].reset_index(drop=True)


def save_fig(fig, base: Path):
    for ext in (".png", ".eps"):
        p = base.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight",
                    dpi=150 if ext == ".png" else None,
                    format=ext.lstrip("."))
        print(f"  Saved → {p}")


# ── panel ─────────────────────────────────────────────────────────────────────

def draw_panel(ax, df_all, metric, x_col, fixed_col, fixed_val, x_label, title,
               marker_every: int = 4):
    """Draw one 2×2 cell with smooth lines, sparse markers, subtle CI band."""
    hib = HIGHER_IS_BETTER.get(metric, True)

    for key in ADR_KEYS:
        sub = pivot(df_all, metric, key, x_col, fixed_col, fixed_val)
        if sub.empty:
            continue
        x  = sub[x_col].values.astype(float)
        y  = sub["mean"].values
        lo = sub["ci95_low"].values
        hi = sub["ci95_high"].values

        xs, ys   = smooth(x, y)
        _,  los  = smooth(x, lo)
        _,  his  = smooth(x, hi)

        col = ADR_COLORS[key]
        lbl = ADR_LABELS[key]

        # Smooth line
        ax.plot(xs, ys, color=col, linewidth=2.2,
                linestyle=ADR_LS[key], label=lbl, zorder=4)

        # CI band — very subtle
        ax.fill_between(xs, los, his, color=col, alpha=0.08, zorder=2, lw=0)

        # Sparse data markers (every marker_every-th original point)
        xm = x[::marker_every]
        ym = y[::marker_every]
        ax.scatter(xm, ym, color=col, marker=ADR_MARKERS[key],
                   s=28, zorder=5, edgecolors=col, linewidths=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label, labelpad=4)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), labelpad=4)

    # y-axis: 0–105 for PDR, otherwise auto
    if metric == "pdr_pct":
        ax.set_ylim(-2, 105)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    # Subtle tick marks inward
    ax.tick_params(axis="both", direction="in", length=3, width=0.7)


def shared_legend(fig, ncol: int = 5):
    """Single legend below the figure with line+marker handles."""
    handles = []
    for key in ADR_KEYS:
        h = mlines.Line2D(
            [], [], color=ADR_COLORS[key],
            linestyle=ADR_LS[key], linewidth=2.2,
            marker=ADR_MARKERS[key], markersize=6,
            label=ADR_LABELS[key],
        )
        handles.append(h)
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol, bbox_to_anchor=(0.5, 0.0),
               fontsize=10, frameon=True,
               handlelength=2.4, handletextpad=0.6, columnspacing=1.2)


# ── Sweep A ───────────────────────────────────────────────────────────────────

def plot_sweep_a(df_all, metric, out_dir):
    # Use ALL radii that have ≥ 4 device counts (the Sweep-A radii)
    radii = sorted([
        r for r in df_all["radius"].unique()
        if df_all[df_all["radius"] == r]["nDevices"].nunique() >= 4
    ])
    if not radii:
        print("Sweep A: no data — skipping"); return

    ncols = 2
    nrows = int(np.ceil(len(radii) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows),
                              sharey=True, sharex=False)
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric)} vs Number of Devices",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for idx, r in enumerate(radii):
        ax = axes_flat[idx]
        x_vals = sorted(df_all.loc[df_all["radius"] == r, "nDevices"].unique())
        draw_panel(ax, df_all, metric,
                   x_col="nDevices", fixed_col="radius", fixed_val=r,
                   x_label="Number of Devices",
                   title=f"Radius = {r} m",
                   marker_every=2)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], rotation=0)
        # Remove y label from right column
        if idx % ncols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[len(radii):]:
        ax.set_visible(False)

    fig.text(0.5, 0.01,
             "WiFi interferers = 5 | LTE interferers = 10 | "
             "appPeriod = 60 s | simTime = 5000 s | environmental = true",
             ha="center", fontsize=8, color="#888888", style="italic")

    shared_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    save_fig(fig, out_dir / f"sweep_a_{metric}")
    plt.close(fig)


# ── Sweep B ───────────────────────────────────────────────────────────────────

def plot_sweep_b(df_all, metric, out_dir):
    # Use ALL device counts present in data, sorted
    devices = sorted(df_all["nDevices"].unique())
    if not devices:
        print("Sweep B: no data — skipping"); return

    ncols = 4
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows),
                              sharey=True, sharex=False)
    axes_flat = np.array(axes).flatten()

    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric)} vs Gateway Radius",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for idx, n in enumerate(devices):
        ax    = axes_flat[idx]
        radii = sorted(df_all.loc[df_all["nDevices"] == n, "radius"].unique())
        # marker_every: denser for sparse data (4 pts), sparser for full sweep (39 pts)
        m_every = 1 if len(radii) <= 4 else 5
        draw_panel(ax, df_all, metric,
                   x_col="radius", fixed_col="nDevices", fixed_val=n,
                   x_label="Radius (m)",
                   title=f"n = {n} Devices",
                   marker_every=m_every)
        # Tick strategy: sparse data → show all 4 points; dense → every 200 m
        if len(radii) <= 4:
            ax.set_xticks(radii)
            ax.set_xticklabels([str(r) for r in radii], rotation=30, ha="right")
        else:
            tick_vals = [r for r in radii if r % 200 == 0]
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([str(r) for r in tick_vals], rotation=0)
        # Remove y label except leftmost column
        if idx % ncols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[len(devices):]:
        ax.set_visible(False)

    fig.text(0.5, 0.01,
             "WiFi interferers = 5 | LTE interferers = 10 | "
             "appPeriod = 60 s | simTime = 5000 s | environmental = true",
             ha="center", fontsize=8, color="#888888", style="italic")

    shared_legend(fig, ncol=5)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    save_fig(fig, out_dir / f"sweep_b_{metric}")
    plt.close(fig)


# ── combined overview ─────────────────────────────────────────────────────────

def plot_combined_overview(df_all, out_dir):
    metric    = "pdr_pct"
    a_radii   = [r for r in [50, 100, 200, 500] if r in df_all["radius"].unique()]
    b_devices = [d for d in [5, 10, 20, 50]     if d in df_all["nDevices"].unique()]
    total     = len(a_radii) + len(b_devices)
    if total == 0:
        return

    fig, axes = plt.subplots(1, total, figsize=(total * 3.6, 5.0),
                              sharey=True)
    axes_flat = list(axes) if total > 1 else [axes]

    fig.suptitle("PDR Parametric Sweeps — All ADR Methods",
                 fontsize=13, fontweight="bold", y=1.03)

    col = 0
    for ridx, r in enumerate(a_radii):
        ax = axes_flat[col]; col += 1
        x_vals = sorted(df_all.loc[df_all["radius"] == r, "nDevices"].unique())
        draw_panel(ax, df_all, metric,
                   x_col="nDevices", fixed_col="radius", fixed_val=r,
                   x_label="Devices", title=f"r = {r} m", marker_every=3)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], rotation=60,
                           ha="right", fontsize=7)
        if ridx != 0:
            ax.set_ylabel("")
        if ridx == 0:
            ax.annotate("Sweep A", xy=(0.04, 1.09), xycoords="axes fraction",
                        fontsize=9, fontweight="bold", color="#555555")

    for didx, n in enumerate(b_devices):
        ax = axes_flat[col]; col += 1
        radii = sorted(df_all.loc[df_all["nDevices"] == n, "radius"].unique())
        draw_panel(ax, df_all, metric,
                   x_col="radius", fixed_col="nDevices", fixed_val=n,
                   x_label="Radius (m)", title=f"n = {n}", marker_every=6)
        tick_vals = [r for r in radii if r % 200 == 0]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([str(r) for r in tick_vals], rotation=60,
                           ha="right", fontsize=7)
        ax.set_ylabel("")
        if didx == 0:
            ax.annotate("Sweep B", xy=(0.04, 1.09), xycoords="axes fraction",
                        fontsize=9, fontweight="bold", color="#555555")

    # Thin divider between groups
    mid = len(a_radii) / total
    fig.add_artist(
        plt.Line2D([mid, mid], [0.08, 0.92],
                   transform=fig.transFigure,
                   color="#cccccc", linewidth=1.2, linestyle="--"))

    shared_legend(fig)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, out_dir / "combined_pdr_overview")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(RESULTS_DIR))
    parser.add_argument("--out",    default=str(PLOTS_DIR))
    parser.add_argument("--metric", nargs="*", default=["pdr_pct"])
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summaries from {in_dir} …")
    df_all = load_all_summaries(in_dir)
    print(f"  Loaded {len(df_all)} rows from "
          f"{df_all.groupby(['nDevices','radius']).ngroups} unique configs\n")

    for metric in args.metric:
        if metric not in df_all["metric"].values:
            print(f"  Warning: metric '{metric}' not found — skipping")
            continue
        print(f"Plotting: {metric}")
        plot_sweep_a(df_all, metric, out_dir)
        plot_sweep_b(df_all, metric, out_dir)

    if "pdr_pct" in args.metric:
        plot_combined_overview(df_all, out_dir)

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
