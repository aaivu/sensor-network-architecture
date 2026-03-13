#!/usr/bin/env python3
"""
8 Individual PDR Plots
======================
Generates exactly 8 standalone graphs, each saved as .png + .eps:

  PDR vs Number of Devices  (r =  50 m)  →  pdr_vs_devices_r050
  PDR vs Number of Devices  (r = 100 m)  →  pdr_vs_devices_r100
  PDR vs Number of Devices  (r = 200 m)  →  pdr_vs_devices_r200
  PDR vs Number of Devices  (r = 500 m)  →  pdr_vs_devices_r500

  PDR vs Radius  (n =  5)  →  pdr_vs_radius_n005
  PDR vs Radius  (n = 10)  →  pdr_vs_radius_n010
  PDR vs Radius  (n = 20)  →  pdr_vs_radius_n020
  PDR vs Radius  (n = 50)  →  pdr_vs_radius_n050

Usage:
    python3 plot_pdr_8.py
    python3 plot_pdr_8.py --input sweep_results --out sweep_results/plots_8
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

# ─── Style constants ──────────────────────────────────────────────────────────
ADR_KEYS = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS = {
    "no_adr":        "No ADR",
    "classical_adr": "Classical ADR",
    "ddqn_adr":      "DDQN-PER",
    "ppo_adr":       "PPO",
    "marl_adr":      "MARL",
}
# Wong (2011) colorblind-safe palette
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
    "marl_adr":      (0, (3, 1, 1, 1)),   # dash-dot-dot
}

# ─── Matplotlib global defaults ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            12,
    "axes.titlesize":       14,
    "axes.titleweight":     "bold",
    "axes.labelsize":       12,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "legend.fontsize":      11,
    "legend.framealpha":    0.95,
    "legend.edgecolor":     "#cccccc",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.linewidth":       0.9,
    "axes.grid":            True,
    "grid.alpha":           0.28,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.7,
    "axes.axisbelow":       True,
    "axes.facecolor":       "#f9f9f9",
    "figure.facecolor":     "white",
    "figure.dpi":           150,
    "lines.linewidth":      2.4,
    "lines.solid_capstyle": "round",
})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(results_dir.glob("n???_r????_summary.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
    if not frames:
        raise FileNotFoundError(
            f"No summary CSVs found in {results_dir}. Run run_sweeps.py first.")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} rows from "
          f"{df.groupby(['nDevices', 'radius']).ngroups} unique (n, r) configs\n")
    return df


def smooth_spline(x: np.ndarray, y: np.ndarray, n_pts: int = 500):
    """Smoothing cubic spline — more points than the original data for a
    visually continuous curve.  Falls back to linear interpolation for <4 pts."""
    if len(x) < 4:
        xs = np.linspace(x[0], x[-1], n_pts)
        return xs, np.interp(xs, x, y)
    rng = max(float(np.ptp(y)), 2.0)
    s   = len(x) * (rng ** 2) * 0.35
    try:
        spl = UnivariateSpline(x, y, k=3, s=s, ext=3)
    except Exception:
        spl = UnivariateSpline(x, y, k=1, s=s, ext=3)
    xs = np.linspace(x[0], x[-1], n_pts)
    # Clamp to the observed range ± a small margin
    ys = np.clip(spl(xs), float(np.min(y)) - 1.5, float(np.max(y)) + 1.5)
    return xs, ys


def get_series(df, adr_key, x_col, fixed_col, fixed_val):
    """Return sorted (x, mean, ci_lo, ci_hi) arrays for one line."""
    sub = df[
        (df["metric"]   == "pdr_pct") &
        (df["adr_key"]  == adr_key)   &
        (df[fixed_col]  == fixed_val)
    ].sort_values(x_col)
    if sub.empty:
        return None
    return (sub[x_col].values.astype(float),
            sub["mean"].values,
            sub["ci95_low"].values,
            sub["ci95_high"].values)


def legend_handles():
    """Build a list of Line2D handles for the shared legend."""
    handles = []
    for key in ADR_KEYS:
        h = mlines.Line2D(
            [], [],
            color=ADR_COLORS[key],
            linestyle=ADR_LS[key],
            linewidth=2.4,
            marker=ADR_MARKERS[key],
            markersize=7,
            label=ADR_LABELS[key],
        )
        handles.append(h)
    return handles


def save_fig(fig, base: Path):
    for ext in (".png", ".eps"):
        out = base.with_suffix(ext)
        fig.savefig(
            out,
            bbox_inches="tight",
            dpi=150 if ext == ".png" else None,
            format=ext.lstrip("."),
        )
        print(f"  Saved  →  {out}")


# ─── Core drawing function ────────────────────────────────────────────────────

def draw_pdr_plot(df, x_col, fixed_col, fixed_val,
                  x_label, x_ticks,
                  title, footnote,
                  out_path: Path,
                  marker_every: int = 4):
    """
    Draw a single PDR plot:
      - smooth continuous lines  (spline)
      - subtle CI shading
      - sparse scatter markers on original data points
      - full legend below
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    any_data = False
    for key in ADR_KEYS:
        result = get_series(df, key, x_col, fixed_col, fixed_val)
        if result is None:
            continue
        any_data = True
        x, y, lo, hi = result

        xs, ys   = smooth_spline(x, y)
        _,  los  = smooth_spline(x, lo)
        _,  his  = smooth_spline(x, hi)

        col = ADR_COLORS[key]
        lbl = ADR_LABELS[key]

        # Smooth curve
        ax.plot(xs, ys,
                color=col, linestyle=ADR_LS[key],
                linewidth=2.4, label=lbl, zorder=4)

        # CI band
        ax.fill_between(xs, los, his,
                        color=col, alpha=0.09, zorder=2, lw=0)

        # Sparse markers on raw data points
        every = max(1, marker_every)
        ax.scatter(x[::every], y[::every],
                   color=col, marker=ADR_MARKERS[key],
                   s=38, zorder=5, edgecolors=col, linewidths=0.6)

    if not any_data:
        print(f"  No data for {fixed_col}={fixed_val} — skipping {out_path.name}")
        plt.close(fig)
        return

    # Axes formatting
    ax.set_xlabel(x_label, labelpad=5)
    ax.set_ylabel("Packet Delivery Ratio (%)", labelpad=5)
    ax.set_title(title, pad=10)
    ax.set_ylim(-3, 108)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.tick_params(axis="both", direction="in", length=4, width=0.8,
                   which="both")

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(v)) for v in x_ticks])

    # Footnote
    fig.text(0.5, -0.04, footnote,
             ha="center", fontsize=8.5, color="#888888", style="italic")

    # Legend below plot
    fig.legend(handles=legend_handles(),
               loc="lower center",
               ncol=3,
               bbox_to_anchor=(0.5, -0.20),
               fontsize=10.5,
               frameon=True,
               handlelength=2.5,
               handletextpad=0.6,
               columnspacing=1.2)

    fig.tight_layout(rect=[0, 0.01, 1, 1])
    save_fig(fig, out_path)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

FOOTNOTE = ("WiFi=5 | LTE=10 | appPeriod=60 s | simTime=5000 s | "
            "shaded band = 95 % CI across 100 runs")


def main():
    parser = argparse.ArgumentParser(description="Generate 8 individual PDR plots")
    parser.add_argument("--input", default="sweep_results",
                        help="Directory containing summary CSVs")
    parser.add_argument("--out",   default="sweep_results/plots_8",
                        help="Output directory for PNGs and EPSs")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from  {in_dir}  …")
    df = load_summaries(in_dir)

    avail_radii   = sorted(df["radius"].unique())
    avail_devices = sorted(df["nDevices"].unique())

    # ── Series A: PDR vs Number of Devices at fixed radius ───────────────────
    series_a = [
        (50,  "r =  50 m"),
        (100, "r = 100 m"),
        (200, "r = 200 m"),
        (500, "r = 500 m"),
    ]

    print("─── Series A: PDR vs Number of Devices ───")
    for r, r_label in series_a:
        if r not in avail_radii:
            print(f"  Radius {r} m not in data — skipping")
            continue
        devices_at_r = sorted(
            df.loc[df["radius"] == r, "nDevices"].unique()
        )
        draw_pdr_plot(
            df,
            x_col       = "nDevices",
            fixed_col   = "radius",
            fixed_val   = r,
            x_label     = "Number of Devices",
            x_ticks     = devices_at_r,
            title       = f"PDR vs Number of Devices  ({r_label})",
            footnote    = FOOTNOTE,
            out_path    = out_dir / f"pdr_vs_devices_r{r:04d}",
            marker_every= 1,          # few x-values → show every point
        )

    print()

    # ── Series B: PDR vs Radius at fixed device count ────────────────────────
    series_b = [
        (5,  "n =  5 devices"),
        (10, "n = 10 devices"),
        (20, "n = 20 devices"),
        (50, "n = 50 devices"),
    ]

    print("─── Series B: PDR vs Radius ───")
    for n, n_label in series_b:
        if n not in avail_devices:
            print(f"  n={n} not in data — skipping")
            continue
        radii_at_n = sorted(
            df.loc[df["nDevices"] == n, "radius"].unique()
        )
        # Tick every 200 m to avoid crowding
        x_ticks = [r for r in radii_at_n if r % 200 == 0]
        draw_pdr_plot(
            df,
            x_col       = "radius",
            fixed_col   = "nDevices",
            fixed_val   = n,
            x_label     = "Radius (m)",
            x_ticks     = x_ticks,
            title       = f"PDR vs Radius  ({n_label})",
            footnote    = FOOTNOTE,
            out_path    = out_dir / f"pdr_vs_radius_n{n:03d}",
            marker_every= 5,          # dense sweep → every 5th point
        )

    print(f"\nAll 8 plots saved to  {out_dir}/")


if __name__ == "__main__":
    main()
