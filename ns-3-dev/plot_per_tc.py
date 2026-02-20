#!/usr/bin/env python3
"""
Per-Test-Case Graph Generator
===============================
Reads tc{id}_summary.csv for each of the 34 test cases and produces
a comprehensive multi-panel comparison figure per TC.

Each figure contains 8 sub-plots:
  1. PDR (%)                   2. Avg Spreading Factor
  3. Avg TX Power (dBm)        4. Total Energy (mJ)
  5. Energy / Success (mJ)     6. Avg SNR (dB)
  7. Avg Time-on-Air (ms)      8. Median SNR Margin (dB)

All bars show mean ± std with individual 95 % CI markers.

Usage:
    cd ns-3-dev
    source venv/bin/activate
    python3 plot_per_tc.py                        # all 34 TCs
    python3 plot_per_tc.py --tc 5 12 31           # specific TCs
    python3 plot_per_tc.py --out my_plots/per_tc  # custom output dir
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── style constants ─────────────────────────────────────────────────────────
ADR_KEYS = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS = {
    "no_adr":       "No ADR",
    "classical_adr":"Classical",
    "ddqn_adr":     "DDQN-PER",
    "ppo_adr":      "PPO",
    "marl_adr":     "MARL",
}
ADR_COLORS = {
    "no_adr":       "#e74c3c",
    "classical_adr":"#e67e22",
    "ddqn_adr":     "#2980b9",
    "ppo_adr":      "#27ae60",
    "marl_adr":     "#8e44ad",
}

# The 8 panels: (metric_col, y-axis label, higher-is-better)
PANELS = [
    ("pdr_pct",               "PDR (%)",                     True ),
    ("avg_sf",                "Avg Spreading Factor",         False),
    ("avg_tx_power_dbm",      "Avg TX Power (dBm)",           False),
    ("total_energy_mJ",       "Total Energy (mJ)",            False),
    ("energy_per_success_mJ", "Energy / Success (mJ)",        False),
    ("avg_snr_db",            "Avg SNR (dB)",                 True ),
    ("avg_toa_ms",            "Avg Time-on-Air (ms)",         False),
    ("snr_margin_median_db",  "Median SNR Margin (dB)",       True ),
]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def _wide(df: pd.DataFrame, metric: str, col: str) -> pd.Series:
    """Return a Series indexed by adr_key for a given metric and column."""
    sub = df[df["metric"] == metric].set_index("adr_key")
    if col not in sub.columns:
        return pd.Series(dtype=float)
    return sub[col]


def _draw_panel(ax, df, metric, ylabel, higher_is_better,
                present_keys, present_labels, present_colors):
    """Draw one bar-chart panel on ax."""
    means   = _wide(df, metric, "mean")
    stds    = _wide(df, metric, "std")
    ci_low  = _wide(df, metric, "ci95_low")
    ci_high = _wide(df, metric, "ci95_high")

    vals, errs, ci_lo, ci_hi, colors = [], [], [], [], []
    for k in present_keys:
        vals.append(  float(means[k])   if k in means.index  else np.nan)
        errs.append(  float(stds[k])    if k in stds.index   else 0.0)
        ci_lo.append( float(ci_low[k])  if k in ci_low.index else np.nan)
        ci_hi.append( float(ci_high[k]) if k in ci_high.index else np.nan)
        colors.append(present_colors[k])

    x = np.arange(len(present_keys))

    # bars with std error
    bars = ax.bar(x, vals, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none",
                ecolor="black", capsize=3, linewidth=0.9, zorder=4)

    # 95 % CI markers (thin horizontal lines)
    for xi, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.plot([xi - 0.18, xi + 0.18], [lo, lo],
                    color="black", linewidth=1.0, zorder=5)
            ax.plot([xi - 0.18, xi + 0.18], [hi, hi],
                    color="black", linewidth=1.0, zorder=5)

    # value annotations inside/above bars
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        h = bar.get_height()
        if h == 0:
            continue
        # choose label position
        y_annot = h * 0.5
        va = "center"
        fmt = f"{v:.1f}" if abs(v) < 1000 else f"{v/1000:.1f}k"
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_annot, fmt,
                ha="center", va=va, fontsize=7,
                color="white" if h > ax.get_ylim()[1] * 0.1 else "black",
                fontweight="bold", zorder=6)

    # highlight best-performing bar
    finite = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if finite:
        best_i = max(finite, key=lambda t: t[1] if higher_is_better else -t[1])[0]
        bars[best_i].set_edgecolor("black")
        bars[best_i].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(present_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)

    # auto y-limit with headroom
    finite_vals = [v for v in vals if not np.isnan(v)]
    if finite_vals:
        ymax = max(finite_vals) + max(errs) if any(errs) else max(finite_vals)
        headroom = max(abs(ymax) * 0.20, 1e-6)
        ax.set_ylim(bottom=min(0, min(finite_vals) - headroom * 0.5),
                    top=ymax + headroom)


def plot_tc(tc_id: int, summary_csv: Path, out_dir: Path):
    df = pd.read_csv(summary_csv)

    # metadata from first row
    meta = df.iloc[0]
    label        = meta["label"]
    scenario     = meta["scenario"]
    n_devices    = int(meta["nDevices"])
    radius       = int(meta["radius"])
    wifi         = int(meta["wifiInterferers"])
    lte          = int(meta["lteInterferers"])
    period       = int(meta["appPeriod"])
    sim_time     = int(meta["simulationTime"])
    try:
        n_runs = int(df[df["metric"] == "pdr_pct"]["n_runs"].max())
    except Exception:
        n_runs = "?"

    # only keep ADR keys present in this file
    present_keys   = [k for k in ADR_KEYS   if k in df["adr_key"].values]
    present_labels = [ADR_LABELS[k] for k in present_keys]
    present_colors = {k: ADR_COLORS[k] for k in present_keys}

    # ── figure layout ────────────────────────────────────────────────────────
    nrows, ncols = 2, 4
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.4 + 1.6))

    # title area
    fig.text(
        0.5, 0.97,
        f"TC-{tc_id:02d}  ▸  {label}",
        ha="center", va="top", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.93,
        f"Scenario: {scenario}    |    "
        f"{n_devices} devices, {radius} m radius, "
        f"WiFi={wifi}, LTE={lte}, period={period}s, sim={sim_time}s    |    "
        f"{n_runs} runs",
        ha="center", va="top", fontsize=8.5, color="#444444",
    )

    # sub-plots
    axes = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        axes.append(ax)

    for ax, (metric, ylabel, hib) in zip(axes, PANELS):
        _draw_panel(ax, df, metric, ylabel, hib,
                    present_keys, present_labels, present_colors)
        ax.set_title(ylabel, pad=4)
        ax.set_ylabel("")          # ylabel already in title for compactness

    # shared legend
    patches = [mpatches.Patch(color=ADR_COLORS[k], label=ADR_LABELS[k])
               for k in present_keys]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(present_keys), bbox_to_anchor=(0.5, 0.0),
               frameon=True, fontsize=8.5)

    # annotation: black border = best bar
    fig.text(0.5, 0.025,
             "★ Black border = best-performing method for that metric",
             ha="center", fontsize=7.5, color="#666666", style="italic")

    fig.subplots_adjust(top=0.88, bottom=0.11, hspace=0.48, wspace=0.38)

    out_path = out_dir / f"tc{tc_id:02d}_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved TC-{tc_id:02d} → {out_path}")


# ── second figure: detailed statistics table ─────────────────────────────────

def plot_tc_stats_table(tc_id: int, summary_csv: Path, out_dir: Path):
    """Render a tidy statistics table (mean / std / min / median / max / ci95)."""
    df = pd.read_csv(summary_csv)
    meta = df.iloc[0]

    metrics_order = [m for m, _, _ in PANELS if m in df["metric"].values]
    present_keys  = [k for k in ADR_KEYS if k in df["adr_key"].values]

    rows = []
    for metric in metrics_order:
        sub = df[df["metric"] == metric].set_index("adr_key")
        for k in present_keys:
            if k not in sub.index:
                continue
            r = sub.loc[k]
            rows.append({
                "Metric":   metric,
                "ADR":      ADR_LABELS[k],
                "n_runs":   int(r["n_runs"]),
                "Mean":     f"{r['mean']:.3f}",
                "Std":      f"{r['std']:.3f}",
                "Min":      f"{r['min']:.3f}",
                "Median":   f"{r['median']:.3f}",
                "Max":      f"{r['max']:.3f}",
                "CI95 low": f"{r['ci95_low']:.3f}",
                "CI95 high":f"{r['ci95_high']:.3f}",
            })

    tdf = pd.DataFrame(rows)
    n_rows = len(tdf)

    fig_h = max(4, n_rows * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    col_labels = list(tdf.columns)
    cell_data  = tdf.values.tolist()

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # alternate row colours + highlight by ADR
    adr_bg = {ADR_LABELS[k]: ADR_COLORS[k] + "22" for k in ADR_KEYS}
    for i, row in enumerate(cell_data):
        adr_name = row[1]
        bg = adr_bg.get(adr_name, "#ffffff")
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title(
        f"TC-{tc_id:02d} — {meta['label']} — Statistics Table",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    out_path = out_dir / f"tc{tc_id:02d}_stats_table.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved TC-{tc_id:02d} table → {out_path}")


# ── third figure: CI-95 dot plot ─────────────────────────────────────────────

def plot_tc_ci_dotplot(tc_id: int, summary_csv: Path, out_dir: Path):
    """
    Dot-plot showing mean ± CI-95 for each metric × ADR type.
    Useful to visually judge statistical overlap between methods.
    """
    df = pd.read_csv(summary_csv)
    metrics = [m for m, _, _ in PANELS if m in df["metric"].values]
    present_keys = [k for k in ADR_KEYS if k in df["adr_key"].values]

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(len(metrics) * 2.2, 3.8),
        sharey=False,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric].set_index("adr_key")
        for yi, k in enumerate(present_keys):
            if k not in sub.index:
                continue
            r = sub.loc[k]
            mean  = float(r["mean"])
            lo    = float(r["ci95_low"])
            hi    = float(r["ci95_high"])
            color = ADR_COLORS[k]
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.0)
            ax.plot(mean, yi, "o", color=color,
                    markersize=6, zorder=5, markeredgecolor="black",
                    markeredgewidth=0.5)

        ax.set_yticks(range(len(present_keys)))
        ax.set_yticklabels([ADR_LABELS[k] for k in present_keys], fontsize=7)
        label = next((lbl for m, lbl, _ in PANELS if m == metric), metric)
        ax.set_title(label, fontsize=7.5, pad=3)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", alpha=0.3)

    meta = df.iloc[0]
    fig.suptitle(
        f"TC-{tc_id:02d} — 95% CI Dot-Plot — {meta['label']}",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / f"tc{tc_id:02d}_ci_dotplot.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved TC-{tc_id:02d} CI dot-plot → {out_path}")


# ── fourth figure: radar per TC ───────────────────────────────────────────────

def plot_tc_radar(tc_id: int, summary_csv: Path, out_dir: Path):
    """
    Radar chart for each TC — normalised scores for 5 key metrics.
    Score = rank among ADR methods on that metric (1=best, 5=worst → inverted to 1=best → 5=max).
    """
    df = pd.read_csv(summary_csv)
    present_keys = [k for k in ADR_KEYS if k in df["adr_key"].values]
    if len(present_keys) < 2:
        return

    radar_metrics = [
        ("pdr_pct",               True ),   # higher better
        ("avg_sf",                False),   # lower better
        ("total_energy_mJ",       False),
        ("energy_per_success_mJ", False),
        ("snr_margin_median_db",  True ),
    ]
    radar_labels = ["PDR", "SF", "Energy", "E/pkt", "SNR\nMargin"]

    # normalise each metric 0–1 (best=1)
    scores = {k: [] for k in present_keys}
    for metric, hib in radar_metrics:
        sub = df[df["metric"] == metric].set_index("adr_key")
        vals = {k: float(sub.loc[k, "mean"]) if k in sub.index else np.nan
                for k in present_keys}
        finite = [v for v in vals.values() if not np.isnan(v)]
        if not finite or max(finite) == min(finite):
            for k in present_keys:
                scores[k].append(0.5)
            continue
        vmin, vmax = min(finite), max(finite)
        for k in present_keys:
            v = vals.get(k, np.nan)
            if np.isnan(v):
                scores[k].append(0.0)
            elif hib:
                scores[k].append((v - vmin) / (vmax - vmin))
            else:
                scores[k].append(1.0 - (v - vmin) / (vmax - vmin))

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    for k in present_keys:
        vals_plot = scores[k] + scores[k][:1]
        ax.plot(angles, vals_plot, color=ADR_COLORS[k],
                linewidth=1.8, label=ADR_LABELS[k])
        ax.fill(angles, vals_plot, color=ADR_COLORS[k], alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5)

    meta = df.iloc[0]
    ax.set_title(f"TC-{tc_id:02d} — Normalised Radar\n{meta['label']}",
                 pad=18, fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=7.5, frameon=True)

    fig.tight_layout()
    out_path = out_dir / f"tc{tc_id:02d}_radar.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved TC-{tc_id:02d} radar → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tc",  nargs="*", type=int, default=None,
                   help="TC IDs to plot (default: all 1-34)")
    p.add_argument("--input", default="test_case_results",
                   help="Directory containing tc{id}_summary.csv files")
    p.add_argument("--out",   default="test_case_results/plots/per_tc",
                   help="Output directory for per-TC plots")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.input)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover available TC summary files
    all_csvs = sorted(results_dir.glob("tc??_summary.csv"))
    if not all_csvs:
        print(f"No tc??_summary.csv files found in {results_dir}")
        return

    # extract TC IDs
    available = {}
    for csv in all_csvs:
        try:
            tc_id = int(csv.stem[2:4])   # "tc01_summary" → 1
            available[tc_id] = csv
        except ValueError:
            pass

    ids_to_plot = sorted(args.tc) if args.tc else sorted(available.keys())
    print(f"Plotting {len(ids_to_plot)} test cases → {out_dir}\n")

    for tc_id in ids_to_plot:
        if tc_id not in available:
            print(f"  TC-{tc_id:02d}: summary CSV not found — skipping")
            continue
        csv_path = available[tc_id]
        print(f"\nTC-{tc_id:02d}:")
        try:
            plot_tc(tc_id, csv_path, out_dir)
            plot_tc_stats_table(tc_id, csv_path, out_dir)
            plot_tc_ci_dotplot(tc_id, csv_path, out_dir)
            plot_tc_radar(tc_id, csv_path, out_dir)
        except Exception as e:
            print(f"  ERROR in TC-{tc_id:02d}: {e}")

    print(f"\nDone. All plots saved in {out_dir}/")
    print("Files per TC:")
    print("  tc{id}_comparison.png  — 8-panel bar chart")
    print("  tc{id}_stats_table.png — statistical table")
    print("  tc{id}_ci_dotplot.png  — 95% CI dot-plot")
    print("  tc{id}_radar.png       — normalised radar chart")


if __name__ == "__main__":
    main()
