#!/usr/bin/env python3
"""
LoRaWAN ADR Test Case Results Visualizer
==========================================
Reads grand_summary_pdr.csv (and per-TC summary CSVs) from test_case_results/
and generates a comprehensive set of comparison graphs.

Usage:
    python3 plot_test_results.py
    python3 plot_test_results.py --input test_case_results/grand_summary_pdr.csv
    python3 plot_test_results.py --out my_plots/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── aesthetics ──────────────────────────────────────────────────────────────
ADR_KEYS   = ["no_adr", "classical_adr", "ddqn_adr", "ppo_adr", "marl_adr"]
ADR_LABELS = {
    "no_adr":       "No ADR",
    "classical_adr":"Classical ADR",
    "ddqn_adr":     "DDQN-PER",
    "ppo_adr":      "PPO",
    "marl_adr":     "MARL",
}
ADR_COLORS = {
    "no_adr":       "#e74c3c",   # red
    "classical_adr":"#e67e22",   # orange
    "ddqn_adr":     "#2980b9",   # blue
    "ppo_adr":      "#27ae60",   # green
    "marl_adr":     "#8e44ad",   # purple
}
ADR_MARKERS = {
    "no_adr": "o", "classical_adr": "s",
    "ddqn_adr": "^", "ppo_adr": "D", "marl_adr": "X",
}

RL_COLORS = {"best": "#2ecc71", "good": "#f1c40f",
             "neutral": "#95a5a6", "fail": "#e74c3c"}

GROUP_LABELS = {
    "G1":  "Baseline",
    "G2":  "Building Size",
    "G3":  "WiFi Scaling\n(100m)",
    "G4":  "WiFi Scaling\n(500m)",
    "G5":  "Device Density",
    "G6":  "Traffic Rate",
    "G7":  "LTE Bleed",
    "G8":  "RL Convergence",
    "G9":  "Stress Tests",
}
TC_GROUPS = {
    1: "G1", 2: "G1", 3: "G1",
    4: "G2", 5: "G2", 6: "G2", 7: "G2", 8: "G2",
    9: "G3", 10: "G3", 11: "G3", 12: "G3", 13: "G3",
    14: "G4", 15: "G4",
    16: "G5", 17: "G5", 18: "G5", 19: "G5",
    20: "G6", 21: "G6", 22: "G6", 23: "G6",
    24: "G7", 25: "G7", 26: "G7", 27: "G7",
    28: "G8", 29: "G8", 30: "G8",
    31: "G9", 32: "G9", 33: "G9", 34: "G9",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


# ── helpers ─────────────────────────────────────────────────────────────────

def _pivot(df: pd.DataFrame, metric: str = "pdr_pct") -> pd.DataFrame:
    """Return wide DataFrame: index=tc_id, columns=adr_key, values=mean."""
    sub = df[df["metric"] == metric].copy()
    return sub.pivot_table(index="tc_id", columns="adr_key", values="mean")


def _pivot_err(df: pd.DataFrame, metric: str = "pdr_pct") -> pd.DataFrame:
    """Return wide DataFrame of std values."""
    sub = df[df["metric"] == metric].copy()
    return sub.pivot_table(index="tc_id", columns="adr_key", values="std")


def _tc_meta(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tc_id with metadata columns."""
    cols = ["tc_id", "label", "scenario", "expected_rl",
            "nDevices", "radius", "wifiInterferers", "lteInterferers",
            "appPeriod", "simulationTime"]
    return df[cols].drop_duplicates("tc_id").set_index("tc_id").sort_index()


def _short_label(label: str, max_len: int = 22) -> str:
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def _legend_patches():
    return [mpatches.Patch(color=ADR_COLORS[k], label=ADR_LABELS[k])
            for k in ADR_KEYS if k in ADR_COLORS]


def _save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 1: Grouped bar chart — PDR by ADR type, all TCs ────────────────────

def plot_grouped_bar_all_tcs(df: pd.DataFrame, out: Path):
    wide  = _pivot(df)
    wide_err = _pivot_err(df)
    meta  = _tc_meta(df)

    tc_ids = sorted(wide.index.tolist())
    n      = len(tc_ids)
    n_adr  = len(ADR_KEYS)
    width  = 0.14
    x      = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(18, n * 0.55), 5))

    for i, key in enumerate(ADR_KEYS):
        if key not in wide.columns:
            continue
        vals = [wide.loc[tc, key] if tc in wide.index else np.nan for tc in tc_ids]
        errs = [wide_err.loc[tc, key] if tc in wide_err.index else 0 for tc in tc_ids]
        offset = (i - n_adr / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width - 0.01,
               color=ADR_COLORS[key], label=ADR_LABELS[key],
               yerr=errs, capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=60, ha="right")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR Comparison — All Test Cases × ADR Methods (mean ± std over runs)")
    ax.legend(handles=_legend_patches(), loc="lower right", ncol=5)

    # colour TC labels by expected_rl
    for tick, tc_id in zip(ax.get_xticklabels(), tc_ids):
        rl = meta.loc[tc_id, "expected_rl"] if tc_id in meta.index else "neutral"
        tick.set_color(RL_COLORS.get(rl, "black"))

    # RL legend
    rl_patches = [mpatches.Patch(color=v, label=f"RL {k}") for k, v in RL_COLORS.items()]
    ax2_legend = ax.legend(handles=rl_patches, title="RL performance\n(TC label colour)",
                           loc="lower left", ncol=4, fontsize=7)
    ax.add_artist(ax2_legend)
    ax.legend(handles=_legend_patches(), loc="upper right", ncol=5)

    _save(fig, out, "01_grouped_bar_all_tcs.png")


# ── Plot 2: Heatmap — PDR for every (TC × ADR) ──────────────────────────────

def plot_heatmap(df: pd.DataFrame, out: Path):
    wide = _pivot(df)
    meta = _tc_meta(df)

    # reorder columns
    cols = [k for k in ADR_KEYS if k in wide.columns]
    wide = wide[cols]

    # short row labels
    row_labels = [f"TC{tc} – {_short_label(meta.loc[tc,'label'])}"
                  for tc in wide.index if tc in meta.index]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.6 + 1, len(wide) * 0.38 + 1.5))
    data = wide.values.astype(float)

    vmin = np.nanmin(data) - 1
    vmax = np.nanmax(data) + 1
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([ADR_LABELS[k] for k in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title("PDR Heatmap — All Test Cases × ADR Methods")

    # annotate cells
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color="black")

    plt.colorbar(im, ax=ax, label="Mean PDR (%)", shrink=0.6)
    _save(fig, out, "02_heatmap_pdr.png")


# ── Plot 3: Line chart — PDR vs TC within each group ────────────────────────

def plot_lines_per_group(df: pd.DataFrame, out: Path):
    wide = _pivot(df)
    meta = _tc_meta(df)

    groups = sorted(set(TC_GROUPS.values()))
    n_groups = len(groups)
    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.5))
    axes = axes.flatten()

    for idx, grp in enumerate(groups):
        ax = axes[idx]
        tc_ids = sorted([tc for tc, g in TC_GROUPS.items() if g == grp
                         and tc in wide.index])
        if not tc_ids:
            ax.set_visible(False)
            continue

        x_labels = [f"TC{tc}" for tc in tc_ids]
        x = np.arange(len(tc_ids))

        for key in ADR_KEYS:
            if key not in wide.columns:
                continue
            yvals = [wide.loc[tc, key] if tc in wide.index else np.nan
                     for tc in tc_ids]
            ax.plot(x, yvals, marker=ADR_MARKERS[key],
                    color=ADR_COLORS[key], label=ADR_LABELS[key],
                    linewidth=1.5, markersize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_title(f"{grp}: {GROUP_LABELS[grp]}")
        ax.set_ylabel("PDR (%)")

        # shade RL expectation
        for xi, tc_id in enumerate(tc_ids):
            rl = meta.loc[tc_id, "expected_rl"] if tc_id in meta.index else "neutral"
            ax.axvspan(xi - 0.4, xi + 0.4,
                       alpha=0.07, color=RL_COLORS.get(rl, "grey"), zorder=0)

    # shared legend
    handles = [plt.Line2D([0], [0], color=ADR_COLORS[k], marker=ADR_MARKERS[k],
                          linewidth=1.5, label=ADR_LABELS[k]) for k in ADR_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.015), frameon=True)

    # hide empty
    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("PDR per TC Group — Line Comparison by ADR Method\n"
                 "(shaded background = expected RL performance)", y=1.01)
    fig.tight_layout()
    _save(fig, out, "03_lines_per_group.png")


# ── Plot 4: WiFi scaling — PDR vs nWiFi interferers ─────────────────────────

def plot_wifi_scaling(df: pd.DataFrame, out: Path):
    """TC 9–13 (radius 100m), TC 14–15 (radius 500m)."""
    wide = _pivot(df)
    meta = _tc_meta(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, tc_list, title in [
        (axes[0], [9, 10, 11, 12, 13], "WiFi Scaling — 100m radius"),
        (axes[1], [14, 15],            "WiFi Scaling — 500m radius"),
    ]:
        x_vals = [meta.loc[tc, "wifiInterferers"] for tc in tc_list if tc in meta.index]
        tcs    = [tc for tc in tc_list if tc in wide.index]

        for key in ADR_KEYS:
            if key not in wide.columns:
                continue
            yvals = [wide.loc[tc, key] for tc in tcs]
            ax.plot(x_vals[:len(yvals)], yvals,
                    marker=ADR_MARKERS[key], color=ADR_COLORS[key],
                    label=ADR_LABELS[key], linewidth=1.8, markersize=6)

        ax.set_xlabel("Number of WiFi Interferers")
        ax.set_ylabel("Mean PDR (%)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("PDR vs WiFi Interference Density")
    fig.tight_layout()
    _save(fig, out, "04_wifi_scaling.png")


# ── Plot 5: Device density scaling ──────────────────────────────────────────

def plot_device_density(df: pd.DataFrame, out: Path):
    """TC 16–19."""
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_list = [16, 17, 18, 19]
    tcs = [tc for tc in tc_list if tc in wide.index]
    x_vals = [meta.loc[tc, "nDevices"] for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        yvals = [wide.loc[tc, key] for tc in tcs]
        ax.plot(x_vals, yvals, marker=ADR_MARKERS[key],
                color=ADR_COLORS[key], label=ADR_LABELS[key],
                linewidth=1.8, markersize=6)

    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR vs Device Density (100m radius, 8 WiFi APs)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "05_device_density_scaling.png")


# ── Plot 6: Traffic rate (app period) scaling ────────────────────────────────

def plot_traffic_rate(df: pd.DataFrame, out: Path):
    """TC 20–23."""
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_list = [20, 21, 22, 23]
    tcs = [tc for tc in tc_list if tc in wide.index]
    x_vals = [meta.loc[tc, "appPeriod"] for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        yvals = [wide.loc[tc, key] for tc in tcs]
        ax.plot(x_vals, yvals, marker=ADR_MARKERS[key],
                color=ADR_COLORS[key], label=ADR_LABELS[key],
                linewidth=1.8, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Transmission Interval (s) — log scale")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR vs Traffic Rate (100m, 8 WiFi APs, 30 devices)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "06_traffic_rate_scaling.png")


# ── Plot 7: LTE bleed scaling ────────────────────────────────────────────────

def plot_lte_scaling(df: pd.DataFrame, out: Path):
    """TC 24–27."""
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_list = [24, 25, 26, 27]
    tcs = [tc for tc in tc_list if tc in wide.index]
    x_vals = [meta.loc[tc, "lteInterferers"] for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        yvals = [wide.loc[tc, key] for tc in tcs]
        ax.plot(x_vals, yvals, marker=ADR_MARKERS[key],
                color=ADR_COLORS[key], label=ADR_LABELS[key],
                linewidth=1.8, markersize=6)

    ax.set_xlabel("Number of LTE Interferers")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR vs LTE Interference (100m, 10 WiFi APs, 30 devices)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "07_lte_scaling.png")


# ── Plot 8: RL convergence — PDR vs simulation time ─────────────────────────

def plot_convergence(df: pd.DataFrame, out: Path):
    """TC 28–30."""
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_list = [28, 29, 30]
    tcs = [tc for tc in tc_list if tc in wide.index]
    x_vals = [meta.loc[tc, "simulationTime"] for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        yvals = [wide.loc[tc, key] for tc in tcs]
        ax.plot(x_vals, yvals, marker=ADR_MARKERS[key],
                color=ADR_COLORS[key], label=ADR_LABELS[key],
                linewidth=1.8, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Simulation Time (s) — log scale")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("RL Convergence: PDR vs Simulation Duration\n"
                 "(100m, 10 WiFi APs, 30 devices, 60s period)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "08_rl_convergence.png")


# ── Plot 9: Building radius scaling — PDR vs radius (no interference) ────────

def plot_radius_scaling(df: pd.DataFrame, out: Path):
    """TC 1 (30m), 2 (100m), 3 (500m) — no WiFi."""
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_list = [1, 2, 3]
    tcs = [tc for tc in tc_list if tc in wide.index]
    x_vals = [meta.loc[tc, "radius"] for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        yvals = [wide.loc[tc, key] for tc in tcs]
        ax.plot(x_vals, yvals, marker=ADR_MARKERS[key],
                color=ADR_COLORS[key], label=ADR_LABELS[key],
                linewidth=1.8, markersize=6)

    ax.set_xlabel("Deployment Radius (m)")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR vs Coverage Radius — Baseline (no interference)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "09_radius_scaling_baseline.png")


# ── Plot 10: DDQN vs Classical ADR — advantage bar chart ────────────────────

def plot_rl_advantage(df: pd.DataFrame, out: Path):
    """
    For each TC show (DDQN PDR - Classical PDR) as bars.
    Positive = DDQN wins, negative = Classical wins.
    """
    wide = _pivot(df)
    meta = _tc_meta(df)
    tc_ids = sorted(wide.index.tolist())

    if "ddqn_adr" not in wide.columns or "classical_adr" not in wide.columns:
        print("  Skipping plot 10: missing ddqn_adr or classical_adr columns.")
        return

    diff_ddqn = [wide.loc[tc, "ddqn_adr"] - wide.loc[tc, "classical_adr"]
                 if tc in wide.index else 0 for tc in tc_ids]
    diff_ppo  = [wide.loc[tc, "ppo_adr"]  - wide.loc[tc, "classical_adr"]
                 if tc in wide.index and "ppo_adr" in wide.columns else 0
                 for tc in tc_ids]
    diff_marl = [wide.loc[tc, "marl_adr"] - wide.loc[tc, "classical_adr"]
                 if tc in wide.index and "marl_adr" in wide.columns else 0
                 for tc in tc_ids]

    x = np.arange(len(tc_ids))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(18, len(tc_ids) * 0.55), 4.5))
    ax.bar(x - w, diff_ddqn, w, label="DDQN-PER",
           color=[("#2980b9" if d >= 0 else "#aec6cf") for d in diff_ddqn])
    ax.bar(x,     diff_ppo,  w, label="PPO",
           color=[("#27ae60" if d >= 0 else "#a9dfbf") for d in diff_ppo])
    ax.bar(x + w, diff_marl, w, label="MARL",
           color=[("#8e44ad" if d >= 0 else "#d7bde2") for d in diff_marl])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=60, ha="right")
    ax.set_ylabel("PDR Advantage over Classical ADR (pp)")
    ax.set_title("RL Advantage over Classical ADR\n"
                 "(positive = RL better, negative = Classical better)")
    ax.legend()

    # colour TC labels
    for tick, tc_id in zip(ax.get_xticklabels(), tc_ids):
        rl = meta.loc[tc_id, "expected_rl"] if tc_id in meta.index else "neutral"
        tick.set_color(RL_COLORS.get(rl, "black"))

    _save(fig, out, "10_rl_advantage_over_classical.png")


# ── Plot 11: DDQN vs No-ADR advantage ───────────────────────────────────────

def plot_rl_vs_no_adr(df: pd.DataFrame, out: Path):
    wide  = _pivot(df)
    meta  = _tc_meta(df)
    tc_ids = sorted(wide.index.tolist())

    if "ddqn_adr" not in wide.columns or "no_adr" not in wide.columns:
        return

    diff_ddqn = [wide.loc[tc, "ddqn_adr"] - wide.loc[tc, "no_adr"]
                 if tc in wide.index else 0 for tc in tc_ids]

    x = np.arange(len(tc_ids))
    colors = ["#2980b9" if d >= 0 else "#e74c3c" for d in diff_ddqn]

    fig, ax = plt.subplots(figsize=(max(18, len(tc_ids) * 0.55), 4))
    ax.bar(x, diff_ddqn, color=colors, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=60, ha="right")
    ax.set_ylabel("DDQN PDR − No-ADR PDR (pp)")
    ax.set_title("DDQN-PER vs No ADR — PDR Difference per Test Case\n"
                 "(blue = DDQN better, red = No-ADR better)")

    for tick, tc_id in zip(ax.get_xticklabels(), tc_ids):
        rl = meta.loc[tc_id, "expected_rl"] if tc_id in meta.index else "neutral"
        tick.set_color(RL_COLORS.get(rl, "black"))

    _save(fig, out, "11_ddqn_vs_no_adr.png")


# ── Plot 12: Error-bar summary — PDR by ADR type, grouped by scenario group ──

def plot_errorbar_by_group(df: pd.DataFrame, out: Path):
    wide     = _pivot(df)
    wide_std = _pivot_err(df)
    meta     = _tc_meta(df)

    groups   = sorted(set(TC_GROUPS.values()))
    n_groups = len(groups)
    ncols    = 3
    nrows    = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2))
    axes = axes.flatten()

    for idx, grp in enumerate(groups):
        ax = axes[idx]
        tc_ids = sorted([tc for tc, g in TC_GROUPS.items()
                         if g == grp and tc in wide.index])
        if not tc_ids:
            ax.set_visible(False)
            continue

        x = np.arange(len(tc_ids))
        for i, key in enumerate(ADR_KEYS):
            if key not in wide.columns:
                continue
            yvals = [wide.loc[tc, key]     for tc in tc_ids]
            yerrs = [wide_std.loc[tc, key] if tc in wide_std.index else 0
                     for tc in tc_ids]
            offset = (i - len(ADR_KEYS) / 2 + 0.5) * 0.12
            ax.errorbar(x + offset, yvals, yerr=yerrs,
                        fmt=ADR_MARKERS[key], color=ADR_COLORS[key],
                        label=ADR_LABELS[key], capsize=3,
                        linewidth=1.2, markersize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=30)
        ax.set_title(f"{grp}: {GROUP_LABELS[grp]}")
        ax.set_ylabel("PDR (%)")

    handles = [plt.Line2D([0], [0], color=ADR_COLORS[k], marker=ADR_MARKERS[k],
                          linewidth=1.2, label=ADR_LABELS[k]) for k in ADR_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02))

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("PDR with Error Bars (mean ± std) per Scenario Group", y=1.01)
    fig.tight_layout()
    _save(fig, out, "12_errorbar_by_group.png")


# ── Plot 13: CI-95 ribbon chart ──────────────────────────────────────────────

def plot_ci_ribbon(df: pd.DataFrame, out: Path):
    """Ribbon (CI-95 band) for each ADR type across all TCs."""
    sub = df[df["metric"] == "pdr_pct"].copy()
    tc_ids = sorted(sub["tc_id"].unique())
    x = np.arange(len(tc_ids))

    fig, ax = plt.subplots(figsize=(max(18, len(tc_ids) * 0.55), 4.5))

    for key in ADR_KEYS:
        d = sub[sub["adr_key"] == key].set_index("tc_id")
        if d.empty:
            continue
        means = [d.loc[tc, "mean"]     if tc in d.index else np.nan for tc in tc_ids]
        lows  = [d.loc[tc, "ci95_low"]  if tc in d.index else np.nan for tc in tc_ids]
        highs = [d.loc[tc, "ci95_high"] if tc in d.index else np.nan for tc in tc_ids]

        ax.plot(x, means, color=ADR_COLORS[key],
                label=ADR_LABELS[key], linewidth=1.6,
                marker=ADR_MARKERS[key], markersize=4)
        ax.fill_between(x, lows, highs, color=ADR_COLORS[key], alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=60, ha="right")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("PDR with 95% Confidence Interval Ribbons — All Test Cases")
    ax.legend(loc="upper right", ncol=5)
    _save(fig, out, "13_ci95_ribbon.png")


# ── Plot 14: Radar chart — average performance per ADR type ─────────────────

def plot_radar(df: pd.DataFrame, out: Path):
    """
    Radar over 5 scenario dimension averages:
      G3 WiFi-stress, G5 device-density, G7 LTE, G8 convergence, G9 stress.
    """
    sub = df[df["metric"] == "pdr_pct"].copy()
    wide = sub.pivot_table(index="tc_id", columns="adr_key", values="mean")

    dim_tcs = {
        "WiFi\nStress":    [11, 12, 13],
        "Device\nDensity": [17, 18, 19],
        "LTE\nBleed":      [25, 26, 27],
        "RL\nConvergence": [28, 29, 30],
        "Worst\nCase":     [31, 32],
    }
    dims = list(dim_tcs.keys())
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for key in ADR_KEYS:
        if key not in wide.columns:
            continue
        vals = []
        for d, tcs in dim_tcs.items():
            v = np.nanmean([wide.loc[tc, key] for tc in tcs if tc in wide.index])
            vals.append(v if not np.isnan(v) else 0)
        vals += vals[:1]
        ax.plot(angles, vals, color=ADR_COLORS[key],
                linewidth=1.8, label=ADR_LABELS[key])
        ax.fill(angles, vals, color=ADR_COLORS[key], alpha=0.07)

    ax.set_thetagrids(np.degrees(angles[:-1]), dims)
    ax.set_title("Radar: Average PDR across Scenario Dimensions", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
    _save(fig, out, "14_radar_scenario_dimensions.png")


# ── Plot 15: Box plot — PDR distribution per ADR type over all TCs ──────────

def plot_boxplot_overall(df: pd.DataFrame, out: Path):
    sub = df[df["metric"] == "pdr_pct"].copy()
    data_by_adr = {ADR_LABELS[k]: sub[sub["adr_key"] == k]["mean"].dropna().values
                   for k in ADR_KEYS if k in sub["adr_key"].values}

    labels = list(data_by_adr.keys())
    data   = list(data_by_adr.values())
    colors = [ADR_COLORS[k] for k in ADR_KEYS if k in sub["adr_key"].values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bps = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                     widths=0.5, medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bps["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean PDR per TC (%)")
    ax.set_title("Distribution of PDR Across All Test Cases per ADR Method")
    _save(fig, out, "15_boxplot_pdr_overall.png")


# ── Plot 16: Scatter — WiFi interferers vs DDQN advantage ────────────────────

def plot_scatter_wifi_vs_advantage(df: pd.DataFrame, out: Path):
    sub  = df[df["metric"] == "pdr_pct"].copy()
    meta = _tc_meta(df)
    wide = sub.pivot_table(index="tc_id", columns="adr_key", values="mean")

    if "ddqn_adr" not in wide.columns or "classical_adr" not in wide.columns:
        return

    tc_ids   = sorted(wide.index)
    wifi_cnt = [meta.loc[tc, "wifiInterferers"] for tc in tc_ids if tc in meta.index]
    advantage= [wide.loc[tc, "ddqn_adr"] - wide.loc[tc, "classical_adr"]
                for tc in tc_ids if tc in meta.index]
    rl_exp   = [meta.loc[tc, "expected_rl"] for tc in tc_ids if tc in meta.index]

    colors   = [RL_COLORS.get(r, "grey") for r in rl_exp]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(wifi_cnt, advantage, c=colors, s=70, edgecolors="black",
                    linewidths=0.5, zorder=3)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    for r, color in RL_COLORS.items():
        ax.scatter([], [], c=color, label=f"RL: {r}", s=50, edgecolors="black")

    ax.set_xlabel("Number of WiFi Interferers")
    ax.set_ylabel("DDQN-PER PDR − Classical ADR PDR (pp)")
    ax.set_title("DDQN Advantage vs WiFi Interference Density")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "16_scatter_wifi_vs_ddqn_advantage.png")


# ── Plot 17: Grouped comparison — RL methods only ──────────────────────────

def plot_rl_only_comparison(df: pd.DataFrame, out: Path):
    rl_keys = ["ddqn_adr", "ppo_adr", "marl_adr"]
    sub = df[df["metric"] == "pdr_pct"].copy()
    wide = sub.pivot_table(index="tc_id", columns="adr_key", values="mean")
    meta = _tc_meta(df)
    tc_ids = sorted(wide.index.tolist())
    x = np.arange(len(tc_ids))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(18, len(tc_ids) * 0.55), 4.5))
    for i, key in enumerate(rl_keys):
        if key not in wide.columns:
            continue
        vals = [wide.loc[tc, key] for tc in tc_ids]
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, color=ADR_COLORS[key],
               label=ADR_LABELS[key], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{tc}" for tc in tc_ids], rotation=60, ha="right")
    ax.set_ylabel("Mean PDR (%)")
    ax.set_title("RL Methods Only (DDQN-PER vs PPO vs MARL)")
    ax.legend()

    for tick, tc_id in zip(ax.get_xticklabels(), tc_ids):
        rl = meta.loc[tc_id, "expected_rl"] if tc_id in meta.index else "neutral"
        tick.set_color(RL_COLORS.get(rl, "black"))

    _save(fig, out, "17_rl_methods_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_case_results/grand_summary_pdr.csv")
    parser.add_argument("--out",   default="test_case_results/plots")
    args = parser.parse_args()

    csv_path = Path(args.input)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"TC ids: {sorted(df['tc_id'].unique())}")
    print(f"ADR keys: {sorted(df['adr_key'].unique())}")
    print(f"Metrics: {sorted(df['metric'].unique())}")
    print(f"Saving plots to: {out_path}\n")

    plots = [
        ("01 Grouped bar — all TCs",         plot_grouped_bar_all_tcs),
        ("02 Heatmap",                        plot_heatmap),
        ("03 Lines per group",                plot_lines_per_group),
        ("04 WiFi scaling",                   plot_wifi_scaling),
        ("05 Device density",                 plot_device_density),
        ("06 Traffic rate",                   plot_traffic_rate),
        ("07 LTE scaling",                    plot_lte_scaling),
        ("08 RL convergence",                 plot_convergence),
        ("09 Radius scaling baseline",        plot_radius_scaling),
        ("10 RL advantage over Classical ADR",plot_rl_advantage),
        ("11 DDQN vs No-ADR",                 plot_rl_vs_no_adr),
        ("12 Error-bar by group",             plot_errorbar_by_group),
        ("13 CI-95 ribbon",                   plot_ci_ribbon),
        ("14 Radar",                          plot_radar),
        ("15 Box-plot overall",               plot_boxplot_overall),
        ("16 Scatter WiFi vs DDQN advantage", plot_scatter_wifi_vs_advantage),
        ("17 RL methods comparison",          plot_rl_only_comparison),
    ]

    for name, fn in plots:
        print(f"Plotting: {name}")
        try:
            fn(df, out_path)
        except Exception as e:
            print(f"  WARNING: {e}")

    print(f"\nDone. All plots saved in {out_path}/")


if __name__ == "__main__":
    main()
