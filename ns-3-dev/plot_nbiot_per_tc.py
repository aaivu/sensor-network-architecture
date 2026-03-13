#!/usr/bin/env python3
"""
NB-IoT Per-Test-Case Graph Generator
====================================
Creates one 8-panel comparison figure per NB-IoT test case.

Panels:
  1. PDR (%)
  2. Avg MCS (NB-IoT analogue to LoRa SF)
  3. Avg TX Power (dBm)
  4. Total Energy (mJ)
  5. Energy / Success (mJ)
  6. Avg SNR (dB)
  7. Avg Time-on-Air (ms)
  8. Median SNR Margin (dB)

PDR is read from nbiot_sweep_results/tcXX_summary.csv.
The remaining metrics are computed from per-policy NB-IoT dataset CSVs.
If a test case has no matching datasets yet, the script runs one NB-IoT probe
simulation for that test case to regenerate them.

Usage:
    cd ns-3-dev
    /path/to/python plot_nbiot_per_tc.py
    /path/to/python plot_nbiot_per_tc.py --tc 1 8 31
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = ["no_la", "3gpp", "ddqn", "ppo", "marl"]
POLICY_LABELS = {
    "no_la": "SDBP",
    "3gpp": "3GPP",
    "ddqn": "DDQN",
    "ppo": "PPO",
    "marl": "MARL",
}
POLICY_COLORS = {
    "no_la": "#e74c3c",
    "3gpp": "#e67e22",
    "ddqn": "#2980b9",
    "ppo": "#27ae60",
    "marl": "#8e44ad",
}
DATASET_PREFIX = {
    "no_la": "no_la",
    "3gpp": "default_la",
    "ddqn": "ddqn_la",
    "ppo": "ppo_la",
    "marl": "marl_la",
}
MCS_REQUIRED_SINR_DB = {
    0: -12.6,
    1: -10.5,
    2: -8.4,
    3: -6.3,
    4: -4.2,
    5: -2.1,
    6: 0.0,
    7: 2.1,
    8: 4.2,
    9: 6.0,
    10: 8.0,
}
PANELS = [
    ("pdr_pct", "PDR (%)", True),
    ("avg_mcs", "Avg MCS (SF analogue)", True),
    ("avg_tx_power_dbm", "Avg TX Power (dBm)", False),
    ("total_energy_mj", "Total Energy (mJ)", False),
    ("energy_per_success_mj", "Energy / Success (mJ)", False),
    ("avg_snr_db", "Avg SNR (dB)", True),
    ("avg_toa_ms", "Avg Time-on-Air (ms)", False),
    ("snr_margin_median_db", "Median SNR Margin (dB)", True),
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    }
)

NS3_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = NS3_DIR / "nbiot_sweep_results"
DATASETS_DIR = NS3_DIR / "nbiot_datasets"
NS3_BIN = NS3_DIR / "ns3"
PYTHON_BIN = shutil.which("python3.13") or shutil.which("python3") or "python3"
DEFAULT_PROGRAM = "advanced-ddqn-per-adr-nbiot"


def _fmt_tc(value: str | int) -> str:
    text = str(value).strip().lower()
    if text.startswith("tc"):
        text = text[2:]
    return text.zfill(2)


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _dataset_files_for_tc(tc: str) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {key: [] for key in POLICY_ORDER}
    if not DATASETS_DIR.exists():
        return files

    for policy in POLICY_ORDER:
        prefix = DATASET_PREFIX[policy]
        candidates = sorted(DATASETS_DIR.glob(f"{prefix}_tc{tc}_*_dataset.csv"))
        files[policy] = [path for path in candidates if not path.name.endswith("_environment.csv")]
    return files


def _has_all_policy_datasets(tc: str) -> bool:
    grouped = _dataset_files_for_tc(tc)
    return all(grouped[policy] for policy in POLICY_ORDER)


def _run_probe_for_tc(tc: str, meta: pd.Series) -> None:
    csv_name = f"tc{tc}_per_tc_plot_dataset.csv"
    total_interferers = int(meta["wifi"]) + int(meta["lte"])
    cmd = [
        PYTHON_BIN,
        str(NS3_BIN),
        "run",
        (
            f"{DEFAULT_PROGRAM} "
            f"--nUEs={int(meta['devices'])} "
            f"--cellRadius={int(meta['radius_m'])} "
            f"--lteInterferers={total_interferers} "
            f"--appPeriod={int(meta['period_s'])} "
            f"--simulationTime={int(meta['sim_time_s'])} "
            f"--adr=all "
            f"--csvFile={csv_name}"
        ),
    ]

    print(f"  Missing NB-IoT datasets for TC-{tc}; running one probe simulation...")
    result = subprocess.run(cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        tail = " | ".join(stderr[-3:]) if stderr else "no stderr"
        raise RuntimeError(f"probe simulation failed for TC-{tc}: {tail}")


def _load_policy_dataset(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _snr_margin_series(df: pd.DataFrame) -> pd.Series:
    if "sinr_db" not in df.columns or "mcs" not in df.columns:
        return pd.Series(dtype=float)

    sinr = pd.to_numeric(df["sinr_db"], errors="coerce")
    mcs = pd.to_numeric(df["mcs"], errors="coerce")
    reps = pd.to_numeric(df.get("repetitions", 1), errors="coerce").fillna(1.0)
    thresholds = mcs.map(
        lambda value: MCS_REQUIRED_SINR_DB.get(int(value), np.nan) if pd.notna(value) else np.nan
    )
    effective_sinr = sinr + 10.0 * np.log10(reps.clip(lower=1.0))
    margins = effective_sinr - thresholds
    if "data_success" in df.columns:
        success = pd.to_numeric(df["data_success"], errors="coerce")
        good = success == 1
        if good.any():
            margins = margins[good]
    return margins.dropna()


def _compute_dataset_metrics(df: pd.DataFrame, pdr_pct: float) -> Dict[str, float]:
    if df.empty:
        return {
            "pdr_pct": pdr_pct,
            "avg_mcs": np.nan,
            "avg_tx_power_dbm": np.nan,
            "total_energy_mj": np.nan,
            "energy_per_success_mj": np.nan,
            "avg_snr_db": np.nan,
            "avg_toa_ms": np.nan,
            "snr_margin_median_db": np.nan,
        }

    mcs = pd.to_numeric(df.get("mcs"), errors="coerce")
    tx_power = pd.to_numeric(df.get("tx_power_dbm"), errors="coerce")
    energy = pd.to_numeric(df.get("energy_consumed_mj"), errors="coerce")
    sinr = pd.to_numeric(df.get("sinr_db"), errors="coerce")
    airtime = pd.to_numeric(df.get("airtime_ms"), errors="coerce")
    success = pd.to_numeric(df.get("data_success"), errors="coerce")

    total_energy = float(energy.sum()) if energy.notna().any() else np.nan
    success_count = int((success == 1).sum()) if success.notna().any() else 0
    margins = _snr_margin_series(df)

    return {
        "pdr_pct": pdr_pct,
        "avg_mcs": float(mcs.mean()) if mcs.notna().any() else np.nan,
        "avg_tx_power_dbm": float(tx_power.mean()) if tx_power.notna().any() else np.nan,
        "total_energy_mj": total_energy,
        "energy_per_success_mj": (
            (total_energy / success_count)
            if success_count > 0 and np.isfinite(total_energy)
            else np.nan
        ),
        "avg_snr_db": float(sinr.mean()) if sinr.notna().any() else np.nan,
        "avg_toa_ms": float(airtime.mean()) if airtime.notna().any() else np.nan,
        "snr_margin_median_db": float(margins.median()) if not margins.empty else np.nan,
    }


def _load_tc_summary(summary_file: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_file)
    df["policy_key"] = df["policy_key"].astype(str)
    df["pdr_mean"] = pd.to_numeric(df["pdr_mean"], errors="coerce")
    return df


def _collect_tc_metrics(summary_file: Path) -> pd.DataFrame:
    summary = _load_tc_summary(summary_file)
    meta = summary.iloc[0]
    tc = _fmt_tc(meta["tc"])

    if not _has_all_policy_datasets(tc):
        _run_probe_for_tc(tc, meta)

    datasets = _dataset_files_for_tc(tc)
    rows = []
    for policy in POLICY_ORDER:
        pdr_row = summary[summary["policy_key"] == policy]
        pdr_pct = _safe_float(pdr_row.iloc[0]["pdr_mean"]) if not pdr_row.empty else np.nan
        df = _load_policy_dataset(datasets[policy])
        metrics = _compute_dataset_metrics(df, pdr_pct)
        rows.append(
            {
                "tc": tc,
                "scenario": meta.get("scenario", ""),
                "policy_key": policy,
                "policy_name": POLICY_LABELS[policy],
                "devices": int(meta["devices"]),
                "radius_m": int(meta["radius_m"]),
                "wifi": int(meta["wifi"]),
                "lte": int(meta["lte"]),
                "period_s": int(meta["period_s"]),
                "sim_time_s": int(meta["sim_time_s"]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _annotate_bars(ax, bars) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        label = f"{value:.1f}" if abs(value) < 1000 else f"{value / 1000.0:.1f}k"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ymax * 0.015, 0.01),
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )


def _plot_panel(ax, table: pd.DataFrame, metric: str, title: str, higher_is_better: bool) -> None:
    values = []
    colors = []
    for policy in POLICY_ORDER:
        row = table[table["policy_key"] == policy]
        values.append(_safe_float(row.iloc[0][metric]) if not row.empty else np.nan)
        colors.append(POLICY_COLORS[policy])

    x = np.arange(len(POLICY_ORDER))
    clean_values = [0.0 if not np.isfinite(v) else v for v in values]
    bars = ax.bar(
        x, clean_values, color=colors, edgecolor="white", linewidth=0.5, alpha=0.88, zorder=3
    )
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            bars[idx].set_alpha(0.18)

    finite = [(idx, value) for idx, value in enumerate(values) if np.isfinite(value)]
    if finite:
        best_idx = max(finite, key=lambda item: item[1] if higher_is_better else -item[1])[0]
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER], rotation=28, ha="right")
    ax.set_title(title, pad=5)
    finite_vals = [value for value in values if np.isfinite(value)]
    if finite_vals:
        vmin = min(finite_vals)
        vmax = max(finite_vals)
        span = vmax - vmin
        pad = max(abs(vmax) * 0.18, span * 0.25, 0.5)
        bottom = min(0.0, vmin - pad * 0.25)
        ax.set_ylim(bottom=bottom, top=vmax + pad)
        _annotate_bars(ax, bars)


def _plot_tc(table: pd.DataFrame, out_dir: Path) -> Path:
    meta = table.iloc[0]
    tc = _fmt_tc(meta["tc"])
    fig = plt.figure(figsize=(14.0, 8.8))
    axes = [fig.add_subplot(2, 4, idx + 1) for idx in range(8)]

    fig.text(
        0.5,
        0.97,
        f"TC-{tc}  |  {meta['scenario']}",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        (
            f"{int(meta['devices'])} UEs, radius={int(meta['radius_m'])} m, "
            f"WiFi={int(meta['wifi'])}, LTE={int(meta['lte'])}, "
            f"period={int(meta['period_s'])} s, sim={int(meta['sim_time_s'])} s"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    for ax, (metric, title, higher_is_better) in zip(axes, PANELS):
        _plot_panel(ax, table, metric, title, higher_is_better)

    fig.text(
        0.5,
        0.045,
        "Black border marks the best method for that metric. Avg MCS is used as the NB-IoT counterpart to LoRa SF.",
        ha="center",
        fontsize=7.5,
        color="#666666",
        style="italic",
    )
    fig.subplots_adjust(top=0.88, bottom=0.11, hspace=0.48, wspace=0.34)

    out_path = out_dir / f"tc{tc}_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tc", nargs="*", type=int, default=None, help="TC IDs to plot, default: all"
    )
    parser.add_argument(
        "--input", default="nbiot_sweep_results", help="Directory containing tcXX_summary.csv"
    )
    parser.add_argument(
        "--out", default="nbiot_sweep_results/plots/nbiot_per_tc", help="Output directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(input_dir.glob("tc??_summary.csv"))
    if not summary_files:
        raise SystemExit(f"No tc??_summary.csv files found in {input_dir}")

    selected = {_fmt_tc(tc) for tc in args.tc} if args.tc else None
    rows = []

    chosen_files = []
    for path in summary_files:
        tc = _fmt_tc(path.stem.split("_")[0])
        if selected and tc not in selected:
            continue
        chosen_files.append(path)

    print(f"Plotting {len(chosen_files)} NB-IoT test cases -> {out_dir}")
    for summary_file in chosen_files:
        tc = _fmt_tc(summary_file.stem.split("_")[0])
        print(f"\nTC-{tc}:")
        table = _collect_tc_metrics(summary_file)
        rows.append(table)
        out_path = _plot_tc(table, out_dir)
        print(f"  Saved TC-{tc} -> {out_path}")

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        summary_csv = out_dir / "nbiot_per_tc_metrics.csv"
        combined.to_csv(summary_csv, index=False)
        print(f"\nSaved metric summary -> {summary_csv}")


if __name__ == "__main__":
    main()
