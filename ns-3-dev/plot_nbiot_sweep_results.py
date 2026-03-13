#!/usr/bin/env python3
"""
NB-IoT Sweep Results Plotter (Per Test Case)
===========================================
Creates separate plots per test case comparing:
  - PDR (%)
  - Average energy per packet (mJ)
across policies: no_la, 3gpp, ddqn, ppo, marl.

Data sources:
  1) nbiot_sweep_results/tcXX_summary.csv  (primary for PDR)
  2) nbiot_sweep_results/full_summary_long.csv (optional fallback)
  3) nbiot_datasets/*tcXX*_dataset.csv (energy fallback)

Usage:
  python3 plot_nbiot_sweep_results.py
  python3 plot_nbiot_sweep_results.py --input nbiot_sweep_results --out nbiot_sweep_results/plots_per_tc
  python3 plot_nbiot_sweep_results.py --tc 01 08 31
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY_ORDER = ["no_la", "3gpp", "ddqn", "ppo", "marl"]
POLICY_LABEL = {
    "no_la": "SDBP",
    "3gpp": "3GPP",
    "ddqn": "DDQN",
    "ppo": "PPO",
    "marl": "MARL",
}
POLICY_COLOR = {
    "no_la": "#e74c3c",
    "3gpp": "#e67e22",
    "ddqn": "#2980b9",
    "ppo": "#27ae60",
    "marl": "#8e44ad",
}

DATASET_PREFIX_BY_POLICY = {
    "no_la": "no_la",
    "3gpp": "default_la",
    "ddqn": "ddqn_la",
    "ppo": "ppo_la",
    "marl": "marl_la",
}

ENERGY_CANDIDATE_COLS = [
    "energy_mean",
    "avg_energy_mj",
    "avg_energy_mJ",
    "total_energy_mJ",
    "energy_per_success_mJ",
    "mean_energy_mj",
    "mean_energy_mJ",
]

NS3_DIR = Path(__file__).parent.resolve()
NS3_BIN = NS3_DIR / "ns3"
PYTHON_BIN = shutil.which("python3.13") or shutil.which("python3") or "python3"
DEFAULT_PROGRAM = "advanced-ddqn-per-adr-nbiot"


def _fmt_tc(tc_value: str) -> str:
    s = str(tc_value).strip()
    if s.lower().startswith("tc"):
        s = s[2:]
    return s.zfill(2)


def _find_summary_files(input_dir: Path, wanted: Optional[List[str]]) -> List[Path]:
    files = sorted(input_dir.glob("tc*_summary.csv"))
    if not wanted:
        return files
    wanted_set = {_fmt_tc(t) for t in wanted}
    out = []
    for f in files:
        stem = f.stem  # tc01_summary
        tc = _fmt_tc(stem.replace("_summary", ""))
        if tc in wanted_set:
            out.append(f)
    return out


def _safe_float(value) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return np.nan


def _extract_energy_from_row(row: pd.Series) -> float:
    for col in ENERGY_CANDIDATE_COLS:
        if col in row:
            v = _safe_float(row[col])
            if np.isfinite(v):
                return v
    return np.nan


def _energy_from_datasets(datasets_dir: Path, tc: str) -> Dict[str, float]:
    """Try to recover per-policy average energy from nbiot_datasets CSVs."""
    energy = {k: np.nan for k in POLICY_ORDER}
    if not datasets_dir.exists():
        return energy

    for policy in POLICY_ORDER:
        prefix = DATASET_PREFIX_BY_POLICY[policy]
        # Pattern example: default_la_tc31_run005_dataset.csv
        candidates = sorted(datasets_dir.glob(f"{prefix}_tc{tc}_*_dataset.csv"))
        if not candidates:
            continue

        # If multiple files exist, aggregate all of them.
        values = []
        for csv_path in candidates:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "energy_consumed_mj" not in df.columns:
                continue
            col = pd.to_numeric(df["energy_consumed_mj"], errors="coerce").dropna()
            if not col.empty:
                values.append(float(col.mean()))

        if values:
            energy[policy] = float(np.mean(values))

    return energy


def _recover_energy_by_rerun(tc: str, table: pd.DataFrame, datasets_dir: Path) -> Dict[str, float]:
    """Run one simulation for this TC to regenerate per-policy datasets and extract energy."""
    required_cols = ["devices", "radius_m", "wifi", "lte", "period_s", "sim_time_s"]
    for col in required_cols:
        if col not in table.columns:
            return {k: np.nan for k in POLICY_ORDER}

    row = table.iloc[0]
    try:
        n_ues = int(float(row["devices"]))
        radius = int(float(row["radius_m"]))
        lte_interferers = int(float(row["wifi"])) + int(float(row["lte"]))
        app_period = int(float(row["period_s"]))
        sim_time = int(float(row["sim_time_s"]))
    except Exception:
        return {k: np.nan for k in POLICY_ORDER}

    csv_name = f"tc{tc}_energy_probe_dataset.csv"
    cmd = [
        PYTHON_BIN,
        str(NS3_BIN),
        "run",
        (
            f"{DEFAULT_PROGRAM} "
            f"--nUEs={n_ues} "
            f"--cellRadius={radius} "
            f"--lteInterferers={lte_interferers} "
            f"--appPeriod={app_period} "
            f"--simulationTime={sim_time} "
            f"--adr=all "
            f"--csvFile={csv_name}"
        ),
    ]

    print(f"Energy missing for TC{tc}: running one probe simulation to recover energy metrics...")
    try:
        result = subprocess.run(cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=1800)
    except Exception as exc:
        print(f"  WARNING: energy recovery run failed for TC{tc}: {exc}")
        return {k: np.nan for k in POLICY_ORDER}

    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        msg = " | ".join(tail) if tail else "no stderr"
        print(f"  WARNING: energy recovery run failed for TC{tc} (rc={result.returncode}): {msg}")
        return {k: np.nan for k in POLICY_ORDER}

    return _energy_from_datasets(datasets_dir, tc)


def _load_tc_summary(summary_file: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_file)

    # Normalize policy keys if necessary.
    if "policy_key" in df.columns:
        policy_col = "policy_key"
    elif "adr_key" in df.columns:
        policy_col = "adr_key"
    else:
        raise ValueError(f"No policy key column in {summary_file.name}")

    # Normalize PDR column names.
    if "pdr_mean" in df.columns:
        pdr_col = "pdr_mean"
    elif "mean" in df.columns and "metric" in df.columns:
        pdr_rows = df[df["metric"] == "pdr_pct"].copy()
        pdr_rows = pdr_rows.rename(columns={"mean": "pdr_mean"})
        pdr_rows["policy_key"] = pdr_rows[policy_col]
        pdr_rows["energy_mean"] = np.nan
        return pdr_rows[["tc", "scenario", "policy_key", "pdr_mean", "energy_mean"]]
    else:
        raise ValueError(f"No PDR mean column in {summary_file.name}")

    out = df.copy()
    out["policy_key"] = out[policy_col].astype(str)
    if "tc" not in out.columns:
        # Fallback from filename if missing
        tc_from_name = _fmt_tc(summary_file.stem.replace("_summary", ""))
        out["tc"] = tc_from_name

    out["pdr_mean"] = pd.to_numeric(out[pdr_col], errors="coerce")
    out["energy_mean"] = out.apply(_extract_energy_from_row, axis=1)
    if "scenario" not in out.columns:
        out["scenario"] = ""

    keep_cols = ["tc", "scenario", "policy_key", "pdr_mean", "energy_mean"]
    for col in ["devices", "radius_m", "wifi", "lte", "period_s", "sim_time_s"]:
        if col in out.columns:
            keep_cols.append(col)

    return out[keep_cols]


def _plot_tc(tc: str, scenario: str, table: pd.DataFrame, out_dir: Path) -> Path:
    labels = [POLICY_LABEL[p] for p in POLICY_ORDER]
    pdr = []
    energy = []
    for p in POLICY_ORDER:
        row = table[table["policy_key"] == p]
        if row.empty:
            pdr.append(np.nan)
            energy.append(np.nan)
        else:
            pdr.append(_safe_float(row.iloc[0]["pdr_mean"]))
            energy.append(_safe_float(row.iloc[0]["energy_mean"]))

    x = np.arange(len(POLICY_ORDER))
    colors = [POLICY_COLOR[p] for p in POLICY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # PDR plot
    ax = axes[0]
    bars = ax.bar(x, pdr, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("PDR (%)")
    ax.set_title("PDR Comparison")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for rect, v in zip(bars, pdr):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.6,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Energy plot
    ax = axes[1]
    bars = ax.bar(x, energy, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Avg energy per packet (mJ)")
    ax.set_title("Energy Comparison")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for rect, v in zip(bars, energy):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + max(0.4, 0.01 * v),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    title = f"TC{tc}"
    if scenario:
        title += f" — {scenario}"
    fig.suptitle(title)
    fig.tight_layout()

    out_file = out_dir / f"tc{tc}_pdr_energy.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NB-IoT sweep results per test case")
    parser.add_argument("--input", default="nbiot_sweep_results", help="Sweep results folder")
    parser.add_argument(
        "--datasets-dir", default="nbiot_datasets", help="Dataset folder used for energy fallback"
    )
    parser.add_argument(
        "--out", default="nbiot_sweep_results/plots_per_tc", help="Output folder for per-TC plots"
    )
    parser.add_argument("--tc", nargs="*", help="Optional list of TCs, e.g. --tc 01 08 31")
    args = parser.parse_args()

    input_dir = Path(args.input)
    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}")
        return

    summary_files = _find_summary_files(input_dir, args.tc)
    if not summary_files:
        print("No tc*_summary.csv files found to plot.")
        return

    all_rows = []
    written = []

    for summary_file in summary_files:
        table = _load_tc_summary(summary_file)
        tc = _fmt_tc(str(table.iloc[0]["tc"]))
        scenario = str(table.iloc[0]["scenario"]) if "scenario" in table.columns else ""

        # Fill missing energy from nbiot_datasets.
        if table["energy_mean"].isna().all():
            fallback = _energy_from_datasets(datasets_dir, tc)
            if all(not np.isfinite(v) for v in fallback.values()):
                fallback = _recover_energy_by_rerun(tc, table, datasets_dir)
            table["energy_mean"] = table["policy_key"].map(fallback)

        # Keep only expected policy order.
        table = table[table["policy_key"].isin(POLICY_ORDER)].copy()

        plot_file = _plot_tc(tc, scenario, table, out_dir)
        written.append(plot_file)

        table_out = table.copy()
        table_out["tc"] = tc
        all_rows.append(table_out)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_csv = out_dir / "per_tc_pdr_energy_summary.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"Saved -> {combined_csv}")

    for p in written:
        print(f"Saved -> {p}")

    print(f"Done. Generated {len(written)} per-test-case plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
