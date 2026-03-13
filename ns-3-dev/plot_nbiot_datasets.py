#!/usr/bin/env python3
"""
NB-IoT Dataset Plotter
======================
Plots simulation outputs from nbiot_datasets/.

Expected files:
  - no_la_<scenario>_dataset.csv
  - default_la_<scenario>_dataset.csv
  - ddqn_la_<scenario>_dataset.csv
  - ppo_la_<scenario>_dataset.csv
  - marl_la_<scenario>_dataset.csv
(Any *_environment.csv file is ignored.)

Usage:
  python3 plot_nbiot_datasets.py
  python3 plot_nbiot_datasets.py --input-dir nbiot_datasets --out nbiot_datasets/plots
  python3 plot_nbiot_datasets.py --scenario tc31_run005
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = ["no_la", "default_la", "ddqn_la", "ppo_la", "marl_la"]
POLICY_LABEL = {
    "no_la": "SDBP",
    "default_la": "3GPP",
    "ddqn_la": "DDQN",
    "ppo_la": "PPO",
    "marl_la": "MARL",
}
POLICY_COLOR = {
    "no_la": "#e74c3c",
    "default_la": "#e67e22",
    "ddqn_la": "#2980b9",
    "ppo_la": "#27ae60",
    "marl_la": "#8e44ad",
}


def parse_policy_and_scenario(path: Path) -> Optional[Dict[str, str]]:
    name = path.name
    if not name.endswith("_dataset.csv"):
        return None
    if name.endswith("_environment.csv"):
        return None

    for policy in POLICY_ORDER:
        prefix = f"{policy}_"
        suffix = "_dataset.csv"
        if name.startswith(prefix) and name.endswith(suffix):
            scenario = name[len(prefix) : -len(suffix)]
            return {"policy": policy, "scenario": scenario}
    return None


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    rows = len(df)
    if rows == 0:
        return {
            "n_rows": 0,
            "pdr_pct": np.nan,
            "avg_sinr_db": np.nan,
            "avg_energy_mj": np.nan,
            "avg_tx_power_dbm": np.nan,
            "avg_rach_attempts": np.nan,
            "collision_rate_pct": np.nan,
        }

    data_success = pd.to_numeric(df.get("data_success"), errors="coerce")
    sinr = pd.to_numeric(df.get("sinr_db"), errors="coerce")
    energy = pd.to_numeric(df.get("energy_consumed_mj"), errors="coerce")
    txp = pd.to_numeric(df.get("tx_power_dbm"), errors="coerce")
    rach = pd.to_numeric(df.get("rach_attempts"), errors="coerce")
    collision = pd.to_numeric(df.get("collision_flag"), errors="coerce")

    pdr = float(data_success.mean() * 100.0) if data_success.notna().any() else np.nan
    coll = float(collision.mean() * 100.0) if collision.notna().any() else np.nan

    return {
        "n_rows": int(rows),
        "pdr_pct": pdr,
        "avg_sinr_db": float(sinr.mean()) if sinr.notna().any() else np.nan,
        "avg_energy_mj": float(energy.mean()) if energy.notna().any() else np.nan,
        "avg_tx_power_dbm": float(txp.mean()) if txp.notna().any() else np.nan,
        "avg_rach_attempts": float(rach.mean()) if rach.notna().any() else np.nan,
        "collision_rate_pct": coll,
    }


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    scenarios = list(summary["scenario"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.14

    # PDR bars
    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.2), 4.8))
    for idx, policy in enumerate(POLICY_ORDER):
        sub = summary[summary["policy"] == policy].set_index("scenario")
        vals = [sub.loc[s, "pdr_pct"] if s in sub.index else np.nan for s in scenarios]
        offset = (idx - len(POLICY_ORDER) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width=width, color=POLICY_COLOR[policy], label=POLICY_LABEL[policy]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("PDR (%)")
    ax.set_title("PDR by Scenario and LA Policy")
    ax.legend(ncol=5, fontsize=8)
    save_fig(fig, out_dir, "01_pdr_by_scenario_policy.png")

    # Energy bars
    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.2), 4.8))
    for idx, policy in enumerate(POLICY_ORDER):
        sub = summary[summary["policy"] == policy].set_index("scenario")
        vals = [sub.loc[s, "avg_energy_mj"] if s in sub.index else np.nan for s in scenarios]
        offset = (idx - len(POLICY_ORDER) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width=width, color=POLICY_COLOR[policy], label=POLICY_LABEL[policy]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Avg energy per packet (mJ)")
    ax.set_title("Energy by Scenario and LA Policy")
    ax.legend(ncol=5, fontsize=8)
    save_fig(fig, out_dir, "02_energy_by_scenario_policy.png")


def plot_sinr_cdf(all_data: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy in POLICY_ORDER:
        sub = all_data[all_data["policy"] == policy]
        vals = pd.to_numeric(sub["sinr_db"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        vals = np.sort(vals)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=POLICY_LABEL[policy], color=POLICY_COLOR[policy], linewidth=1.8)

    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("SINR Distribution by LA Policy")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    save_fig(fig, out_dir, "03_sinr_cdf_by_policy.png")


def plot_energy_vs_success(all_data: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for policy in POLICY_ORDER:
        sub = all_data[all_data["policy"] == policy].copy()
        if sub.empty:
            continue
        x = pd.to_numeric(sub["energy_consumed_mj"], errors="coerce")
        y = pd.to_numeric(sub["data_success"], errors="coerce") * 100.0
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=9,
            alpha=0.35,
            label=POLICY_LABEL[policy],
            color=POLICY_COLOR[policy],
        )

    ax.set_xlabel("Energy consumed (mJ)")
    ax.set_ylabel("Packet success (%)")
    ax.set_title("Energy vs Success (per-packet samples)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(markerscale=2)
    save_fig(fig, out_dir, "04_energy_vs_success_scatter.png")


def load_dataset_rows(input_dir: Path, scenario_filter: Optional[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for csv_file in sorted(input_dir.glob("*.csv")):
        if csv_file.name.endswith("_environment.csv"):
            continue

        parsed = parse_policy_and_scenario(csv_file)
        if not parsed:
            continue

        scenario = parsed["scenario"]
        if scenario_filter and scenario != scenario_filter:
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            print(f"Skipping unreadable file {csv_file.name}: {exc}")
            continue

        df["policy"] = parsed["policy"]
        df["scenario"] = scenario
        df["source_file"] = csv_file.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_summary(all_data: pd.DataFrame) -> pd.DataFrame:
    if all_data.empty:
        return pd.DataFrame()

    out_rows = []
    grouped = all_data.groupby(["scenario", "policy"], sort=True)
    for (scenario, policy), g in grouped:
        metrics = compute_metrics(g)
        out_rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "policy_name": POLICY_LABEL.get(policy, policy),
                **metrics,
            }
        )

    df = pd.DataFrame(out_rows)
    df = df.sort_values(
        by=["scenario", "policy"],
        key=lambda c: (
            c.map({k: i for i, k in enumerate(POLICY_ORDER)}) if c.name == "policy" else c
        ),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="nbiot_datasets", help="Folder containing NB-IoT dataset CSV files"
    )
    parser.add_argument("--out", default="nbiot_datasets/plots", help="Output folder for plots")
    parser.add_argument(
        "--scenario", default=None, help="Optional scenario filter, e.g. tc31_run005"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input folder not found: {input_dir}")
        return

    all_data = load_dataset_rows(input_dir, args.scenario)
    if all_data.empty:
        print("No dataset CSV files found to plot.")
        return

    summary = build_summary(all_data)
    summary_csv = out_dir / "nbiot_summary_metrics.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved -> {summary_csv}")

    plot_summary_bars(summary, out_dir)
    plot_sinr_cdf(all_data, out_dir)
    plot_energy_vs_success(all_data, out_dir)

    print(f"Done. Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
