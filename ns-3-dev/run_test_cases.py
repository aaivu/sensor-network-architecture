#!/usr/bin/env python3
"""
LoRaWAN ADR Test Case Automation
=================================
Runs each test case N times (default 100), collects per-run metrics from the
lorawan_datasets folder using the existing analysis logic, clears the folder
between runs, and writes a final per-test-case summary CSV + console report.

Usage:
    # Run all test cases, 100 repetitions each (default)
    python3 run_test_cases.py

    # Run only specific test cases by ID (1-indexed), 20 repetitions each
    python3 run_test_cases.py --tc 1,5,13 --runs 20

    # Resume from a previous partial run (skips TCs already in results file)
    python3 run_test_cases.py --resume

    # Dry-run: print commands without executing
    python3 run_test_cases.py --dry-run
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NS3_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = NS3_DIR / "lorawan_datasets"
RESULTS_DIR = NS3_DIR / "test_case_results"
NS3_BIN = NS3_DIR / "ns3"

# Try python3.13 first (avoids Python 3.14 argparse regression), else python3
PYTHON_BIN = shutil.which("python3.13") or shutil.which("python3") or "python3"

# ---------------------------------------------------------------------------
# ADR column helpers (mirrors analysis.py)
# ---------------------------------------------------------------------------
SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
PAYLOAD_SIZE_BYTES = 20
PREAMBLE_SYMBOLS = 8


def _toa_ms(sf: int, bw_khz: int = 125, cr: int = 1,
            payload_bytes: int = PAYLOAD_SIZE_BYTES) -> float:
    t_sym = (2 ** sf) / (bw_khz * 1000) * 1000
    t_preamble = (PREAMBLE_SYMBOLS + 4.25) * t_sym
    de = 1 if sf >= 11 else 0
    n = 8 * payload_bytes - 4 * sf + 28 + 16 - 0
    d = 4 * (sf - 2 * de) or (4 * sf)
    n_payload = 8 + max(np.ceil(n / d) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    tc_id: int
    label: str
    scenario: str
    expected_rl: str           # "best" | "good" | "neutral" | "fail"
    nDevices: int
    radius: int
    wifiInterferers: int
    lteInterferers: int
    appPeriod: int
    simulationTime: int
    environmental: bool = True

    def ns3_args(self) -> str:
        env = "true" if self.environmental else "false"
        return (
            f"advanced-ddqn-per-adr-example "
            f"--nDevices={self.nDevices} "
            f"--radius={self.radius} "
            f"--wifiInterferers={self.wifiInterferers} "
            f"--lteInterferers={self.lteInterferers} "
            f"--appPeriod={self.appPeriod} "
            f"--adr=all "
            f"--simulationTime={self.simulationTime} "
            f"--environmental={env}"
        )


TEST_CASES: List[TestCase] = [
    # ── GROUP 1: Baseline (no interference) ──────────────────────────────────
    TestCase(1,  "Tiny room, minimal devices",        "Single open room",                    "fail",    5,  30,  0,  0,  60,  5000),
    TestCase(2,  "Medium building, no interference",  "100m, obstacles only",                "neutral", 20, 100, 0,  0,  60,  5000),
    TestCase(3,  "Large building, no interference",   "500m, obstacles only",                "neutral", 20, 500, 0,  0,  60,  5000),

    # ── GROUP 2: Building size progression (WiFi fixed) ──────────────────────
    TestCase(4,  "Small office (50m)",                "50m, 3 APs",                          "fail",    15, 50,  3,  0,  60,  5000),
    TestCase(5,  "Medium office (100m)",              "100m, 8 APs",                         "good",    30, 100, 8,  0,  60,  5000),
    TestCase(6,  "Hospital wing (200m)",              "200m, 15 APs",                        "good",    50, 200, 15, 0,  30,  5000),
    TestCase(7,  "Factory floor (300m)",              "300m, 18 APs",                        "good",    60, 300, 18, 0,  30,  5000),
    TestCase(8,  "Campus / terminal (500m)",          "500m, 20 APs",                        "best",    80, 500, 20, 0,  60,  7000),

    # ── GROUP 3: WiFi scaling — radius 100m ──────────────────────────────────
    TestCase(9,  "No WiFi (100m)",                    "100m, 0 APs",                         "neutral", 30, 100, 0,  0,  60,  5000),
    TestCase(10, "Light WiFi (3 APs, 100m)",          "100m, 3 APs",                         "fail",    30, 100, 3,  0,  60,  5000),
    TestCase(11, "Moderate WiFi (10 APs, 100m)",      "100m, 10 APs",                        "good",    30, 100, 10, 0,  60,  5000),
    TestCase(12, "Heavy WiFi (20 APs, 100m)",         "100m, 20 APs",                        "best",    30, 100, 20, 0,  60,  5000),
    TestCase(13, "Extreme WiFi (30 APs, 100m)",       "100m, 30 APs — shopping mall/airport","best",    30, 100, 30, 0,  60,  5000),

    # ── GROUP 4: WiFi scaling — radius 500m ──────────────────────────────────
    TestCase(14, "Light WiFi, large building (500m)", "500m, 5 APs",                         "neutral", 50, 500, 5,  0,  60,  7000),
    TestCase(15, "Heavy WiFi, large building (500m)", "500m, 20 APs",                        "best",    50, 500, 20, 0,  60,  7000),

    # ── GROUP 5: Device density scaling — 100m, moderate WiFi ────────────────
    TestCase(16, "Very sparse (5 devices, 100m)",     "100m, 8 APs, 5 devices",              "fail",    5,  100, 8,  0,  60,  5000),
    TestCase(17, "Low density (20 devices, 100m)",    "100m, 8 APs, 20 devices",             "good",    20, 100, 8,  0,  60,  5000),
    TestCase(18, "Medium density (50 devices, 100m)", "100m, 8 APs, 50 devices",             "best",    50, 100, 8,  0,  60,  5000),
    TestCase(19, "High density (100 devices, 100m)",  "100m, 8 APs, 100 devices — tracking", "best",    100,100, 8,  0,  60,  5000),

    # ── GROUP 6: Traffic rate scaling — 100m, moderate WiFi ──────────────────
    TestCase(20, "Very frequent (10s interval)",      "100m, 8 APs, 30 dev, 10s period",     "good",    30, 100, 8,  0,  10,  5000),
    TestCase(21, "Normal IoT rate (60s interval)",    "100m, 8 APs, 30 dev, 60s period",     "good",    30, 100, 8,  0,  60,  5000),
    TestCase(22, "Slow sensor (300s interval)",       "100m, 8 APs, 30 dev, 300s period",    "fail",    30, 100, 8,  0,  300, 15000),
    TestCase(23, "Very slow (600s interval)",         "100m, 8 APs, 30 dev, 600s period",    "fail",    30, 100, 8,  0,  600, 30000),

    # ── GROUP 7: LTE bleed-through ────────────────────────────────────────────
    TestCase(24, "Deep indoor — no LTE",              "100m, 10 APs, 0 LTE",                 "good",    30, 100, 10, 0,  60,  5000),
    TestCase(25, "Ground floor retail (weak LTE)",    "100m, 10 APs, 2 LTE",                 "good",    30, 100, 10, 2,  60,  5000),
    TestCase(26, "Open-plan ground floor (mod LTE)",  "100m, 10 APs, 5 LTE",                 "best",    30, 100, 10, 5,  60,  5000),
    TestCase(27, "Lobby / entrance (heavy LTE)",      "100m, 15 APs, 8 LTE",                 "best",    30, 100, 15, 8,  60,  5000),

    # ── GROUP 8: RL convergence / learning behaviour ──────────────────────────
    TestCase(28, "Short run — RL undertrained",       "100m, 10 APs, 500s",                  "fail",    30, 100, 10, 0,  60,  500),
    TestCase(29, "Medium run — partial convergence",  "100m, 10 APs, 5000s",                 "good",    30, 100, 10, 0,  60,  5000),
    TestCase(30, "Long run — full convergence",       "100m, 10 APs, 50000s",                "best",    30, 100, 10, 0,  60,  50000),

    # ── GROUP 9: Stress / worst cases ─────────────────────────────────────────
    TestCase(31, "Worst-case stress (100m)",          "100m, 25 APs, 5 LTE, 100 dev, 15s",  "best",    100,100, 25, 5,  15,  10000),
    TestCase(32, "Worst-case stress (500m)",          "500m, 25 APs, 5 LTE, 100 dev, 15s",  "best",    100,500, 25, 5,  15,  10000),
    TestCase(33, "Edge devices only (500m)",          "500m, 10 APs, 10 devices",            "good",    10, 500, 10, 0,  60,  5000),
    TestCase(34, "Gateway overload (150 dev)",        "200m, 15 APs, 150 dev — noisy reward","fail",    150,200, 15, 0,  30,  10000),
]

# ---------------------------------------------------------------------------
# Per-run metric extraction from lorawan_datasets/
# ---------------------------------------------------------------------------
ADR_PATTERNS = {
    "no_adr":       "no_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "classical_adr":"adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "ddqn_adr":     "ddqn_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "ppo_adr":      "ppo_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "marl_adr":     "marl_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
}


def _extract_metrics_from_datasets(datasets_dir: Path) -> Dict[str, dict]:
    """
    Read all per-device CSVs from datasets_dir and return per-ADR-type metrics dict.
    Returns {} if no files found.
    """
    results: Dict[str, dict] = {}

    for adr_key, pattern in ADR_PATTERNS.items():
        files = glob.glob(str(datasets_dir / pattern))
        if not files:
            continue

        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
                if df.empty:
                    continue
                dfs.append(df)
            except Exception as e:
                log.warning(f"Could not read {fp}: {e}")

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)

        # Basic columns expected by analysis.py
        has_snr   = "snr_db" in combined.columns
        has_rssi  = "pre_tx_rssi_dBm" in combined.columns
        has_sf    = "sf" in combined.columns
        has_tp    = "tx_power_dBm" in combined.columns
        has_nrg   = "energy_consumed_mJ" in combined.columns

        total     = len(combined)
        n_success = int((combined["snr_db"] != -999.0).sum()) if has_snr else 0
        pdr       = n_success / total * 100.0 if total > 0 else 0.0

        snr_vals  = combined.loc[combined["snr_db"] != -999.0, "snr_db"] if has_snr else pd.Series(dtype=float)

        # Time-on-air per packet (ms) then mean
        toa_mean = float("nan")
        if has_sf:
            try:
                toa_series = combined["sf"].apply(lambda s: _toa_ms(int(s)))
                toa_mean = toa_series.mean()
            except Exception:
                pass

        # SNR margin = snr - threshold
        snr_margin_median = float("nan")
        if has_snr and len(snr_vals) > 0:
            margins = snr_vals.apply(lambda s: s - SNR_THRESHOLDS.get(
                int(combined.loc[combined["snr_db"] == s, "sf"].iloc[0])
                if has_sf else 7, -7.5))
            snr_margin_median = float(margins.median())

        # Energy per successful packet
        eps = float("nan")
        if has_nrg and n_success > 0:
            total_nrg = float(combined["energy_consumed_mJ"].sum())
            eps = total_nrg / n_success

        results[adr_key] = {
            "total_packets":    total,
            "n_success":        n_success,
            "pdr_pct":          round(pdr, 4),
            "avg_snr_db":       round(float(snr_vals.mean()), 4) if len(snr_vals) > 0 else float("nan"),
            "std_snr_db":       round(float(snr_vals.std()),  4) if len(snr_vals) > 0 else float("nan"),
            "avg_sf":           round(float(combined["sf"].mean()), 4) if has_sf else float("nan"),
            "std_sf":           round(float(combined["sf"].std()),  4) if has_sf else float("nan"),
            "avg_tx_power_dbm": round(float(combined["tx_power_dBm"].mean()), 4) if has_tp else float("nan"),
            "avg_rssi_dbm":     round(float(combined["pre_tx_rssi_dBm"].mean()), 4) if has_rssi else float("nan"),
            "total_energy_mJ":  round(float(combined["energy_consumed_mJ"].sum()), 4) if has_nrg else float("nan"),
            "energy_per_success_mJ": round(eps, 4) if not np.isnan(eps) else float("nan"),
            "avg_toa_ms":       round(toa_mean, 4) if not np.isnan(toa_mean) else float("nan"),
            "snr_margin_median_db": round(snr_margin_median, 4) if not np.isnan(snr_margin_median) else float("nan"),
            "n_devices":        len(files),
        }

    return results


# ---------------------------------------------------------------------------
# Dataset folder helpers
# ---------------------------------------------------------------------------

def _datasets_empty(datasets_dir: Path) -> bool:
    """True if the folder does not exist or has no CSV files."""
    if not datasets_dir.exists():
        return True
    csvs = list(datasets_dir.glob("*.csv"))
    return len(csvs) == 0


def _clear_datasets(datasets_dir: Path):
    """Remove lorawan_datasets folder completely."""
    if datasets_dir.exists():
        shutil.rmtree(datasets_dir)
        log.debug(f"Cleared {datasets_dir}")


def _wait_for_empty(datasets_dir: Path, timeout: int = 30):
    """Poll until datasets folder is confirmed empty/gone (handles NFS/slow FS)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _datasets_empty(datasets_dir):
            return
        time.sleep(0.5)
    log.warning("Datasets folder not empty after clearing — proceeding anyway.")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_simulation(tc: TestCase, run_idx: int, dry_run: bool = False,
                    timeout_sec: int = 7200) -> bool:
    """
    Run one simulation. Returns True on success (exit code 0).
    Note: the simulation itself cleans lorawan_datasets at startup (see C++ main).
    We double-check before launching.
    """
    # Pre-flight: ensure datasets folder is empty before we start
    if not _datasets_empty(DATASETS_DIR):
        log.warning(f"  TC{tc.tc_id} run {run_idx}: datasets folder not empty before run — clearing now.")
        _clear_datasets(DATASETS_DIR)
        _wait_for_empty(DATASETS_DIR)

    cmd = [PYTHON_BIN, str(NS3_BIN), "run", tc.ns3_args()]
    log.info(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        log.info("  [DRY-RUN] skipping execution")
        return True

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(NS3_DIR),
            capture_output=False,
            timeout=timeout_sec,
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            log.error(f"  Simulation exited with code {proc.returncode} after {elapsed:.0f}s")
            return False
        log.info(f"  Simulation completed in {elapsed:.0f}s")
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  Simulation timed out after {timeout_sec}s")
        return False
    except Exception as e:
        log.error(f"  Simulation error: {e}")
        return False


# ---------------------------------------------------------------------------
# Summary statistics across N runs
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "pdr_pct", "avg_snr_db", "std_snr_db", "avg_sf", "std_sf",
    "avg_tx_power_dbm", "avg_rssi_dbm", "total_energy_mJ",
    "energy_per_success_mJ", "avg_toa_ms", "snr_margin_median_db",
]

ADR_DISPLAY = {
    "no_adr":       "No ADR",
    "classical_adr":"Classical ADR",
    "ddqn_adr":     "DDQN-PER ADR",
    "ppo_adr":      "PPO ADR",
    "marl_adr":     "MARL ADR",
}


def _summarise_runs(run_records: List[dict]) -> pd.DataFrame:
    """
    Given a list of per-run dicts (each keyed by adr_key → metric_dict),
    compute mean/std/min/max/median/p5/p95/ci95_low/ci95_high for every metric.
    Returns a DataFrame with one row per (adr_key, metric).
    """
    rows = []
    # Transpose: adr_key → list of metric values
    per_adr: Dict[str, Dict[str, List]] = {k: {m: [] for m in METRIC_COLS} for k in ADR_PATTERNS}

    for run in run_records:
        for adr_key in ADR_PATTERNS:
            if adr_key not in run:
                continue
            for metric in METRIC_COLS:
                val = run[adr_key].get(metric, float("nan"))
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    per_adr[adr_key][metric].append(float(val))

    for adr_key in ADR_PATTERNS:
        adr_name = ADR_DISPLAY.get(adr_key, adr_key)
        for metric in METRIC_COLS:
            vals = np.array(per_adr[adr_key][metric])
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se   = std / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "adr_key":   adr_key,
                "adr_name":  adr_name,
                "metric":    metric,
                "n_runs":    n,
                "mean":      round(mean, 4),
                "std":       round(std, 4),
                "min":       round(float(np.min(vals)), 4),
                "p5":        round(float(np.percentile(vals, 5)), 4),
                "median":    round(float(np.median(vals)), 4),
                "p95":       round(float(np.percentile(vals, 95)), 4),
                "max":       round(float(np.max(vals)), 4),
                "ci95_low":  round(mean - 1.96 * se, 4),
                "ci95_high": round(mean + 1.96 * se, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print_tc_summary(tc: TestCase, summary_df: pd.DataFrame,
                      n_successful: int, n_total: int, failed_runs: List[int]):
    bar = "=" * 80
    print(f"\n{bar}")
    print(f" TC-{tc.tc_id:02d}  {tc.label}")
    print(f" Scenario : {tc.scenario}")
    print(f" Params   : {tc.nDevices} devices | {tc.radius}m radius | "
          f"WiFi={tc.wifiInterferers} | LTE={tc.lteInterferers} | "
          f"period={tc.appPeriod}s | sim={tc.simulationTime}s")
    print(f" RL expected to work: {tc.expected_rl.upper()}")
    print(f" Successful runs: {n_successful}/{n_total}")
    if failed_runs:
        print(f" Failed run indices: {failed_runs}")
    print(bar)

    if summary_df.empty:
        print(" No data to display.")
        return

    pdr_rows = summary_df[summary_df["metric"] == "pdr_pct"].copy()
    if pdr_rows.empty:
        return

    # Header
    print(f"  {'ADR Method':<20} {'Mean PDR%':>10} {'Std':>7} "
          f"{'Min':>7} {'Median':>8} {'Max':>7} "
          f"{'Energy/succ(mJ)':>17} {'Avg SF':>8} {'CI95 low':>10} {'CI95 high':>10}")
    print("  " + "-" * 108)

    for _, row in pdr_rows.sort_values("adr_key").iterrows():
        adr = row["adr_key"]
        nrg_row = summary_df[(summary_df["adr_key"] == adr) &
                             (summary_df["metric"] == "energy_per_success_mJ")]
        sf_row  = summary_df[(summary_df["adr_key"] == adr) &
                             (summary_df["metric"] == "avg_sf")]

        nrg = f"{nrg_row['mean'].values[0]:>11.4f}" if not nrg_row.empty else "         N/A"
        sf  = f"{sf_row['mean'].values[0]:>8.2f}"   if not sf_row.empty  else "     N/A"

        print(f"  {row['adr_name']:<20} "
              f"{row['mean']:>10.2f} "
              f"{row['std']:>7.2f} "
              f"{row['min']:>7.2f} "
              f"{row['median']:>8.2f} "
              f"{row['max']:>7.2f} "
              f"{nrg:>17} "
              f"{sf:>8} "
              f"{row['ci95_low']:>10.2f} "
              f"{row['ci95_high']:>10.2f}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRaWAN ADR test case automation")
    p.add_argument("--tc",    type=str,  default=None,
                   help="Comma-separated TC IDs to run (e.g. 1,5,13). Default: all.")
    p.add_argument("--runs",  type=int,  default=100,
                   help="Number of repetitions per test case (default: 100).")
    p.add_argument("--out",   type=str,  default=str(RESULTS_DIR),
                   help="Output directory for results.")
    p.add_argument("--resume", action="store_true",
                   help="Skip TCs that already have a results file in --out.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands but do not execute simulations.")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-simulation timeout in seconds (default: 7200).")
    return p.parse_args()


def main():
    args = parse_cli()

    results_dir = Path(args.out)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter TCs
    selected_ids = None
    if args.tc:
        selected_ids = {int(x.strip()) for x in args.tc.split(",")}
    tcs = [tc for tc in TEST_CASES if selected_ids is None or tc.tc_id in selected_ids]

    log.info(f"Test cases to run: {[tc.tc_id for tc in tcs]}")
    log.info(f"Repetitions per TC: {args.runs}")
    log.info(f"Python binary: {PYTHON_BIN}")
    log.info(f"ns-3 dir: {NS3_DIR}")
    log.info(f"Results dir: {results_dir}")

    grand_summary_rows = []

    for tc in tcs:
        tc_raw_path    = results_dir / f"tc{tc.tc_id:02d}_raw_runs.jsonl"
        tc_summary_path = results_dir / f"tc{tc.tc_id:02d}_summary.csv"

        # Resume: skip if already done
        if args.resume and tc_summary_path.exists():
            log.info(f"TC-{tc.tc_id:02d}: resume — summary already exists, skipping.")
            # Still load for grand summary
            if tc_summary_path.exists():
                df = pd.read_csv(tc_summary_path)
                pdr_rows = df[df["metric"] == "pdr_pct"]
                for _, row in pdr_rows.iterrows():
                    grand_summary_rows.append({
                        "tc_id": tc.tc_id, "label": tc.label,
                        "scenario": tc.scenario, "expected_rl": tc.expected_rl,
                        "nDevices": tc.nDevices, "radius": tc.radius,
                        "wifiInterferers": tc.wifiInterferers, "lteInterferers": tc.lteInterferers,
                        "appPeriod": tc.appPeriod, "simulationTime": tc.simulationTime,
                        **row.to_dict()
                    })
            continue

        log.info(f"\n{'█'*60}")
        log.info(f"Starting TC-{tc.tc_id:02d}: {tc.label} ({args.runs} runs)")

        run_records: List[dict] = []
        failed_runs: List[int] = []

        # Load any previously saved raw runs (partial resume within a TC)
        existing_runs = 0
        if tc_raw_path.exists():
            with open(tc_raw_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        run_records.append(json.loads(line))
            existing_runs = len(run_records)
            log.info(f"  Loaded {existing_runs} existing runs from {tc_raw_path}")

        for run_idx in range(existing_runs + 1, args.runs + 1):
            log.info(f"  — Run {run_idx}/{args.runs}")

            # 1. Make sure datasets folder is empty before launch
            if not _datasets_empty(DATASETS_DIR):
                log.warning(f"    Leftover files in {DATASETS_DIR} — clearing.")
                _clear_datasets(DATASETS_DIR)
                _wait_for_empty(DATASETS_DIR)

            # 2. Run the simulation
            success = _run_simulation(tc, run_idx, dry_run=args.dry_run,
                                      timeout_sec=args.timeout)

            if not success:
                failed_runs.append(run_idx)
                log.warning(f"    Run {run_idx} failed — skipping metric extraction.")
                # Clear any partial output
                _clear_datasets(DATASETS_DIR)
                continue

            # 3. Extract metrics
            if args.dry_run:
                # Dummy data for dry-run testing
                metrics = {k: {"pdr_pct": 85.0 + run_idx * 0.01,
                               "avg_snr_db": 10.0, "std_snr_db": 2.0,
                               "avg_sf": 8.0, "std_sf": 0.5,
                               "avg_tx_power_dbm": 14.0, "avg_rssi_dbm": -90.0,
                               "total_energy_mJ": 5.0, "energy_per_success_mJ": 0.06,
                               "avg_toa_ms": 200.0, "snr_margin_median_db": 3.5,
                               "n_devices": tc.nDevices}
                           for k in ADR_PATTERNS}
            else:
                metrics = _extract_metrics_from_datasets(DATASETS_DIR)

            if not metrics:
                log.warning(f"    No metrics extracted for run {run_idx}.")
                failed_runs.append(run_idx)
            else:
                log.info(f"    Metrics extracted for ADR types: {list(metrics.keys())}")
                run_record = {"run": run_idx, "tc_id": tc.tc_id, **metrics}
                run_records.append(run_record)

                # Append raw record to JSONL file immediately (crash-safe)
                with open(tc_raw_path, "a") as f:
                    f.write(json.dumps(run_record) + "\n")

            # 4. Clear datasets after extraction
            _clear_datasets(DATASETS_DIR)
            _wait_for_empty(DATASETS_DIR)

        # 5. Summarise all runs for this TC
        n_successful = len(run_records)
        summary_df = _summarise_runs(run_records)

        # Save summary CSV
        summary_df.insert(0, "tc_id",    tc.tc_id)
        summary_df.insert(1, "label",    tc.label)
        summary_df.insert(2, "scenario", tc.scenario)
        summary_df.insert(3, "expected_rl", tc.expected_rl)
        summary_df.insert(4, "nDevices", tc.nDevices)
        summary_df.insert(5, "radius",   tc.radius)
        summary_df.insert(6, "wifiInterferers", tc.wifiInterferers)
        summary_df.insert(7, "lteInterferers",  tc.lteInterferers)
        summary_df.insert(8, "appPeriod",       tc.appPeriod)
        summary_df.insert(9, "simulationTime",  tc.simulationTime)
        summary_df.to_csv(tc_summary_path, index=False)
        log.info(f"  Summary saved → {tc_summary_path}")

        # Print to console
        _print_tc_summary(tc, summary_df, n_successful, args.runs, failed_runs)

        # Accumulate for grand summary
        pdr_rows = summary_df[summary_df["metric"] == "pdr_pct"]
        for _, row in pdr_rows.iterrows():
            grand_summary_rows.append(row.to_dict())

    # ── Grand summary across all TCs ──────────────────────────────────────────
    if grand_summary_rows:
        grand_df = pd.DataFrame(grand_summary_rows)
        grand_path = results_dir / "grand_summary_pdr.csv"
        grand_df.to_csv(grand_path, index=False)
        log.info(f"\nGrand summary (PDR) saved → {grand_path}")

        print("\n" + "=" * 80)
        print(" GRAND SUMMARY — Mean PDR% across all test cases")
        print("=" * 80)
        print(f"  {'TC':>4}  {'Label':<38} {'RL?':<8} "
              f"{'No ADR':>8} {'Cls ADR':>8} {'DDQN':>8} {'PPO':>8} {'MARL':>8}")
        print("  " + "-" * 86)

        for tc in tcs:
            tc_rows = grand_df[grand_df["tc_id"] == tc.tc_id]
            vals = {}
            for k in ADR_PATTERNS:
                r = tc_rows[tc_rows["adr_key"] == k]
                vals[k] = f"{r['mean'].values[0]:>8.2f}" if not r.empty else "     N/A"
            print(f"  {tc.tc_id:>4}  {tc.label:<38} {tc.expected_rl:<8} "
                  f"{vals.get('no_adr','     N/A')} "
                  f"{vals.get('classical_adr','     N/A')} "
                  f"{vals.get('ddqn_adr','     N/A')} "
                  f"{vals.get('ppo_adr','     N/A')} "
                  f"{vals.get('marl_adr','     N/A')}")

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
