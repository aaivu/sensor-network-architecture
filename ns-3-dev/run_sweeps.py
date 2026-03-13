#!/usr/bin/env python3
"""
NB-IoT LA Scenario Runner
=========================
Runs 34 fixed NB-IoT environments and extracts PDR for:
  - no_la
  - 3gpp
  - marl
  - ppo
  - ddqn

Outputs under nbiot_sweep_results/:
  - per-scenario JSONL: tcXX_raw.jsonl
  - per-scenario summary CSV: tcXX_summary.csv
  - scenario index CSV: scenario_index.csv
  - global long summary CSV: full_summary_long.csv
  - global pivot summary CSV: full_summary_pdr.csv

Usage:
    cd ns-3-dev
    source venv/bin/activate
    python3 run_sweeps.py
    python3 run_sweeps.py --resume
    python3 run_sweeps.py --dry-run
    python3 run_sweeps.py --runs 5
    python3 run_sweeps.py --tc 01 08 31
    python3 run_sweeps.py --program nbiot-la-example
"""

import argparse
import glob
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
NS3_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = NS3_DIR / "nbiot_sweep_results"
DATASETS_DIR = NS3_DIR / "nbiot_datasets"
FALLBACK_DATASETS_DIR = NS3_DIR / "lorawan_datasets"
NS3_BIN = NS3_DIR / "ns3"
PYTHON_BIN = shutil.which("python3.13") or shutil.which("python3") or "python3"

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_RUNS = 100
DEFAULT_TIMEOUT = 900
DEFAULT_PROGRAM = "advanced-ddqn-per-adr-nbiot"

LA_KEYS = ["no_la", "3gpp", "marl", "ppo", "ddqn"]
LA_DISPLAY = {
    "no_la": "Static distance-based profile",
    "3gpp": "3GPP",
    "marl": "MARL",
    "ppo": "PPO",
    "ddqn": "DDQN",
}

# Multiple patterns per policy to be robust with naming conventions.
LA_PATTERNS = {
    "no_la": [
        "no_la_*.csv",
        "no_adr_*.csv",
    ],
    "3gpp": [
        "default_la_*.csv",
        "3gpp_*.csv",
        "classical_*.csv",
        "adr_*.csv",
    ],
    "marl": [
        "marl_la_*.csv",
        "marl_*.csv",
    ],
    "ppo": [
        "ppo_la_*.csv",
        "ppo_*.csv",
    ],
    "ddqn": [
        "ddqn_la_*.csv",
        "ddqn_*.csv",
    ],
}


@dataclass(frozen=True)
class Scenario:
    tc: str
    devices: int
    radius_m: int
    wifi: int
    lte: int
    period_s: int
    sim_time_s: int
    rl_works: str
    scenario: str

    @property
    def key(self) -> str:
        return f"tc{self.tc}"

    @property
    def raw_jsonl(self) -> Path:
        return RESULTS_DIR / f"{self.key}_raw.jsonl"

    @property
    def summary_csv(self) -> Path:
        return RESULTS_DIR / f"{self.key}_summary.csv"


def build_scenarios() -> List[Scenario]:
    table = [
        ("01", 5, 30, 0, 0, 60, 5000, "Too simple", "Single room baseline"),
        ("02", 20, 100, 0, 0, 60, 5000, "Neutral", "100m no interference"),
        ("03", 20, 500, 0, 0, 60, 5000, "Neutral", "500m no interference"),
        ("04", 15, 50, 3, 0, 60, 5000, "Classical sufficient", "Small office"),
        ("05", 30, 100, 8, 0, 60, 5000, "Starting", "Medium office"),
        ("06", 50, 200, 15, 0, 30, 5000, "Good", "Hospital wing"),
        ("07", 60, 300, 18, 0, 30, 5000, "Good", "Factory floor"),
        ("08", 80, 500, 20, 0, 60, 7000, "Strong", "Campus/terminal"),
        ("09", 30, 100, 0, 0, 60, 5000, "Equal", "No WiFi control"),
        ("10", 30, 100, 3, 0, 60, 5000, "Marginal", "Light WiFi"),
        ("11", 30, 100, 10, 0, 60, 5000, "Starting", "Moderate WiFi"),
        ("12", 30, 100, 20, 0, 60, 5000, "Strong", "Heavy WiFi"),
        ("13", 30, 100, 30, 0, 60, 5000, "Best", "Extreme WiFi"),
        ("14", 50, 500, 5, 0, 60, 7000, "Neutral", "Large + light WiFi"),
        ("15", 50, 500, 20, 0, 60, 7000, "Strong", "Large + heavy WiFi"),
        ("16", 5, 100, 8, 0, 60, 5000, "Undertrained", "Very sparse devices"),
        ("17", 20, 100, 8, 0, 60, 5000, "Starting", "Low density"),
        ("18", 50, 100, 8, 0, 60, 5000, "Strong", "Medium density"),
        ("19", 100, 100, 8, 0, 60, 5000, "Best", "High density tracking"),
        ("20", 30, 100, 8, 0, 10, 5000, "Strong", "Alarm/event burst"),
        ("21", 30, 100, 8, 0, 60, 5000, "Good", "Normal IoT reference"),
        ("22", 30, 100, 8, 0, 300, 15000, "Sparse rewards", "Slow sensor rate"),
        ("23", 30, 100, 8, 0, 600, 30000, "Very sparse", "Battery-optimized"),
        ("24", 30, 100, 10, 0, 60, 5000, "Good", "Deep indoor, no LTE"),
        ("25", 30, 100, 10, 2, 60, 5000, "Good", "Ground floor retail"),
        ("26", 30, 100, 10, 5, 60, 5000, "Strong", "Dynamic LTE"),
        ("27", 30, 100, 15, 8, 60, 5000, "Best", "Lobby/entrance"),
        ("28", 30, 100, 10, 0, 60, 500, "Not converged", "Short run"),
        ("29", 30, 100, 10, 0, 60, 5000, "Partial", "Medium run"),
        ("30", 30, 100, 10, 0, 60, 50000, "Best", "Full convergence"),
        ("31", 100, 100, 25, 5, 15, 10000, "Best", "Worst-case 100m"),
        ("32", 100, 500, 25, 5, 15, 10000, "Best", "Worst-case 500m"),
        ("33", 10, 500, 10, 0, 60, 5000, "Good", "Edge devices only"),
        ("34", 150, 200, 15, 0, 30, 10000, "Noisy reward", "Gateway overload"),
    ]
    return [
        Scenario(
            tc=tc,
            devices=devices,
            radius_m=radius,
            wifi=wifi,
            lte=lte,
            period_s=period,
            sim_time_s=sim_time,
            rl_works=rl_works,
            scenario=scenario,
        )
        for tc, devices, radius, wifi, lte, period, sim_time, rl_works, scenario in table
    ]


def _dataset_roots() -> List[Path]:
    roots = []
    if DATASETS_DIR.exists():
        roots.append(DATASETS_DIR)
    if FALLBACK_DATASETS_DIR.exists() and FALLBACK_DATASETS_DIR not in roots:
        roots.append(FALLBACK_DATASETS_DIR)
    return roots


def _clear_datasets() -> None:
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)
    if FALLBACK_DATASETS_DIR.exists():
        shutil.rmtree(FALLBACK_DATASETS_DIR)


def _as_numeric_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0})
    if mapped.notna().any():
        return mapped
    return pd.to_numeric(series, errors="coerce")


def compute_pdr_pct(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None

    # Direct PDR if available
    for col in ["pdr_pct", "pdr", "pdr_percent"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            value = float(s.mean())
            if col != "pdr_pct" and value <= 1.0:
                value *= 100.0
            return max(0.0, min(100.0, value))

    # Success indicator columns
    for col in [
        "success",
        "is_success",
        "packet_success",
        "data_success",
        "rx_success",
        "delivered",
        "received",
    ]:
        if col in df.columns:
            s = _as_numeric_bool_series(df[col]).dropna()
            if s.empty:
                continue
            value = float(s.mean())
            if value <= 1.0:
                value *= 100.0
            return max(0.0, min(100.0, value))

    # SNR sentinel style from prior datasets
    if "snr_db" in df.columns:
        snr = pd.to_numeric(df["snr_db"], errors="coerce")
        valid = snr.dropna()
        if not valid.empty:
            succ = (valid != -999.0).sum()
            return float(succ / len(valid) * 100.0)

    # Aggregate counters
    if "packets_sent" in df.columns and "packets_received" in df.columns:
        sent = pd.to_numeric(df["packets_sent"], errors="coerce").sum()
        recv = pd.to_numeric(df["packets_received"], errors="coerce").sum()
        if sent > 0:
            return float(recv / sent * 100.0)

    return None


def extract_pdr_by_policy() -> Dict[str, float]:
    out: Dict[str, float] = {}
    roots = _dataset_roots()
    if not roots:
        return out

    for la_key in LA_KEYS:
        files: List[str] = []
        for root in roots:
            for pattern in LA_PATTERNS[la_key]:
                files.extend(glob.glob(str(root / pattern)))
        if not files:
            continue

        pdr_values: List[float] = []
        for file_path in sorted(set(files)):
            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue
            pdr = compute_pdr_pct(df)
            if pdr is not None and np.isfinite(pdr):
                pdr_values.append(float(pdr))

        if pdr_values:
            out[la_key] = round(float(np.mean(pdr_values)), 4)

    return out


def build_ns3_cmd(s: Scenario, program: str, run_id: int) -> List[str]:
    csv_out = f"{s.key}_run{run_id:03d}_dataset.csv"
    total_interferers = s.wifi + s.lte
    return [
        PYTHON_BIN,
        str(NS3_BIN),
        "run",
        (
            f"{program} "
            f"--nUEs={s.devices} "
            f"--cellRadius={s.radius_m} "
            f"--lteInterferers={total_interferers} "
            f"--appPeriod={s.period_s} "
            f"--simulationTime={s.sim_time_s} "
            f"--adr=all "
            f"--csvFile={csv_out}"
        ),
    ]


def completed_runs(s: Scenario) -> int:
    if not s.raw_jsonl.exists():
        return 0
    count = 0
    with open(s.raw_jsonl) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_failed_summary(s: Scenario) -> None:
    rows = []
    for la in LA_KEYS:
        rows.append(
            {
                "tc": s.tc,
                "devices": s.devices,
                "radius_m": s.radius_m,
                "wifi": s.wifi,
                "lte": s.lte,
                "period_s": s.period_s,
                "sim_time_s": s.sim_time_s,
                "rl_works": s.rl_works,
                "scenario": s.scenario,
                "policy_key": la,
                "policy_name": LA_DISPLAY[la],
                "n_runs": 0,
                "pdr_mean": np.nan,
                "pdr_std": np.nan,
                "pdr_min": np.nan,
                "pdr_median": np.nan,
                "pdr_max": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "failed": True,
            }
        )
    pd.DataFrame(rows).to_csv(s.summary_csv, index=False)


def summarise_scenario(s: Scenario) -> None:
    if not s.raw_jsonl.exists():
        return

    records = []
    with open(s.raw_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    rows = []
    for la in LA_KEYS:
        vals = []
        for rec in records:
            pdr_map = rec.get("pdr", {})
            if la in pdr_map and pdr_map[la] is not None:
                vals.append(float(pdr_map[la]))

        if len(vals) == 0:
            rows.append(
                {
                    "tc": s.tc,
                    "devices": s.devices,
                    "radius_m": s.radius_m,
                    "wifi": s.wifi,
                    "lte": s.lte,
                    "period_s": s.period_s,
                    "sim_time_s": s.sim_time_s,
                    "rl_works": s.rl_works,
                    "scenario": s.scenario,
                    "policy_key": la,
                    "policy_name": LA_DISPLAY[la],
                    "n_runs": 0,
                    "pdr_mean": np.nan,
                    "pdr_std": np.nan,
                    "pdr_min": np.nan,
                    "pdr_median": np.nan,
                    "pdr_max": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "failed": False,
                }
            )
            continue

        arr = np.array(vals, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = (std / np.sqrt(n)) if n > 1 else 0.0

        rows.append(
            {
                "tc": s.tc,
                "devices": s.devices,
                "radius_m": s.radius_m,
                "wifi": s.wifi,
                "lte": s.lte,
                "period_s": s.period_s,
                "sim_time_s": s.sim_time_s,
                "rl_works": s.rl_works,
                "scenario": s.scenario,
                "policy_key": la,
                "policy_name": LA_DISPLAY[la],
                "n_runs": n,
                "pdr_mean": round(mean, 4),
                "pdr_std": round(std, 4),
                "pdr_min": round(float(np.min(arr)), 4),
                "pdr_median": round(float(np.median(arr)), 4),
                "pdr_max": round(float(np.max(arr)), 4),
                "ci95_low": round(mean - 1.96 * se, 4),
                "ci95_high": round(mean + 1.96 * se, 4),
                "failed": False,
            }
        )

    pd.DataFrame(rows).to_csv(s.summary_csv, index=False)


def run_scenario(
    s: Scenario,
    runs: int,
    timeout: int,
    program: str,
    dry_run: bool = False,
    max_consecutive_fails: int = 5,
) -> None:
    done = completed_runs(s)
    if done >= runs:
        log.info(f"  [{s.key}] already complete ({done}/{runs})")
        return

    consecutive_fails = 0

    for run_id in range(done + 1, runs + 1):
        log.info(f"  [{s.key}] run {run_id}/{runs}")
        cmd = build_ns3_cmd(s, program, run_id)
        cmd_str = " ".join(cmd)

        if dry_run:
            log.info(f"    DRY-RUN: {cmd_str}")
            rec = {"run": run_id, "dry_run": True, "pdr": {}}
            with open(s.raw_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            continue

        _clear_datasets()
        t0 = time.time()
        try:
            res = subprocess.run(
                cmd,
                cwd=NS3_DIR,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            consecutive_fails += 1
            log.warning(
                f"    timeout after {timeout}s [consec={consecutive_fails}/{max_consecutive_fails}]"
            )
            if consecutive_fails >= max_consecutive_fails:
                log.error(f"  [{s.key}] too many failures -> marking failed")
                write_failed_summary(s)
                return
            continue
        except Exception as ex:
            consecutive_fails += 1
            log.warning(
                f"    execution error: {ex} [consec={consecutive_fails}/{max_consecutive_fails}]"
            )
            if consecutive_fails >= max_consecutive_fails:
                log.error(f"  [{s.key}] too many failures -> marking failed")
                write_failed_summary(s)
                return
            continue

        elapsed = time.time() - t0
        if res.returncode != 0:
            consecutive_fails += 1
            log.warning(
                f"    sim failed rc={res.returncode} ({elapsed:.1f}s) "
                f"[consec={consecutive_fails}/{max_consecutive_fails}]"
            )
            stderr_tail = (res.stderr or "").strip().splitlines()[-5:]
            if stderr_tail:
                log.warning("    stderr tail: %s", " | ".join(stderr_tail))
            if consecutive_fails >= max_consecutive_fails:
                log.error(f"  [{s.key}] too many failures -> marking failed")
                write_failed_summary(s)
                return
            continue

        pdr = extract_pdr_by_policy()
        if not pdr:
            consecutive_fails += 1
            log.warning(
                f"    no datasets/PDR extracted [consec={consecutive_fails}/{max_consecutive_fails}]"
            )
            if consecutive_fails >= max_consecutive_fails:
                log.error(f"  [{s.key}] too many failures -> marking failed")
                write_failed_summary(s)
                return
            continue

        consecutive_fails = 0
        rec = {
            "run": run_id,
            "elapsed_s": round(elapsed, 2),
            "tc": s.tc,
            "params": {
                "devices": s.devices,
                "radius_m": s.radius_m,
                "wifi": s.wifi,
                "lte": s.lte,
                "period_s": s.period_s,
                "sim_time_s": s.sim_time_s,
            },
            "pdr": pdr,
        }
        with open(s.raw_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(
            "    done in %.1fs | %s",
            elapsed,
            " ".join(f"{LA_DISPLAY[k]}={v:.2f}%" for k, v in pdr.items()),
        )

    summarise_scenario(s)


def write_index(scenarios: List[Scenario]) -> None:
    rows = []
    for s in scenarios:
        rows.append(
            {
                "tc": s.tc,
                "key": s.key,
                "devices": s.devices,
                "radius_m": s.radius_m,
                "wifi": s.wifi,
                "lte": s.lte,
                "period_s": s.period_s,
                "sim_time_s": s.sim_time_s,
                "rl_works": s.rl_works,
                "scenario": s.scenario,
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "scenario_index.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Scenario index written -> %s", out)


def write_global_summary(scenarios: List[Scenario]) -> None:
    frames = []
    for s in scenarios:
        if s.summary_csv.exists():
            try:
                frames.append(pd.read_csv(s.summary_csv))
            except Exception:
                pass

    if not frames:
        return

    all_df = pd.concat(frames, ignore_index=True)
    long_out = RESULTS_DIR / "full_summary_long.csv"
    all_df.to_csv(long_out, index=False)

    pdr_df = all_df[
        [
            "tc",
            "scenario",
            "rl_works",
            "devices",
            "radius_m",
            "wifi",
            "lte",
            "period_s",
            "sim_time_s",
            "policy_key",
            "policy_name",
            "n_runs",
            "pdr_mean",
            "pdr_std",
            "ci95_low",
            "ci95_high",
            "failed",
        ]
    ].copy()

    pivot = pdr_df.pivot_table(
        index=[
            "tc",
            "scenario",
            "rl_works",
            "devices",
            "radius_m",
            "wifi",
            "lte",
            "period_s",
            "sim_time_s",
        ],
        columns="policy_key",
        values="pdr_mean",
        aggfunc="first",
    ).reset_index()

    for la in LA_KEYS:
        if la not in pivot.columns:
            pivot[la] = np.nan

    pivot = pivot[
        [
            "tc",
            "scenario",
            "rl_works",
            "devices",
            "radius_m",
            "wifi",
            "lte",
            "period_s",
            "sim_time_s",
            "no_la",
            "3gpp",
            "marl",
            "ppo",
            "ddqn",
        ]
    ].sort_values(by="tc")

    piv_out = RESULTS_DIR / "full_summary_pdr.csv"
    pivot.to_csv(piv_out, index=False)
    log.info("Global summaries written -> %s, %s", long_out, piv_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NB-IoT LA sweep scenarios")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Runs per scenario")
    parser.add_argument(
        "--resume", action="store_true", help="Skip scenarios with existing summary CSV"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands and write dry-run records only"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-run timeout (seconds)"
    )
    parser.add_argument(
        "--program", default=DEFAULT_PROGRAM, help="ns-3 program name passed to ./ns3 run"
    )
    parser.add_argument("--tc", nargs="*", help="Subset of test-cases, e.g. --tc 01 08 31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_index(scenarios)

    if args.tc:
        wanted = {tc.zfill(2) for tc in args.tc}
        scenarios = [s for s in scenarios if s.tc in wanted]
        log.info("Filtered to %d scenario(s)", len(scenarios))

    if args.resume:
        scenarios = [s for s in scenarios if not s.summary_csv.exists()]
        log.info("After resume filter: %d scenario(s)", len(scenarios))

    log.info(
        "Running %d scenario(s) x %d run(s) = %d simulation run(s)",
        len(scenarios),
        args.runs,
        len(scenarios) * args.runs,
    )

    t0 = time.time()
    for i, s in enumerate(scenarios, 1):
        log.info(
            "\n[%d/%d] %s | devices=%d radius=%dm wifi=%d lte=%d period=%ds sim=%ds | %s",
            i,
            len(scenarios),
            s.key,
            s.devices,
            s.radius_m,
            s.wifi,
            s.lte,
            s.period_s,
            s.sim_time_s,
            s.scenario,
        )
        run_scenario(
            s,
            runs=args.runs,
            timeout=args.timeout,
            program=args.program,
            dry_run=args.dry_run,
        )
        if not s.summary_csv.exists() and not args.dry_run:
            summarise_scenario(s)

    write_global_summary(scenarios)
    elapsed = time.time() - t0
    log.info("\nAll done in %.2fh -> %s", elapsed / 3600.0, RESULTS_DIR)


if __name__ == "__main__":
    main()
