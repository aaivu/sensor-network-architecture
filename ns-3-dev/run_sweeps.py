#!/usr/bin/env python3
"""
LoRaWAN ADR Parametric Sweep Runner
=====================================
Generates and runs two sweeps:
  Sweep A — PDR vs nDevices  (devices: 10..100 step 10, radii: 50,100,200,500m)
  Sweep B — PDR vs Radius    (radii: 50..1000m step 25, devices: 5,10,20,50)

Common fixed params: WiFi=5, LTE=10, appPeriod=60s, simTime=5000s,
                     environmental=true, adr=all, 100 runs per config.

Configs that appear in BOTH sweeps are run only ONCE (deduplication).
Results stored in sweep_results/ as per-config JSONL + summary CSV.

Usage:
    cd ns-3-dev
    source venv/bin/activate
    python3 run_sweeps.py                   # all configs
    python3 run_sweeps.py --resume          # skip already-finished configs
    python3 run_sweeps.py --dry-run         # print commands only
    python3 run_sweeps.py --runs 5          # override run count
    python3 run_sweeps.py --config A-01 B-003  # specific configs
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ───────────────────────────────────────────────────────────────────
NS3_DIR      = Path(__file__).parent.resolve()
DATASETS_DIR = NS3_DIR / "lorawan_datasets"
RESULTS_DIR  = NS3_DIR / "sweep_results"
NS3_BIN      = NS3_DIR / "ns3"
PYTHON_BIN   = shutil.which("python3.13") or shutil.which("python3") or "python3"

# ── fixed sweep parameters ──────────────────────────────────────────────────
WIFI        = 5
LTE         = 10
APP_PERIOD  = 60
SIM_TIME    = 5000
ENV         = True
RUNS        = 100

# ── sweep definitions ────────────────────────────────────────────────────────
SWEEP_A_RADII   = [50, 100, 200, 500]
SWEEP_A_DEVICES = list(range(10, 101, 10))   # 10,20,...,100

SWEEP_B_DEVICES = [5, 10, 20, 50]
SWEEP_B_RADII   = list(range(50, 1001, 25))  # 50,75,...,1000

# ── ADR file patterns ────────────────────────────────────────────────────────
ADR_PATTERNS = {
    "no_adr":       "no_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "classical_adr":"adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "ddqn_adr":     "ddqn_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "ppo_adr":      "ppo_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
    "marl_adr":     "marl_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv",
}
ADR_DISPLAY = {
    "no_adr":       "No ADR",
    "classical_adr":"Classical ADR",
    "ddqn_adr":     "DDQN-PER ADR",
    "ppo_adr":      "PPO ADR",
    "marl_adr":     "MARL ADR",
}
METRIC_COLS = [
    "pdr_pct", "avg_snr_db", "std_snr_db", "avg_sf", "std_sf",
    "avg_tx_power_dbm", "avg_rssi_dbm", "total_energy_mJ",
    "energy_per_success_mJ", "avg_toa_ms", "snr_margin_median_db",
]

SNR_THRESHOLDS = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


# ── config dataclass ─────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    nDevices: int
    radius:   int
    wifi:     int = WIFI
    lte:      int = LTE
    period:   int = APP_PERIOD
    sim_time: int = SIM_TIME
    env:      bool = ENV
    # which named sweeps reference this config (for bookkeeping)
    sweep_a_ids: List[str] = field(default_factory=list)
    sweep_b_ids: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique filesystem-safe identifier."""
        return f"n{self.nDevices:03d}_r{self.radius:04d}"

    @property
    def raw_jsonl(self) -> Path:
        return RESULTS_DIR / f"{self.key}_raw.jsonl"

    @property
    def summary_csv(self) -> Path:
        return RESULTS_DIR / f"{self.key}_summary.csv"

    def ns3_cmd(self) -> List[str]:
        env_str = "true" if self.env else "false"
        return [
            PYTHON_BIN, str(NS3_BIN), "run",
            (f"advanced-ddqn-per-adr-example "
             f"--nDevices={self.nDevices} "
             f"--radius={self.radius} "
             f"--wifiInterferers={self.wifi} "
             f"--lteInterferers={self.lte} "
             f"--appPeriod={self.period} "
             f"--simulationTime={self.sim_time} "
             f"--adr=all "
             f"--environmental={env_str}"),
        ]

    def completed_runs(self) -> int:
        """Count runs already stored in the JSONL file."""
        if not self.raw_jsonl.exists():
            return 0
        count = 0
        with open(self.raw_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
        return count


# ── build deduplicated config list ─────────────────────────────────────────

def build_configs() -> List[SweepConfig]:
    cfg_map: Dict[str, SweepConfig] = {}

    # Sweep A
    for ridx, r in enumerate(SWEEP_A_RADII):
        for didx, n in enumerate(SWEEP_A_DEVICES):
            aid = f"A-{ridx * len(SWEEP_A_DEVICES) + didx + 1:02d}"
            key = f"n{n:03d}_r{r:04d}"
            if key not in cfg_map:
                cfg_map[key] = SweepConfig(nDevices=n, radius=r)
            cfg_map[key].sweep_a_ids.append(aid)

    # Sweep B
    seq = 1
    for n in SWEEP_B_DEVICES:
        for r in SWEEP_B_RADII:
            bid = f"B-{seq:03d}"
            seq += 1
            key = f"n{n:03d}_r{r:04d}"
            if key not in cfg_map:
                cfg_map[key] = SweepConfig(nDevices=n, radius=r)
            cfg_map[key].sweep_b_ids.append(bid)

    configs = sorted(cfg_map.values(), key=lambda c: (c.nDevices, c.radius))
    return configs


# ── metric extraction ───────────────────────────────────────────────────────

def _toa_ms(sf: int, bw_khz: int = 125, cr: int = 1,
            payload_bytes: int = 20) -> float:
    t_sym     = (2 ** sf) / (bw_khz * 1000) * 1000
    t_pre     = (8 + 4.25) * t_sym
    de        = 1 if sf >= 11 else 0
    n         = 8 * payload_bytes - 4 * sf + 28 + 16
    d         = 4 * (sf - 2 * de) or (4 * sf)
    n_payload = 8 + max(np.ceil(n / d) * (cr + 4), 0)
    return t_pre + n_payload * t_sym


def extract_metrics(datasets_dir: Path) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for adr_key, pattern in ADR_PATTERNS.items():
        files = glob.glob(str(datasets_dir / pattern))
        if not files:
            continue
        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)

        has_snr  = "snr_db"            in combined.columns
        has_rssi = "pre_tx_rssi_dBm"   in combined.columns
        has_sf   = "sf"                in combined.columns
        has_tp   = "tx_power_dBm"      in combined.columns
        has_nrg  = "energy_consumed_mJ" in combined.columns

        total     = len(combined)
        n_success = int((combined["snr_db"] != -999.0).sum()) if has_snr else 0
        pdr       = n_success / total * 100.0 if total > 0 else 0.0
        snr_vals  = (combined.loc[combined["snr_db"] != -999.0, "snr_db"]
                     if has_snr else pd.Series(dtype=float))

        toa_mean = float("nan")
        if has_sf:
            try:
                toa_mean = combined["sf"].apply(lambda s: _toa_ms(int(s))).mean()
            except Exception:
                pass

        snr_margin_median = float("nan")
        if has_snr and has_sf and len(snr_vals) > 0:
            try:
                margins = combined.loc[combined["snr_db"] != -999.0].apply(
                    lambda row: row["snr_db"] - SNR_THRESHOLDS.get(int(row["sf"]), -7.5),
                    axis=1,
                )
                snr_margin_median = float(margins.median())
            except Exception:
                pass

        total_nrg = float(combined["energy_consumed_mJ"].sum()) if has_nrg else float("nan")
        eps = (total_nrg / n_success) if (has_nrg and n_success > 0) else float("nan")

        results[adr_key] = {
            "pdr_pct":               round(pdr,   4),
            "avg_snr_db":            round(float(snr_vals.mean()), 4) if len(snr_vals) else float("nan"),
            "std_snr_db":            round(float(snr_vals.std()),  4) if len(snr_vals) else float("nan"),
            "avg_sf":                round(float(combined["sf"].mean()), 4) if has_sf else float("nan"),
            "std_sf":                round(float(combined["sf"].std()),  4) if has_sf else float("nan"),
            "avg_tx_power_dbm":      round(float(combined["tx_power_dBm"].mean()), 4) if has_tp else float("nan"),
            "avg_rssi_dbm":          round(float(combined["pre_tx_rssi_dBm"].mean()), 4) if has_rssi else float("nan"),
            "total_energy_mJ":       round(total_nrg, 4) if not np.isnan(total_nrg) else float("nan"),
            "energy_per_success_mJ": round(eps,   4) if not np.isnan(eps) else float("nan"),
            "avg_toa_ms":            round(toa_mean, 4) if not np.isnan(toa_mean) else float("nan"),
            "snr_margin_median_db":  round(snr_margin_median, 4) if not np.isnan(snr_margin_median) else float("nan"),
        }
    return results


# ── summary statistics ──────────────────────────────────────────────────────

def summarise(run_records: List[dict], cfg: SweepConfig) -> pd.DataFrame:
    per_adr: Dict[str, Dict[str, List]] = {
        k: {m: [] for m in METRIC_COLS} for k in ADR_PATTERNS
    }
    for run in run_records:
        for adr_key in ADR_PATTERNS:
            if adr_key not in run:
                continue
            for m in METRIC_COLS:
                v = run[adr_key].get(m, float("nan"))
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    per_adr[adr_key][m].append(float(v))

    rows = []
    for adr_key in ADR_PATTERNS:
        for m in METRIC_COLS:
            vals = np.array(per_adr[adr_key][m])
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se   = std / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "nDevices":   cfg.nDevices,
                "radius":     cfg.radius,
                "wifi":       cfg.wifi,
                "lte":        cfg.lte,
                "appPeriod":  cfg.period,
                "simTime":    cfg.sim_time,
                "adr_key":    adr_key,
                "adr_name":   ADR_DISPLAY[adr_key],
                "metric":     m,
                "n_runs":     n,
                "mean":       round(mean, 4),
                "std":        round(std,  4),
                "min":        round(float(np.min(vals)), 4),
                "p5":         round(float(np.percentile(vals,  5)), 4),
                "median":     round(float(np.median(vals)), 4),
                "p95":        round(float(np.percentile(vals, 95)), 4),
                "max":        round(float(np.max(vals)), 4),
                "ci95_low":   round(mean - 1.96 * se, 4),
                "ci95_high":  round(mean + 1.96 * se, 4),
            })
    return pd.DataFrame(rows)


# ── runner ──────────────────────────────────────────────────────────────────

def _clear_datasets():
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)



MAX_CONSECUTIVE_FAILS = 5


def _write_failed_summary(cfg):
    """Write a NaN-filled summary CSV so this config is skipped on resume."""
    nan = float("nan")
    rows = []
    for adr_key in ADR_PATTERNS:
        for m in METRIC_COLS:
            rows.append({
                "nDevices":   cfg.nDevices,
                "radius":     cfg.radius,
                "wifi":       cfg.wifi,
                "lte":        cfg.lte,
                "appPeriod":  cfg.period,
                "simTime":    cfg.sim_time,
                "adr_key":    adr_key,
                "adr_name":   ADR_DISPLAY[adr_key],
                "metric":     m,
                "n_runs":     0,
                "mean":       nan, "std": nan,
                "min":        nan, "p5": nan, "median": nan,
                "p95":        nan, "max": nan,
                "ci95_low":   nan, "ci95_high": nan,
                "failed":     True,
            })
    import pandas as _pd
    df = _pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.summary_csv, index=False)
    log.warning(f"  [{cfg.key}] FAILED summary written (all NaN) -> {cfg.summary_csv.name}")


def run_config(cfg: SweepConfig, target_runs: int,
               dry_run: bool = False, timeout: int = 900):
    """Run one config up to target_runs times (resumes if JSONL already partial)."""
    done_so_far = cfg.completed_runs()
    remaining   = target_runs - done_so_far
    if remaining <= 0:
        log.info(f"  [{cfg.key}] Already complete ({done_so_far} runs) — skipping")
        return
    if done_so_far:
        log.info(f"  [{cfg.key}] Resuming from run {done_so_far + 1}/{target_runs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = cfg.ns3_cmd()
    cmd_str = " ".join(cmd)

    consecutive_fails = 0
    for run_i in range(done_so_far + 1, target_runs + 1):
        log.info(f"  [{cfg.key}] Run {run_i}/{target_runs}")
        if dry_run:
            log.info(f"    DRY-RUN: {cmd_str}")
            # write placeholder
            record = {"run": run_i, "dry_run": True}
            with open(cfg.raw_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
            continue

        _clear_datasets()
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            log.error(f"    Timeout after {timeout}s")
            continue
        except Exception as e:
            log.error(f"    Error: {e}")
            continue
        elapsed = time.time() - t0

        if result.returncode != 0:
            consecutive_fails += 1
            log.warning(f"    Sim failed (rc={result.returncode}) in {elapsed:.1f}s [consec={consecutive_fails}/{MAX_CONSECUTIVE_FAILS}]")
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                log.error(f"  [{cfg.key}] {MAX_CONSECUTIVE_FAILS} consecutive sim failures"
                          f" -- marking as failed and skipping")
                _write_failed_summary(cfg)
                return
            continue

        metrics = extract_metrics(DATASETS_DIR)
        if not metrics:
            log.warning(f"    No datasets found after run {run_i}")
            continue

        consecutive_fails = 0
        record = {"run": run_i, "elapsed_s": round(elapsed, 1), **metrics}
        with open(cfg.raw_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(f"    Done in {elapsed:.1f}s — PDR(MARL)="
                 f"{metrics.get('marl_adr', {}).get('pdr_pct', '?')}%")

    # Write summary CSV
    records = []
    if cfg.raw_jsonl.exists():
        with open(cfg.raw_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if records:
        df_sum = summarise(records, cfg)
        df_sum.to_csv(cfg.summary_csv, index=False)
        log.info(f"  [{cfg.key}] Summary written → {cfg.summary_csv.name}")


# ── index file ──────────────────────────────────────────────────────────────

def write_index(configs: List[SweepConfig]):
    rows = []
    for c in configs:
        rows.append({
            "key":         c.key,
            "nDevices":    c.nDevices,
            "radius":      c.radius,
            "wifi":        c.wifi,
            "lte":         c.lte,
            "appPeriod":   c.period,
            "simTime":     c.sim_time,
            "sweep_a_ids": ",".join(c.sweep_a_ids),
            "sweep_b_ids": ",".join(c.sweep_b_ids),
            "in_sweep_a":  len(c.sweep_a_ids) > 0,
            "in_sweep_b":  len(c.sweep_b_ids) > 0,
        })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "sweep_index.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"Index written → {out}  ({len(rows)} unique configs)")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",     type=int, default=RUNS)
    parser.add_argument("--resume",   action="store_true",
                        help="Skip configs whose summary CSV already exists")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--timeout",  type=int, default=900,
                        help="Per-simulation timeout in seconds")
    parser.add_argument("--config",   nargs="*",
                        help="Run only specific config keys or sweep IDs, e.g. A-01 B-003 n010_r0050")
    args = parser.parse_args()

    configs = build_configs()
    write_index(configs)

    # Count deduplication
    a_ids = sum(len(c.sweep_a_ids) for c in configs)
    b_ids = sum(len(c.sweep_b_ids) for c in configs)
    both  = sum(1 for c in configs if c.sweep_a_ids and c.sweep_b_ids)
    log.info(f"Sweep A: {a_ids} configs | Sweep B: {b_ids} configs | "
             f"Shared (run once): {both} | Unique total: {len(configs)}")
    log.info(f"Target: {args.runs} runs × {len(configs)} configs = "
             f"{args.runs * len(configs):,} simulation runs")

    # Filter if --config specified
    if args.config:
        wanted: Set[str] = set(args.config)
        configs = [
            c for c in configs
            if (c.key in wanted
                or any(aid in wanted for aid in c.sweep_a_ids)
                or any(bid in wanted for bid in c.sweep_b_ids))
        ]
        log.info(f"Filtered to {len(configs)} configs")

    # Skip completed if --resume
    if args.resume:
        configs = [c for c in configs if not c.summary_csv.exists()]
        log.info(f"After resume filter: {len(configs)} configs remaining")

    t_start = time.time()
    for i, cfg in enumerate(configs, 1):
        log.info(f"\n[{i}/{len(configs)}] Config {cfg.key} "
                 f"(n={cfg.nDevices}, r={cfg.radius}m) "
                 f"sweeps: A={cfg.sweep_a_ids or '-'} B={cfg.sweep_b_ids or '-'}")
        run_config(cfg, target_runs=args.runs,
                   dry_run=args.dry_run, timeout=args.timeout)

    elapsed = time.time() - t_start
    log.info(f"\nAll done in {elapsed/3600:.2f}h. "
             f"Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
