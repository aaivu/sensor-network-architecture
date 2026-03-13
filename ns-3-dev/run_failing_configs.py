"""
Run 100 simulations for n030_r0500, n060_r0500, n070_r0500 using simTime=500
(simTime=5000 triggers a SIGSEGV for these specific configs).
Writes proper summary CSVs so they appear as real data in plots.
"""
import json, shutil, subprocess, time, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

NS3_DIR     = Path(__file__).parent
RESULTS_DIR = NS3_DIR / "sweep_results"
DATASETS_DIR = NS3_DIR / "lorawan_datasets"

ADR_PATTERNS = {
    "no_adr":    "no_adr_",
    "adr":       "adr_",
    "ddqn_adr":  "ddqn_adr_",
    "ppo_adr":   "ppo_adr_",
    "marl_adr":  "marl_adr_",
}
ADR_DISPLAY = {
    "no_adr":   "No ADR",
    "adr":      "Classical ADR",
    "ddqn_adr": "DDQN-PER ADR",
    "ppo_adr":  "PPO ADR",
    "marl_adr": "MARL ADR",
}
METRIC_COLS = [
    "pdr_pct", "avg_snr_db", "std_snr_db", "avg_sf", "std_sf",
    "avg_tx_power_dbm", "avg_rssi_dbm", "total_energy_mJ",
    "energy_per_success_mJ", "avg_toa_ms", "snr_margin_median_db",
]

FAILING_CONFIGS = [
    {"nDevices": 30, "radius": 500},
    {"nDevices": 60, "radius": 500},
    {"nDevices": 70, "radius": 500},
]
SIM_TIME = 500   # reduced from 5000 due to SIGSEGV at higher times
RUNS     = 100
WIFI     = 5
LTE      = 10
PERIOD   = 60


def clear_datasets():
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)


def extract_metrics(datasets_dir: Path) -> dict:
    if not datasets_dir.exists():
        return {}
    results = {}
    for adr_key, prefix in ADR_PATTERNS.items():
        csvs = list(datasets_dir.glob(f"{prefix}*.csv"))
        if not csvs:
            continue
        frames = []
        for c in csvs:
            try:
                df = pd.read_csv(c)
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        has_pdr  = "packets_received" in combined.columns and "packets_sent" in combined.columns
        pdr = (combined["packets_received"].sum() / combined["packets_sent"].sum() * 100
               if has_pdr and combined["packets_sent"].sum() > 0 else float("nan"))
        snr_col   = next((c for c in combined.columns if "snr" in c.lower()), None)
        snr_vals  = combined[snr_col].dropna() if snr_col else pd.Series([], dtype=float)
        has_sf    = "sf" in combined.columns
        has_tp    = "tx_power_dBm" in combined.columns
        has_rssi  = "pre_tx_rssi_dBm" in combined.columns
        has_toa   = "toa_ms" in combined.columns
        has_snrm  = "snr_margin_dB" in combined.columns
        has_nrg   = "energy_consumed_mJ" in combined.columns
        n_success = int(combined["packets_received"].sum()) if has_pdr else 0
        toa_mean  = float(combined["toa_ms"].mean()) if has_toa and not combined["toa_ms"].empty else float("nan")
        snr_margin_median = float(combined["snr_margin_dB"].median()) if has_snrm else float("nan")
        total_nrg = float(combined["energy_consumed_mJ"].sum()) if has_nrg else float("nan")
        eps = (total_nrg / n_success) if (has_nrg and n_success > 0) else float("nan")
        results[adr_key] = {
            "pdr_pct":               round(pdr, 4),
            "avg_snr_db":            round(float(snr_vals.mean()), 4) if len(snr_vals) else float("nan"),
            "std_snr_db":            round(float(snr_vals.std()),  4) if len(snr_vals) else float("nan"),
            "avg_sf":                round(float(combined["sf"].mean()), 4) if has_sf else float("nan"),
            "std_sf":                round(float(combined["sf"].std()),  4) if has_sf else float("nan"),
            "avg_tx_power_dbm":      round(float(combined["tx_power_dBm"].mean()), 4) if has_tp else float("nan"),
            "avg_rssi_dbm":          round(float(combined["pre_tx_rssi_dBm"].mean()), 4) if has_rssi else float("nan"),
            "total_energy_mJ":       round(total_nrg, 4) if not np.isnan(total_nrg) else float("nan"),
            "energy_per_success_mJ": round(eps, 4) if not np.isnan(eps) else float("nan"),
            "avg_toa_ms":            round(toa_mean, 4) if not np.isnan(toa_mean) else float("nan"),
            "snr_margin_median_db":  round(snr_margin_median, 4) if not np.isnan(snr_margin_median) else float("nan"),
        }
    return results


def run_config(nDevices, radius):
    key = f"n{nDevices:03d}_r{radius:04d}"
    raw_jsonl   = RESULTS_DIR / f"{key}_raw.jsonl"
    summary_csv = RESULTS_DIR / f"{key}_summary.csv"

    # Delete NaN placeholder
    if summary_csv.exists():
        df_check = pd.read_csv(summary_csv)
        if "failed" in df_check.columns and df_check["failed"].iloc[0] == True:
            print(f"[{key}] Removing NaN placeholder CSV")
            summary_csv.unlink()
        else:
            print(f"[{key}] Real summary already exists — skipping")
            return

    # Count already-completed runs
    done = 0
    if raw_jsonl.exists():
        with open(raw_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if not rec.get("failed"):
                        done += 1
                except Exception:
                    pass
    if done >= RUNS:
        print(f"[{key}] Already {done} runs — writing summary")
    else:
        print(f"[{key}] Starting {RUNS} runs with simTime={SIM_TIME} (crash workaround)")
        cmd = [
            "python3.13", "ns3", "run",
            f"advanced-ddqn-per-adr-example "
            f"--nDevices={nDevices} --radius={radius} "
            f"--wifiInterferers={WIFI} --lteInterferers={LTE} "
            f"--appPeriod={PERIOD} --simulationTime={SIM_TIME} "
            f"--adr=all --environmental=true"
        ]
        for run_i in range(done + 1, RUNS + 1):
            clear_datasets()
            t0 = time.time()
            result = subprocess.run(cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=300)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"  Run {run_i}/{RUNS}: FAILED (rc={result.returncode}) in {elapsed:.1f}s")
                continue
            metrics = extract_metrics(DATASETS_DIR)
            if not metrics:
                print(f"  Run {run_i}/{RUNS}: no datasets")
                continue
            record = {"run": run_i, "elapsed_s": round(elapsed, 1), **metrics}
            with open(raw_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
            pdr = metrics.get("marl_adr", {}).get("pdr_pct", "?")
            print(f"  Run {run_i}/{RUNS}: done in {elapsed:.1f}s  PDR(MARL)={pdr}%")

    # Build summary
    records = []
    if raw_jsonl.exists():
        with open(raw_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if not r.get("failed"):
                        records.append(r)
                except Exception:
                    pass

    if not records:
        print(f"[{key}] No valid records — leaving NaN placeholder")
        return

    per_adr = {k: {m: [] for m in METRIC_COLS} for k in ADR_PATTERNS}
    for run in records:
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
                "nDevices":   nDevices,
                "radius":     radius,
                "wifi":       WIFI,
                "lte":        LTE,
                "appPeriod":  PERIOD,
                "simTime":    SIM_TIME,  # NOTE: 500 not 5000
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

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[{key}] Summary written ({len(records)} runs, simTime={SIM_TIME})")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in FAILING_CONFIGS:
        print(f"\n{'='*60}")
        run_config(cfg["nDevices"], cfg["radius"])
    print("\nDone.")
