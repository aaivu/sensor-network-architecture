"""Patch run_sweeps.py to bail out after 5 consecutive failures."""

with open("run_sweeps.py", "r") as f:
    src = f.read()

# 1. Add MAX_CONSECUTIVE_FAILS constant + _write_failed_summary function
#    inserted right before def run_config(
failed_summary_fn = '''
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


'''

assert "def run_config(cfg: SweepConfig" in src, "anchor not found"
src = src.replace(
    "def run_config(cfg: SweepConfig",
    failed_summary_fn + "def run_config(cfg: SweepConfig"
)

# 2. Add consecutive_fails = 0 before the run loop
old_loop_start = (
    "    for run_i in range(done_so_far + 1, target_runs + 1):\n"
    "        log.info(f\"  [{cfg.key}] Run {run_i}/{target_runs}\")"
)
new_loop_start = (
    "    consecutive_fails = 0\n"
    "    for run_i in range(done_so_far + 1, target_runs + 1):\n"
    "        log.info(f\"  [{cfg.key}] Run {run_i}/{target_runs}\")"
)
assert old_loop_start in src, "loop start anchor not found"
src = src.replace(old_loop_start, new_loop_start, 1)

# 3. After "Sim failed" log, add consecutive failure check
old_fail = (
    "            log.warning(f\"    Sim failed (rc={result.returncode}) in {elapsed:.1f}s\")\n"
    "            continue"
)
new_fail = (
    "            consecutive_fails += 1\n"
    "            log.warning(f\"    Sim failed (rc={result.returncode}) in {elapsed:.1f}s"
    " [consec={consecutive_fails}/{MAX_CONSECUTIVE_FAILS}]\")\n"
    "            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:\n"
    "                log.error(f\"  [{cfg.key}] {MAX_CONSECUTIVE_FAILS} consecutive sim failures\"\n"
    "                          f\" -- marking as failed and skipping\")\n"
    "                _write_failed_summary(cfg)\n"
    "                return\n"
    "            continue"
)
assert old_fail in src, "fail anchor not found"
src = src.replace(old_fail, new_fail, 1)

# 4. Reset consecutive_fails on a successful run (before writing record)
old_success = (
    "        record = {\"run\": run_i, \"elapsed_s\": round(elapsed, 1), **metrics}\n"
    "        with open(cfg.raw_jsonl, \"a\") as f:"
)
new_success = (
    "        consecutive_fails = 0\n"
    "        record = {\"run\": run_i, \"elapsed_s\": round(elapsed, 1), **metrics}\n"
    "        with open(cfg.raw_jsonl, \"a\") as f:"
)
assert old_success in src, "success anchor not found"
src = src.replace(old_success, new_success, 1)

with open("run_sweeps.py", "w") as f:
    f.write(src)

print("Patch applied successfully!")
print("Verifying...")
with open("run_sweeps.py") as f:
    text = f.read()
assert "MAX_CONSECUTIVE_FAILS" in text
assert "_write_failed_summary" in text
assert "consecutive_fails" in text
print("All checks passed.")
