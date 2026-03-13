#!/bin/bash
# resume_sweeps.sh — Resume LoRaWAN ADR sweep from wherever it stopped
# Safe to run any number of times. Already-finished configs are skipped.
# Usage:
#   ./resume_sweeps.sh               (run / restart anytime)
#   ./resume_sweeps.sh --runs 5      (quick test with fewer runs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"
RUNNER="$SCRIPT_DIR/run_sweeps.py"
RESULTS="$SCRIPT_DIR/sweep_results"
LOG_FILE="$RESULTS/sweep_run.log"
RUNS="100"
MAX_RETRIES=999
RETRY_WAIT=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: run_sweeps.py not found at $RUNNER"
    exit 1
fi

mkdir -p "$RESULTS"
source "$VENV"

count_done() {
    find "$RESULTS" -maxdepth 1 -name "n???_r????_summary.csv" 2>/dev/null | wc -l | tr -d ' '
}

count_total() {
    local index="$RESULTS/sweep_index.csv"
    if [[ -f "$index" ]]; then
        local lines
        lines=$(wc -l < "$index" | tr -d ' ')
        echo $(( lines - 1 ))
    else
        echo "184"
    fi
}

echo "════════════════════════════════════════════════════════════════"
echo "  LoRaWAN ADR Sweep — Auto-Resume Runner"
echo "  Runs per config : $RUNS"
echo "  Results dir     : $RESULTS"
echo "  Log file        : $LOG_FILE"
echo "════════════════════════════════════════════════════════════════"
echo ""

attempt=0
while true; do
    attempt=$(( attempt + 1 ))
    done_count=$(count_done)
    total_count=$(count_total)

    if [[ "$total_count" -gt 0 && "$done_count" -ge "$total_count" ]]; then
        echo ""
        echo "All $total_count configs complete."
        echo "Generate plots:  python3 plot_sweeps.py"
        exit 0
    fi

    remaining=$(( total_count - done_count ))
    echo "[$(date '+%H:%M:%S')] Attempt $attempt — $done_count/$total_count done, $remaining remaining"
    echo "────────────────────────────────────────────────────────────"

    python3 "$RUNNER" --runs "$RUNS" --resume 2>&1 | tee -a "$LOG_FILE"
    py_exit=${PIPESTATUS[0]}

    done_count=$(count_done)
    total_count=$(count_total)

    if [[ "$total_count" -gt 0 && "$done_count" -ge "$total_count" ]]; then
        echo ""
        echo "Sweep complete! $done_count/$total_count configs finished."
        echo "Generate plots:  python3 plot_sweeps.py"
        exit 0
    fi

    remaining=$(( total_count - done_count ))
    echo ""
    echo "[$(date '+%H:%M:%S')] Exited (code $py_exit). $done_count/$total_count done, $remaining remaining."

    if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
        echo "ERROR: Reached max retries. Check $LOG_FILE"
        exit 1
    fi

    echo "Restarting in ${RETRY_WAIT}s … (Ctrl+C to stop)"
    sleep "$RETRY_WAIT"
done
