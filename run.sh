#!/usr/bin/env bash
# =============================================================================
# run.sh — Convenience wrapper to launch lorawan_rl_adr simulations
#
# Prerequisites: run ./setup.sh first
#
# Usage examples:
#   ./run.sh --adr=ddqn
#   ./run.sh --adr=all  --nDevices=50 --simulationTime=7200
#   ./run.sh --adr=ppo  --nDevices=20 --rngSeed=42
#   ./run.sh --help
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="$REPO_DIR/ns3/ns-3-dev"
EXAMPLE="lorawan_rl_adr"

# ── Help ──────────────────────────────────────────────────────────────────────
usage() {
cat <<EOF
${BOLD}run.sh${NC} — Run a LoRaWAN RL ADR simulation

${BOLD}Usage:${NC}
  ./run.sh [NS-3 args...]

${BOLD}ADR Modes (--adr=<mode>):${NC}
  off         No ADR (baseline)
  on          Classical LoRaWAN ADR (LoRaWAN 1.0.4)
  ddqn        DDQN-PER (Dueling DQN + Prioritised Experience Replay)
  ppo         PPO (Proximal Policy Optimisation)
  marl        MARL — Independent Q-Learning with coordination signal
  marl_ppo    MARL-PPO — Independent PPO with coordination
  mappo_gat   MAPPO-GAT — Multi-Agent PPO with Graph Attention (CTDE)
  all         Run all 7 modes sequentially (produces 7 CSV files)

${BOLD}Common flags:${NC}
  --nDevices=N        Number of end-devices           (default: 20)
  --radius=M          Network radius in metres         (default: 500)
  --simulationTime=S  Simulation duration in seconds   (default: 300)
  --rngSeed=K         RNG seed for reproducibility     (default: 12345)
  --coordDivisor=D    Coordination-signal divisor       (default: 3.0)
  --rewardSchedule=R  0=adaptive  1=fixed  2=outcome   (default: 0)
  --csvFile=FILE      Override output CSV filename

${BOLD}Examples:${NC}
  ./run.sh --adr=ddqn
  ./run.sh --adr=all --nDevices=50 --simulationTime=7200
  ./run.sh --adr=mappo_gat --nDevices=30 --rngSeed=42

${BOLD}Tip:${NC} Run ./setup.sh first if you haven't already.
EOF
  exit 0
}

for arg in "$@"; do
  [[ "$arg" == "--help" || "$arg" == "-h" ]] && usage
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d "$NS3_DIR" ]] || die "ns3/ns-3-dev/ not found. Run ./setup.sh first."
[[ -f "$NS3_DIR/ns3" ]] || die "ns3 build script not found at $NS3_DIR/ns3. Run ./setup.sh first."

BINARY="$NS3_DIR/build/src/lorawan/examples/ns3.45-${EXAMPLE}-default"
if [[ ! -f "$BINARY" ]]; then
  # Try alternate naming convention
  BINARY=$(find "$NS3_DIR/build" -name "*${EXAMPLE}*" -type f -perm +111 2>/dev/null | head -1 || true)
  [[ -n "$BINARY" ]] || die "Simulation binary not found. Run ./setup.sh to build."
fi

# ── Build args string ─────────────────────────────────────────────────────────
ARGS="$*"
[[ -z "$ARGS" ]] && { warn "No arguments given. Using defaults (--adr=ddqn)."; ARGS="--adr=ddqn"; }

# ── Run ───────────────────────────────────────────────────────────────────────
info "Launching: $EXAMPLE $ARGS"
echo
cd "$NS3_DIR"
./ns3 run "${EXAMPLE} ${ARGS}"
echo
ok "Simulation finished. CSV written to the ns3/ working directory."
echo -e "  Analyse results: ${CYAN}source .venv/bin/activate && python analysis/analysis.py${NC}"
