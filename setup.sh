#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot setup for lorawan-rl-adr
#
# What this does:
#   1. Initialises the ns-3 git submodule (ns3/)
#   2. Copies our simulation files into ns3/src/lorawan/examples/
#   3. Registers lorawan_rl_adr in the lorawan examples CMakeLists.txt
#   4. Configures and builds ns-3 (Release, examples enabled)
#   5. Sets up a Python virtualenv for the analysis scripts
#
# Usage:
#   ./setup.sh               # full setup
#   ./setup.sh --skip-ns3    # skip ns-3 build (e.g. already built)
#   ./setup.sh --skip-python # skip Python venv setup
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Argument parsing ─────────────────────────────────────────────────────────
SKIP_NS3=false
SKIP_PYTHON=false
for arg in "$@"; do
  case $arg in
    --skip-ns3)    SKIP_NS3=true ;;
    --skip-python) SKIP_PYTHON=true ;;
    --help|-h)
      echo "Usage: $0 [--skip-ns3] [--skip-python]"
      exit 0 ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_SUBMODULE="$REPO_DIR/ns3"
NS3_DIR="$NS3_SUBMODULE/ns-3-dev"
EXAMPLES_DIR="$NS3_DIR/src/lorawan/examples"
SIM_DIR="$REPO_DIR/sim"

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  LoRaWAN Multi-Algorithm RL ADR — Setup${NC}"
echo -e "${BOLD}============================================================${NC}"
echo

# ── 1. Prerequisites check ───────────────────────────────────────────────────
info "Checking prerequisites..."

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    warn "$1 not found — $2"
    return 1
  fi
  ok "$1 found: $(command -v "$1")"
  return 0
}

check_cmd git    "install via package manager" || die "git is required"
check_cmd cmake  "install cmake >= 3.16"       || die "cmake is required"
check_cmd python3 "install Python >= 3.9"      || SKIP_PYTHON=true

# C++ compiler
if ! check_cmd g++ "install GCC >= 11" && ! check_cmd clang++ "install Clang >= 13"; then
  die "No C++ compiler found. Install GCC (>= 11) or Clang (>= 13)."
fi

echo

# ── 2. Initialise ns-3 submodule ─────────────────────────────────────────────
if [[ "$SKIP_NS3" == false ]]; then
  info "Initialising ns-3 submodule (ns3/)..."
  cd "$REPO_DIR"

  if [[ -f "$NS3_DIR/ns3" ]]; then
    ok "ns3/ submodule already initialised."
  else
    git submodule update --init --recursive --progress ns3
    ok "ns3/ submodule ready (ns-3 at ns3/ns-3-dev/)."
  fi
  echo
fi

# ── 3. Copy simulation files into ns-3 ───────────────────────────────────────
info "Copying simulation files into ns3/src/lorawan/examples/..."

[[ -d "$EXAMPLES_DIR" ]] || die "Expected directory not found: $EXAMPLES_DIR"

# Copy entry-point
cp -f "$SIM_DIR/lorawan_rl_adr.cc" "$EXAMPLES_DIR/"
ok "  Copied lorawan_rl_adr.cc"

# Copy headers (create include/ dir if needed)
mkdir -p "$EXAMPLES_DIR/include"
cp -rf "$SIM_DIR/include/." "$EXAMPLES_DIR/include/"
ok "  Copied include/ ($(ls "$SIM_DIR/include/" | wc -l | tr -d ' ') headers)"

echo

# ── 4. Register example in CMakeLists.txt ────────────────────────────────────
CMAKE_FILE="$EXAMPLES_DIR/CMakeLists.txt"
EXAMPLE_NAME="lorawan_rl_adr"

if grep -q "$EXAMPLE_NAME" "$CMAKE_FILE"; then
  ok "Example already registered in CMakeLists.txt — skipping."
else
  info "Registering $EXAMPLE_NAME in CMakeLists.txt..."
  # Insert before the foreach() that builds base_examples
  sed -i.bak \
    "s|set(base_examples|set(base_examples\n    ${EXAMPLE_NAME}|" \
    "$CMAKE_FILE"
  ok "Registered $EXAMPLE_NAME in CMakeLists.txt."
fi
echo

# ── 5. macOS / Anaconda guard ─────────────────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    warn "Anaconda environment detected. Temporarily unsetting CONDA vars to avoid"
    warn "CMake header conflicts between conda and Homebrew during ns-3 build."
    export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_EXE 2>/dev/null || true
  fi
fi

# ── 6. Configure & build ns-3 ────────────────────────────────────────────────
if [[ "$SKIP_NS3" == false ]]; then
  info "Configuring ns-3 (Release, examples enabled)..."
  cd "$NS3_DIR"

  cd "$NS3_DIR"
  ./ns3 configure --enable-examples --disable-tests \
    --build-profile release 2>&1 | tail -10

  echo
  info "Building ns-3 (this takes 5–20 min on first run)..."
  ./ns3 build lorawan_rl_adr 2>&1 | tail -20

  echo
  ok "ns-3 build complete."
  echo
fi

# ── 7. Python virtualenv ─────────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" == false ]]; then
  info "Setting up Python virtualenv (.venv/)..."
  cd "$REPO_DIR"

  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  ok "Python virtualenv ready at .venv/"
  echo
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Setup complete!${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo
echo -e "  Run a quick simulation:  ${CYAN}./run.sh --adr=ddqn${NC}"
echo -e "  See all options:         ${CYAN}./run.sh --help${NC}"
echo
