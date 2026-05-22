# LoRaWAN DDQN-PER Adaptive Data Rate

A research implementation of Dueling DDQN with Prioritized Experience Replay (PER) for Adaptive Data Rate (ADR) optimization in LoRaWAN networks, built on top of the [ns-3](https://www.nsnam.org/) network simulator.

This repository accompanies the FYP research "Coordinated Multi-Agent Reinforcement Learning for Adaptive Data Rate Optimization in Dense LoRaWAN Networks" (University of Moratuwa / University of Oulu).

---

## Repository Structure

```
lorawan-ddqn-adr/
├── sim/                            # ns-3 C++ simulation
│   ├── advanced-ddqn-per-adr-example.cc   # Main simulation file (DDQN / Classical / No-ADR)
│   └── README.md
├── analysis/                       # Python post-processing
│   └── analysis.py                 # PDR, energy, fairness, stability analysis
├── gui/                            # Qt desktop GUI
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── main.cpp
│   ├── RUNNING.md
│   └── ui/
│       ├── mainwindow.{cpp,h}
│       └── mapscene.{cpp,h}
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Quick Start

### 1 — Prerequisites

| Tool | Version tested | Install |
|------|---------------|---------|
| ns-3 | 3.40+ | see §2 below |
| GCC / Clang | ≥ 11 / ≥ 13 | system package manager |
| CMake | ≥ 3.16 | `brew install cmake` / `apt install cmake` |
| Qt | 5.15 or 6.x | `brew install qt` / `apt install qt6-base-dev` |
| Python | ≥ 3.9 | system / conda |
| pip packages | — | `pip install -r requirements.txt` |

---

### 2 — Build ns-3 with the LoRaWAN Module

The simulation requires ns-3 with the `lorawan` contrib module.

```bash
# Clone ns-3
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev

# Clone the lorawan contrib module into contrib/
git clone https://github.com/signetlabdei/lorawan src/lorawan

# Configure and build (Release mode for speed)
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

> **macOS note**: If you use Anaconda, unset conda env vars before building to avoid header conflicts:
> ```bash
> export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
> unset CONDA_PREFIX
> ./ns3 configure --enable-examples
> ```

---

### 3 — Place the Simulation File

Copy the example into ns-3's lorawan examples directory:

```bash
cp sim/advanced-ddqn-per-adr-example.cc \
   ns-3-dev/src/lorawan/examples/
```

Rebuild:

```bash
cd ns-3-dev
./ns3 build
```

---

### 4 — Run a Simulation

All three ADR modes are controlled by the `--adr` flag.

```bash
# From the ns-3-dev directory
cd ns-3-dev

# No ADR baseline
./ns3 run "advanced-ddqn-per-adr-example --adr=off --nDevices=20 --radius=300 --simTime=3600"

# Classical LoRaWAN ADR
./ns3 run "advanced-ddqn-per-adr-example --adr=on  --nDevices=20 --radius=300 --simTime=3600"

# DDQN-PER ADR
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --nDevices=20 --radius=300 --simTime=3600"
```

#### Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--adr` | `ddqn` | ADR mode: `off`, `on`, `ddqn`, `ppo` |
| `--nDevices` | `20` | Number of end devices |
| `--radius` | `300` | Deployment radius (m) |
| `--simTime` | `3600` | Simulation duration (s) |
| `--appPeriod` | `30` | Application packet period (s) |
| `--coordDivisor` | `3.0` | Coordination signal normalisation denominator |
| `--rewardSchedule` | `0` | Reward weight schedule (0=adaptive, 1=uniform, 2=outcome-only) |
| `--rngSeed` | `12345` | RNG seed for reproducible runs |

Output CSV files are written to the working directory (e.g. `advanced-ddqn-per-adr-results.csv`).

---

### 5 — Analyse Results

Install Python dependencies once:

```bash
pip install -r requirements.txt
```

Run the analysis script, pointing it at a results directory:

```bash
cd analysis
python analysis.py --results-dir ../ns-3-dev/
```

The script generates:
- Per-device PDR, energy, and SNR plots
- Fairness (Jain's index) comparison
- TX power and SF distribution bar charts
- Summary statistics CSV

---

### 6 — Qt GUI

The GUI provides a visual interface for configuring and launching simulations without touching the command line.

```bash
cd gui
./build.sh          # Detects Qt5 or Qt6 automatically
./build/lorawan-gui
```

See [gui/RUNNING.md](gui/RUNNING.md) for detailed build instructions and ns-3 integration notes.

---

## Algorithms Implemented

### DDQN-PER (`--adr=ddqn`)

- **Dueling DQN**: Separates state-value $V(s)$ and advantage $A(s,a)$ streams for more stable learning.
- **Prioritized Experience Replay**: Samples transitions proportional to TD-error, biased towards informative experiences.
- **Extended action space**: 48 actions covering SF (7–12), TX power (2–14 dBm), and channel selection (8 EU868 channels).
- **12-dimensional state**: 11 local RF features + per-channel device-count coordination signal $n_\text{same}/N$.
- **Congestion-adaptive reward**: Weights shift dynamically between outcome, interference, and stability objectives based on estimated congestion.

### Classical ADR (`--adr=on`)

Standard LoRaWAN ADR algorithm (LoRa Alliance TS002): SNR margin-based SF/TP adjustment with LinkADRReq MAC commands.

---

## Reproducibility

To exactly replicate a published result, fix the seed:

```bash
./ns3 run "advanced-ddqn-per-adr-example \
    --adr=ddqn --nDevices=40 --radius=100 \
    --simTime=10000 --rngSeed=42"
```

For a sweep over multiple seeds:

```bash
for seed in 42 123 456 789 1337; do
  ./ns3 run "advanced-ddqn-per-adr-example \
      --adr=ddqn --nDevices=40 --radius=100 \
      --simTime=10000 --rngSeed=$seed" \
      2>&1 | tee "results_seed${seed}.log"
done
```

---

## Citation

If you use this code in your work, please cite:

```bibtex
@article{perera2026marl_adr,
  title   = {Coordinated Multi-Agent Reinforcement Learning for Adaptive Data Rate Optimization in Dense {LoRaWAN} Networks},
  author  = {Perera, Nipuna and others},
  journal = {IEEE Access},
  year    = {2026}
}
```

---

## License

This project is released for academic research use. The ns-3 simulator and the lorawan module are separately licensed under GPL-2.0 and MIT respectively — please consult their repositories.
