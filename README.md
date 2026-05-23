# LoRaWAN Multi-Algorithm RL ADR

> **Coordinated Multi-Agent Reinforcement Learning for Adaptive Data Rate Optimisation in Dense LoRaWAN Networks**

A self-contained research codebase that implements and compares **seven ADR strategies** — from the classical LoRaWAN 1.0.4 algorithm to a Centralised-Training / Decentralised-Execution (CTDE) multi-agent system with Graph Attention Networks — inside a realistic urban ns-3 simulation with Wi-Fi and LTE interference.

This repository accompanies the paper *"Coordinated Multi-Agent Reinforcement Learning for Adaptive Data Rate Optimisation in Dense LoRaWAN Networks"*.

---

## Algorithms at a glance

| # | Mode flag | Algorithm | Notes |
|---|-----------|-----------|-------|
| 0 | `off`        | **No ADR** — static SF7 / 14 dBm | baseline |
| 1 | `on`         | **Classical ADR** (LoRaWAN 1.0.4) | SNR-margin step rule |
| 2 | `ddqn`       | **DDQN-PER** — Dueling DQN + Prioritised Experience Replay | single-agent |
| 3 | `ppo`        | **PPO** — Proximal Policy Optimisation | single-agent |
| 4 | `marl`       | **MARL** — Independent Q-Learning + coordination signal | multi-agent |
| 5 | `marl_ppo`   | **MARL-PPO** — Independent PPO with coordination | multi-agent |
| 6 | `mappo_gat`  | **MAPPO-GAT** — Multi-Agent PPO + Graph Attention (CTDE) | multi-agent CTDE |

---

## Repository layout

```
lorawan-rl-adr/
├── ns3/                            ← git submodule: ns-3.45 + lorawan module
│   ├── ns-3-dev/                   ← ns-3 source tree (built in place)
│   │   └── src/lorawan/examples/   ← sim files are copied here by setup.sh
│   └── lorawan-qt-gui/             ← Qt GUI source (within submodule)
├── sim/                            ← our C++ simulation source (modular headers)
│   ├── lorawan_rl_adr.cc           ← entry point: main() + RunSimulation()
│   └── include/                    ← 15 header files (one concern each)
├── analysis/
│   ├── analysis.py                 ← PDR / energy / fairness / stability plots
│   └── README.md
├── gui/                            ← Qt GUI source copy (standalone build)
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── main.cpp
│   └── ui/  mainwindow.{cpp,h}  mapscene.{cpp,h}
├── setup.sh                        ← one-shot automated setup script
├── run.sh                          ← convenience simulation launcher
├── requirements.txt                ← Python dependencies
└── README.md
```

---

## Quick start

### Step 1 — Clone with submodule

```bash
git clone --recursive https://github.com/<your-org>/lorawan-rl-adr.git
cd lorawan-rl-adr
```

> Already cloned without `--recursive`?
> ```bash
> git submodule update --init --recursive
> ```

---

### Step 2 — Install system dependencies

| Tool | Min version | macOS (Homebrew) | Ubuntu / Debian |
|------|------------|-----------------|-----------------|
| GCC **or** Clang | GCC ≥ 11 / Clang ≥ 13 | `brew install gcc` | `apt install g++-12` |
| CMake | 3.16 | `brew install cmake` | `apt install cmake` |
| Python | 3.9 | `brew install python` | `apt install python3 python3-venv` |
| Qt | 5.15 **or** 6.x | `brew install qt` | `apt install qt6-base-dev` |
| ninja *(faster builds)* | any | `brew install ninja` | `apt install ninja-build` |

> **macOS + Anaconda**: `setup.sh` automatically clears conda env vars before the ns-3 build to prevent CMake header conflicts.

---

### Step 3 — Run automated setup

```bash
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` performs these steps automatically:

| Step | What happens |
|------|-------------|
| 1 | `git submodule update --init --recursive` — fetches `ns3/` (~300 MB, **5–10 min**) |
| 2 | Copies `sim/lorawan_rl_adr.cc` + `sim/include/` → `ns3/ns-3-dev/src/lorawan/examples/` |
| 3 | Registers `lorawan_rl_adr` in the lorawan `CMakeLists.txt` |
| 4 | `./ns3 configure --enable-examples --build-profile release` |
| 5 | `./ns3 build lorawan_rl_adr` (**5–20 min** first time; incremental afterwards) |
| 6 | Creates `.venv/` and installs `requirements.txt` |

Skip individual steps: `./setup.sh --skip-ns3` or `./setup.sh --skip-python`

---

### Step 4 — Run a simulation

```bash
# Convenience wrapper (recommended)
./run.sh --adr=ddqn --nDevices=20 --simTime=3600

# All 7 algorithms in one go
./run.sh --adr=all --nDevices=20 --simTime=3600

# Equivalent direct ns-3 call
cd ns3/ns-3-dev
./ns3 run "lorawan_rl_adr --adr=mappo_gat --nDevices=50 --simTime=7200"
```

---

### Step 5 — Analyse results

```bash
source .venv/bin/activate
python analysis/analysis.py --csv lorawan_rl_adr_results.csv
```

---

## Simulation parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--adr` | `ddqn` | ADR algorithm (see table above, or `all`) |
| `--nDevices` | `20` | Number of LoRa end-devices |
| `--radius` | `500` | Network radius (m) |
| `--simTime` | `3600` | Simulation duration (s) |
| `--rngSeed` | `12345` | RNG seed for reproducibility |
| `--csvFile` | `lorawan_rl_adr_results.csv` | Output CSV filename |
| `--coordDivisor` | `3.0` | Normalisation factor for coordination signal |
| `--rewardSchedule` | `0` | `0` = adaptive · `1` = fixed · `2` = outcome-only |

---

## Qt GUI

A desktop visualiser lets you inspect device positions, SF assignments, and per-device metrics.

```bash
cd gui
./build.sh            # auto-detects Qt5 or Qt6
./build/lorawan-gui
```

See [gui/README.md](gui/README.md) for full build options and troubleshooting.

---

## Simulation source layout

The C++ code is split into 15 focused headers included from a single `.cc` (single-TU model — no linker issues).

| Header | Contents |
|--------|----------|
| `simple_ddqn.h` | `Experience`, `SimpleQNetwork`, `PrioritizedReplayBuffer`, `DDQNPERADRAgent` |
| `device_history.h` | `DeviceHistory`, `CollisionIndicators`, `ChannelSelector` |
| `dueling_ddqn.h` | `ExtendedActionSpace` (48 actions), `DuelingQNetwork`, `OptimizedDDQNAgent` |
| `reward.h` | Congestion estimation, SF ceiling, reward functions, ablation globals |
| `ppo_agent.h` | `PolicyNetwork`, `ValueNetwork`, `PPOAgent` |
| `marl_agent.h` | `MARLAgent` (IQL + coordination signal) |
| `marl_ppo_agent.h` | `MARLPPOActor`, `MARLPPOCritic`, `MARLPPOAgent` |
| `mappo_gat_agent.h` | `GATLayer`, `MAPPOGATActor`, `MAPPOCentralizedCritic`, `MAPPOGATAgent` |
| `globals.h` | Global state maps, `ADRMethod` enum, `UrbanObstacle`, `PacketRecord`, `DeviceMetrics` |
| `environment.h` | `SetupUrbanEnvironment`, Wi-Fi/LTE interferers, `SamplePreTxRssi` |
| `sim_core.h` | Transmission callbacks, `ApplyADRDecision`, `UpdateDDQNADR`, `UpdateOptimizedDDQN` |
| `update_ppo.h` | `UpdateClassicalADR`, `UpdatePPOADR` |
| `update_marl.h` | `UpdateMARLADR` |
| `update_marl_ppo.h` | `UpdateMARLPPOADR` |
| `update_mappo_gat.h` | `UpdateMAPPOGATADR` |

Full details: [sim/README.md](sim/README.md)

---

## Reproducibility

```bash
# Fix the seed to reproduce a published result
./run.sh --adr=ddqn --nDevices=40 --simTime=10000 --rngSeed=42

# Sweep over seeds
for seed in 42 123 456 789 1337; do
  ./run.sh --adr=all --nDevices=40 --simTime=10000 --rngSeed=$seed
done
```

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@article{perera2026marl_adr,
  title   = {Coordinated Multi-Agent Reinforcement Learning for Adaptive Data Rate
             Optimisation in Dense {LoRaWAN} Networks},
  author  = {Perera, Nipuna and others},
  journal = {IEEE Access},
  year    = {2026}
}
```

---

## License

This project is released for academic research use.
The ns-3 simulator is licensed under **GPL-2.0**.
The lorawan module is licensed under **MIT**.
Please consult their respective repositories for details.
