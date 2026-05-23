# sim — ns-3 Simulation Module

This directory contains the C++ simulation source for **LoRaWAN Multi-Algorithm RL ADR**.
The code is written as a set of focused header files included from a single translation unit, so there are no multiple-definition linker errors and the build system sees exactly one `.cc` file.

---

## File structure

```
sim/
├── lorawan_rl_adr.cc       ← entry point: #includes, main(), RunSimulation()
└── include/                ← 15 header files (one concern per file)
    ├── simple_ddqn.h
    ├── device_history.h
    ├── dueling_ddqn.h
    ├── reward.h
    ├── ppo_agent.h
    ├── marl_agent.h
    ├── marl_ppo_agent.h
    ├── mappo_gat_agent.h
    ├── globals.h
    ├── environment.h
    ├── sim_core.h
    ├── update_ppo.h
    ├── update_marl.h
    ├── update_marl_ppo.h
    └── update_mappo_gat.h
```

### Header reference

| Header | Original lines | Contents |
|--------|---------------|----------|
| `simple_ddqn.h` | 65–574 | `Experience`, `SimpleQNetwork`, `PrioritizedReplayBuffer`, `DDQNPERADRAgent` |
| `device_history.h` | 575–814 | `DeviceHistory`, `CollisionIndicators`, `ChannelSelector`, `channelSelector` global |
| `dueling_ddqn.h` | 815–1294 | `ExtendedActionSpace` (48 actions), `DuelingQNetwork`, `OptimizedDDQNAgent` |
| `reward.h` | 1295–1722 | `estimateCongestionLevel`, `getMandatorySFCeiling`, `applySFCeiling`, `calculateCongestionAdaptiveReward`, `calculateInterferenceAwareReward`, ablation globals |
| `ppo_agent.h` | 1723–2025 | `PolicyNetwork`, `ValueNetwork`, `PPOAgent`, `ppoAgents` map |
| `marl_agent.h` | 2026–2212 | `MARLAgent` (IQL + coordination signal), `marlAgents`, `deviceChannels` |
| `marl_ppo_agent.h` | 2213–2549 | `MARLPPOActor`, `MARLPPOCritic`, `MARLPPOAgent`, `marlPpoAgents` |
| `mappo_gat_agent.h` | 2550–2996 | `GATLayer`, `MAPPOGATActor`, `MAPPOCentralizedCritic`, `MAPPOGATAgent`, `mappoGatAgents`, `sharedGATLayer`, `sharedMAPPOCritic` |
| `globals.h` | 2997–3090 | `activeTransmitters`, `nodeTxPowers`, `nodePositions`, `UrbanObstacle`, `PacketRecord`, `DeviceMetrics`, `ddqnAgents`, `ADRMethod` enum, forward declarations |
| `environment.h` | 3091–3385 | `SetupUrbanEnvironment`, `RssiSample`, `SetupWiFiInterferers`, `SetupLTEMobileInterferers`, `InitializeHardwareVariability`, `SamplePreTxRssi` |
| `sim_core.h` | 3386–4222 | `OnTxPowerChange`, `OnDataRateChange`, `CalculateEnergyConsumption`, `ApplyADRDecision`, `OnTransmissionStart`, `RecordTransmissionParameters`, `CheckPacketLoss`, `OnPacketReceived`, `UpdateDeviceMetrics`, `UpdateDDQNADR`, `UpdateOptimizedDDQN` |
| `update_ppo.h` | 4223–4521 | `UpdateClassicalADR`, `UpdatePPOADR` |
| `update_marl.h` | 4522–4984 | `UpdateMARLADR` |
| `update_marl_ppo.h` | 4985–5398 | `UpdateMARLPPOADR` |
| `update_mappo_gat.h` | 5399–5687 | `UpdateMAPPOGATADR` |

`lorawan_rl_adr.cc` contains lines 1–64 (ns-3 includes, `using namespace`, `NS_LOG_COMPONENT_DEFINE`) and lines 5688–6407 (`main()`, `RunSimulation()`, output helpers), plus the 15 `#include "include/..."` directives.

---

## Automated setup (recommended)

Run from the repo root:

```bash
./setup.sh
```

This copies the files to the right place inside `ns3/ns-3-dev/`, registers the example in CMakeLists.txt, and builds ns-3. See the [top-level README](../README.md) for full details.

---

## Manual installation

If you prefer to manage ns-3 yourself:

```bash
# 1. Copy simulation files into ns-3
cp lorawan_rl_adr.cc    <ns-3-dev>/src/lorawan/examples/
cp -r include/          <ns-3-dev>/src/lorawan/examples/

# 2. Register the example in CMakeLists.txt
#    Add lorawan_rl_adr to the set(base_examples ...) list in:
#    <ns-3-dev>/src/lorawan/examples/CMakeLists.txt

# 3. Rebuild
cd <ns-3-dev>
./ns3 build lorawan_rl_adr
```

---

## Running

All commands assume `cd ns3/ns-3-dev` or `cd <ns-3-dev>` first.

```bash
# All 7 algorithms in sequence (recommended for comparison)
./ns3 run "lorawan_rl_adr --adr=all --nDevices=20 --simTime=3600"

# DDQN-PER
./ns3 run "lorawan_rl_adr --adr=ddqn --nDevices=20 --simTime=3600"

# PPO
./ns3 run "lorawan_rl_adr --adr=ppo --nDevices=20 --simTime=3600"

# MARL (IQL)
./ns3 run "lorawan_rl_adr --adr=marl --nDevices=20 --simTime=3600"

# MARL-PPO
./ns3 run "lorawan_rl_adr --adr=marl_ppo --nDevices=20 --simTime=3600"

# MAPPO-GAT (CTDE)
./ns3 run "lorawan_rl_adr --adr=mappo_gat --nDevices=20 --simTime=3600"

# Classical ADR
./ns3 run "lorawan_rl_adr --adr=on --nDevices=20 --simTime=3600"

# No ADR baseline
./ns3 run "lorawan_rl_adr --adr=off --nDevices=20 --simTime=3600"
```

Or use the repo-level wrapper from any directory:

```bash
./run.sh --adr=ddqn --nDevices=20 --simTime=3600
```

---

## Simulation parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--adr` | `ddqn` | Algorithm (see above, or `all`) |
| `--nDevices` | `20` | Number of end-devices |
| `--radius` | `500` | Deployment radius (m) |
| `--simTime` | `3600` | Simulation duration (s) |
| `--rngSeed` | `12345` | RNG seed |
| `--csvFile` | `lorawan_rl_adr_results.csv` | Output filename |
| `--coordDivisor` | `3.0` | Normalisation divisor for coordination signal |
| `--rewardSchedule` | `0` | `0` = adaptive · `1` = fixed · `2` = outcome-only |

---

## Output CSV columns

| Column | Description |
|--------|-------------|
| `node_id` | End-device identifier |
| `timestamp` | Simulation time (s) |
| `sf` | Spreading factor used |
| `tx_power_dBm` | Transmit power |
| `channel` | EU868 channel index (0–7) |
| `packet_received` | `1` = received at gateway · `0` = lost |
| `rssi_dbm` | RSSI at gateway |
| `snr_db` | SNR at gateway |
| `energy_consumed_mJ` | Energy for this transmission |
| `pre_tx_rssi` | Channel RSSI sampled before transmitting |
| `adr_method` | Algorithm that produced this row |
