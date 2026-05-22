# sim — ns-3 Simulation Module

This directory contains the self-contained ns-3 C++ example that implements three ADR strategies in a LoRaWAN network.

## File

| File | Description |
|------|-------------|
| `advanced-ddqn-per-adr-example.cc` | Combined simulation: No-ADR / Classical ADR / DDQN-PER ADR / PPO ADR |

## Installation

1. Build ns-3 with the lorawan module (see top-level README §2).
2. Copy the file into the examples directory:
   ```bash
   cp advanced-ddqn-per-adr-example.cc \
      <ns-3-dev>/src/lorawan/examples/
   ```
3. Rebuild:
   ```bash
   cd <ns-3-dev>
   ./ns3 build
   ```

## Running

```bash
# DDQN-PER — 20 devices, 300 m radius, 1 hour
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --nDevices=20 --radius=300 --simTime=3600"

# Classical ADR comparison
./ns3 run "advanced-ddqn-per-adr-example --adr=on  --nDevices=20 --radius=300 --simTime=3600"

# Baseline: no ADR
./ns3 run "advanced-ddqn-per-adr-example --adr=off --nDevices=20 --radius=300 --simTime=3600"
```

## Output

The simulation writes a CSV file (`advanced-ddqn-per-adr-results.csv` by default) with one row per packet transmission. Key columns:

| Column | Description |
|--------|-------------|
| `node_id` | End-device identifier |
| `timestamp` | Simulation time (s) |
| `sf` | Spreading factor used |
| `tx_power_dBm` | Transmit power |
| `channel` | EU868 channel index (0–7) |
| `packet_received` | 1 = gateway received, 0 = lost |
| `rssi_dbm` | Received signal strength at gateway |
| `snr_db` | Signal-to-noise ratio at gateway |
| `energy_consumed_mJ` | Energy for this transmission (mJ) |
| `pre_tx_rssi` | Channel RSSI measured before transmitting |

Pass `--csvFile=myfile.csv` to override the output filename.

## Ablation / Sensitivity Flags

```bash
# Sweep coordination-signal denominator
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --coordDivisor=5"

# Fixed uniform reward (no congestion adaptation)
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --rewardSchedule=1"

# Reproducible run with explicit seed
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --rngSeed=42"
```
