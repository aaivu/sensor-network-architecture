# LoRaWAN Pre-TX RSSI Simulation Setup

This setup provides a complete ns-3 simulation environment for LoRaWAN networks with pre-transmission RSSI sampling capability for machine learning dataset generation.

## Files Created

### 1. `setup_ns3_lorawan.sh`
- Complete installation script for ns-3 with LoRaWAN module on macOS
- Installs dependencies via Homebrew
- Clones ns-3-dev and lorawan module
- Builds and tests the environment

### 2. `pre-tx-rssi-example.cc`
- Complete C++ simulation example
- Implements pre-TX RSSI sampling by computing ambient interference
- Logs comprehensive packet data to CSV for ML training
- Configurable simulation parameters

## Quick Start

1. **Run the installation script:**
   ```bash
   cd /Users/shakir/Desktop/fyp/ns-3
   ./setup_ns3_lorawan.sh
   ```

2. **Copy the example to ns-3:**
   ```bash
   cp pre-tx-rssi-example.cc ns-3-dev/src/lorawan/examples/
   ```

3. **Build and run the example:**
   ```bash
   cd ns-3-dev
   ./ns3 configure --enable-tests --enable-examples --enable-modules lorawan
   ./ns3 build
   ./ns3 run pre-tx-rssi-example
   ```

## Simulation Features

### Pre-TX RSSI Sampling
- Computes total received power at transmitter location before transmission
- Sums contributions from all active interfering transmitters
- Uses ns-3's PropagationLossModel for realistic path loss calculations
- Includes configurable ambient noise floor

### Comprehensive Logging
The simulation outputs CSV files with the following columns:
- `timestamp_s`: Simulation time when packet was sent
- `devEui`: Device identifier
- `nodeId`: ns-3 node ID
- `fcnt`: Frame counter
- `tx_power_dBm`: Transmission power
- `sf`: Spreading factor
- `bw`: Bandwidth
- `cr`: Coding rate
- `tx_x`, `tx_y`, `tx_z`: Transmitter position
- `pre_tx_rssi_dBm`: **Pre-transmission RSSI measurement**
- `ambient_noise_dBm`: Background noise level
- `expected_gateway_rssi_dBm`: Received power at gateway
- `gateway_snr_dB`: Signal-to-noise ratio at gateway
- `gateway_id`: Gateway identifier
- `packet_received`: Success/failure flag (0/1)
- `collision_flag`: Collision detection flag
- `payload_bytes`: Packet size
- `channel`: Channel number

### Simulation Parameters

You can configure the simulation with command-line arguments:

```bash
./ns3 run "pre-tx-rssi-example --nDevices=20 --nGateways=2 --simulationTime=600 --appPeriod=60 --csvFile=custom_dataset.csv --radius=2000"
```

Parameters:
- `--nDevices`: Number of end devices (default: 10)
- `--nGateways`: Number of gateways (default: 1)  
- `--simulationTime`: Simulation duration in seconds (default: 300)
- `--appPeriod`: Packet transmission period in seconds (default: 30)
- `--csvFile`: Output CSV filename (default: lorawan_rssi_dataset.csv)
- `--radius`: Simulation area radius in meters (default: 1000)

## Key Implementation Details

### 1. Active Transmitter Tracking
```cpp
std::map<uint32_t, bool> activeTransmitters; // nodeId -> isTransmitting
```
Tracks which nodes are currently transmitting to compute interference.

### 2. Pre-TX RSSI Calculation
```cpp
double ComputePreTxRssi(Ptr<Node> txNode, Ptr<LoraChannel> channel, double ambientNoiseDbm)
```
- Iterates through all active transmitters
- Calculates received power using propagation loss model
- Sums all interference sources plus ambient noise
- Returns total RSSI in dBm

### 3. Power Conversion Utilities
```cpp
double DbmToW(double dbm)  // Convert dBm to Watts
double WToDbm(double w)    // Convert Watts to dBm
```

### 4. Packet Tracking
```cpp
std::map<std::pair<uint32_t, uint32_t>, PacketData> pendingPackets;
```
Maps (nodeId, frameCount) to packet metadata for correlation between transmission and reception events.

## Realistic Propagation Model

The simulation uses:
- **Log-distance path loss model** with exponent 3.76
- **Configurable shadowing** (can be added)
- **Multiple gateway support** for diversity
- **Realistic antenna heights** (1.5m for devices, 15m for gateways)

## Dataset Generation Tips

1. **Vary node density**: Run multiple scenarios with different device counts
2. **Adjust transmission power**: Use random TX powers between 2-14 dBm
3. **Multiple environments**: Change radius and propagation parameters
4. **Traffic patterns**: Modify packet intervals for different congestion levels
5. **Gateway placement**: Try single vs. multiple gateway configurations

## Validation Against Hardware

To validate the simulation against real hardware:

1. **Collect reference data** with 2-6 real LoRaWAN nodes
2. **Compare RSSI distributions** between simulation and hardware
3. **Calibrate noise floor** and propagation parameters
4. **Adjust for hardware-specific RSSI offsets** (e.g., SX127x chips)

## Next Steps

1. **Run multiple scenarios** with different parameters
2. **Merge CSV outputs** from multiple runs
3. **Preprocess data** for ML training (normalization, feature engineering)
4. **Train models** to predict packet loss from pre-TX RSSI
5. **Deploy models** on real LoRaWAN nodes for adaptive transmission

## Troubleshooting

If you encounter build errors:
1. Ensure ns-3 version matches the LoRaWAN module's NS3-VERSION file
2. Check that all dependencies are installed via Homebrew
3. Try `./ns3 clean` and rebuild
4. Verify the example is in `src/lorawan/examples/` directory

For runtime issues:
1. Check CSV file permissions and disk space
2. Reduce simulation time or device count for testing
3. Enable more detailed logging with `NS_LOG=PreTxRssiExample=level_all`
# sensor-network-architecture
