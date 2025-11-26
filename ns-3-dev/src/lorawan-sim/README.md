# LoRaWAN Simulation Module (lorawan-sim)

A clean, modular ns-3 simulation framework for LoRaWAN networks with advanced ADR capabilities including DDQN-PER reinforcement learning.

This module refactors the monolithic `advanced-ddqn-per-adr-example.cc` into focused, maintainable components.

## Overview

This module provides:

- **DDQN-PER ADR**: Deep Q-Network with Prioritized Experience Replay for adaptive data rate control
- **Classical ADR**: Standard LoRaWAN ADR implementation via ns-3's AdrComponent
- **Realistic Environment**: Urban obstacles, WiFi interference, hardware variability
- **Pre-TX RSSI Sampling**: Real-time interference measurement at transmitter location
- **Comprehensive Logging**: Per-device CSV datasets for ML training

## Module Structure

```
lorawan-sim/
├── model/
│   ├── adr-ddqn.h/.cc           # DDQN-PER ADR agent
│   ├── environment.h/.cc         # Environment modeling
│   ├── packet-recorder.h/.cc     # Packet tracking and CSV output
│   └── simulation-runner.h/.cc   # Main simulation orchestration
├── helper/
│   └── (future helper classes)
├── examples/
│   └── lorawan-sim-example.cc    # Example usage
├── CMakeLists.txt                # ns-3 build configuration
└── README.md                     
```

## Key Components

### 1. ADR DDQN (`adr-ddqn.h/cc`)

Reinforcement learning agent for adaptive data rate control:

- **Experience**: Replay buffer structure
- **SimpleQNetwork**: 3-layer neural network (6→128→64→15)
- **PrioritizedReplayBuffer**: Prioritized experience replay (10k buffer)
- **DDQNPERADRAgent**: Main RL agent with epsilon-greedy exploration

**Features**:
- 15-action space (SF 7-12, TX Power 2-17 dBm)
- Advanced reward function (PDR, energy, SNR-based)
- Target network for stable learning
- Adaptive exploration (epsilon decay)

### 2. Environment Manager (`environment.h/cc`)

Handles realistic environment modeling:

- **UrbanObstacle**: Virtual building structures
- **RssiSample**: Pre-transmission RSSI measurements
- **EnvironmentManager**: Coordinates all environmental effects

**Capabilities**:
- Urban obstacle generation (configurable density)
- WiFi interferer modeling
- Hardware variability (antenna gain, noise figure)
- Real-time RSSI sampling at transmitter location

### 3. Packet Recorder (`packet-recorder.h/cc`)

Tracks all transmissions and generates datasets:

- **PacketRecord**: Complete transmission/reception metadata
- **DeviceMetrics**: Per-device performance tracking (PDR, energy)
- **PacketRecorder**: Manages packet collection and CSV export

**Output Format**:
- Per-device CSV files with 18 columns
- Timestamp, power, SF, BW, position, RSSI, SNR, success, energy, distance
- Suitable for ML training and analysis

### 4. Simulation Runner (`simulation-runner.h/cc`)

Main simulation orchestrator (replaces monolithic main):

- **SimulationRunner**: Sets up nodes, channels, devices, ADR
- Handles all ns-3 API calls
- Manages simulation lifecycle
- Coordinates ADR callbacks

**ADR Modes**:
- `ADRMethod::OFF` - No ADR (fixed parameters)
- `ADRMethod::ON` - Classical LoRaWAN ADR (ns-3 AdrComponent)
- `ADRMethod::DDQN` - DDQN-PER reinforcement learning ADR

## Quick Start

### Build the Module

```bash
cd /path/to/ns-3-dev
./ns3 configure --enable-examples
./ns3 build lorawan-sim
```

### Run Your First Simulation

```bash
# Run with DDQN-PER ADR
./ns3 run "lorawan-sim-example --adr=ddqn --nDevices=10"

# Compare all ADR methods (NO ADR, Classical, DDQN)
./ns3 run "lorawan-sim-example --adr=all --nDevices=20 --simulationTime=60000 --appPeriod=60 --radius=1500"
```

### Check Output

Generated CSV datasets will be in:
```bash
ls lorawan_datasets/
# ddqn_adr_lorawan_sim_dataset_device_0_dataset.csv
# ddqn_adr_lorawan_sim_dataset_device_1_dataset.csv
# ...
```

## Usage Examples

### Basic Commands

```bash
# Run with NO ADR (fixed parameters)
./ns3 run "lorawan-sim-example --adr=off"

# Run with Classical ADR (ns-3 AdrComponent)
./ns3 run "lorawan-sim-example --adr=on"

# Run with DDQN-PER ADR (reinforcement learning)
./ns3 run "lorawan-sim-example --adr=ddqn"

# Run all three modes sequentially for comparison
./ns3 run "lorawan-sim-example --adr=all"
```

### Replicate Original Simulation

To get the same results as the original `advanced-ddqn-per-adr-example`:

```bash
# Original command:
# ./ns3 run "advanced-ddqn-per-adr-example --adr=all --nDevices=20 --simulationTime=60000 --appPeriod=60 --radius=1500"

# Equivalent refactored command:
./ns3 run "lorawan-sim-example --adr=all --nDevices=20 --simulationTime=60000 --appPeriod=60 --radius=1500"
```

**The output will be identical!** Same CSV format, same packet counts, same PDR values.

### Custom Scenarios

```bash
./ns3 run "lorawan-sim-example \
    --adr=ddqn \
    --nDevices=50 \
    --simulationTime=600 \
    --appPeriod=20 \
    --radius=2000 \
    --wifiInterferers=20 \
    --environmental=true \
    --csvFile=my_dataset.csv"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--adr` | off | ADR mode: off, on, ddqn, all |
| `--nDevices` | 10 | Number of end devices |
| `--simulationTime` | 300 | Simulation duration (seconds) |
| `--appPeriod` | 30 | Packet transmission interval (seconds) |
| `--radius` | 1000 | Deployment radius (meters) |
| `--wifiInterferers` | 12 | Number of WiFi interferers |
| `--environmental` | true | Enable environmental modeling |
| `--csvFile` | lorawan_sim_dataset.csv | Output CSV filename |

## Using as a Library

### In Your Own ns-3 Module/Example

```cpp
#include "ns3/simulation-runner.h"

using namespace ns3;
using namespace lorawan;

int main(int argc, char* argv[]) {
    // Parse command line
    CommandLine cmd;
    uint32_t nDevices = 10;
    cmd.AddValue("nDevices", "Number of devices", nDevices);
    cmd.Parse(argc, argv);
    
    // Create and run simulation
    SimulationRunner sim(
        nDevices,         // Number of end devices
        300.0,            // Simulation time (seconds)
        30.0,             // App period (seconds)
        1000.0,           // Deployment radius (meters)
        "output.csv",     // CSV output filename
        ADRMethod::DDQN,  // ADR method
        12,               // WiFi interferers
        true              // Enable environmental modeling
    );
    
    sim.Run();
    
    // Access results
    auto& recorder = sim.GetRecorder();
    const auto& packets = recorder.GetCompletedPackets();
    std::cout << "Total packets: " << packets.size() << std::endl;
    
    return 0;
}
```

### In External Project (Qt, Web Server, etc.)

See `INTEGRATION.md` for detailed instructions on linking this module to external projects.

## Output

### CSV Datasets

Generated in `lorawan_datasets/` folder:

```
lorawan_datasets/
├── no_adr_lorawan_sim_dataset_device_0_dataset.csv
├── no_adr_lorawan_sim_dataset_device_1_dataset.csv
├── ...
└── ddqn_adr_lorawan_sim_dataset_device_9_dataset.csv
```

### CSV Format

```csv
timestamp,devEUI,fcnt,tx_power_dBm,sf,bw,cr,tx_x,tx_y,tx_z,
pre_tx_rssi_dBm,ambient_noise_dBm,expected_rx_rssi_dBm,snr_db,
gateway_id,packet_received,energy_consumed_mJ,distance_m
```

### Console Output

```
=== LoRaWAN Simulation Module ===
Environmental modeling: ENABLED
  - Urban obstacle modeling
  - WiFi interference modeling (12 nodes)
  - Hardware variability modeling

▶ Running simulation for DDQN-PER ADR...
Urban environment: Created 156 virtual obstacles
Interference modeling: Created 12 WiFi interferers
Hardware modeling: Added manufacturing tolerances for 10 devices
📋 Mapped Node ID 1 → Device Index 0
...
🤖 DDQN ADR: Device 0 SF=10, TP=8.0 dBm
📊 Device 0 metrics: PDR=0.850, Energy/pkt=0.325 mJ
...

=== REALISTIC ns-3 LoRaWAN SIMULATION RESULTS ===
Total packets transmitted: 100
Successfully received: 78 (78.0%)
Per-device datasets saved to: lorawan_datasets/ folder
```

## Extending the Module

### Add New ADR Algorithm

```cpp
// In simulation-runner.cc
void SimulationRunner::ApplyADRDecision(uint32_t nodeId) {
    if (m_adrMethod == ADRMethod::CUSTOM) {
        // Your custom ADR logic here
        uint8_t newSF = CalculateOptimalSF(nodeId);
        double newTP = CalculateOptimalTP(nodeId);
        ApplyParameters(nodeId, newSF, newTP);
    }
}
```

### Add New Environment Feature

```cpp
// In environment.cc
void EnvironmentManager::SetupRainfallAttenuation(double intensity) {
    m_rainfallIntensity = intensity;
    // Apply rainfall attenuation model
}
```

### Custom Packet Analysis

```cpp
// Access packet data
auto& recorder = sim.GetRecorder();
const auto& packets = recorder.GetCompletedPackets();

for (const auto& pkt : packets) {
    if (pkt.sf == 12 && !pkt.packetReceived) {
        std::cout << "SF12 packet lost at " 
                  << pkt.timestamp << "s\n";
    }
}
```

## Validation & Testing

### Verify Against Original

The refactored module produces **identical results** to the original:

```bash
# 1. Run original implementation
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --nDevices=10 --simulationTime=600"

# 2. Run refactored module
./ns3 run "lorawan-sim-example --adr=ddqn --nDevices=10 --simulationTime=600"

# 3. Compare outputs (should be identical)
diff lorawan_datasets/ddqn_adr_advanced_pre_tx_rssi_dataset_device_0_dataset.csv \
     lorawan_datasets/ddqn_adr_lorawan_sim_dataset_device_0_dataset.csv
```

### Quick Build & Test Script

Use the provided test script:

```bash
cd src/lorawan-sim
./build-and-test.sh
```

This will:
1. Build the module
2. Run quick tests (NO ADR, Classical ADR, DDQN ADR)
3. Verify module structure
4. Report results

## Troubleshooting

### Build Errors

**Error**: `fatal error: 'simulation-runner.h' file not found`
```bash
# Solution: Use ns3/ prefix in includes
#include "ns3/simulation-runner.h"  // ✅ Correct
#include "simulation-runner.h"       // ❌ Wrong
```

**Error**: `Couldn't find the specified program: lorawan-sim-example`
```bash
# Solution: Rebuild with examples enabled
./ns3 configure --enable-examples
./ns3 build
```

### Runtime Issues

**Issue**: No CSV files generated
- Check you're in `ns-3-dev/` directory when running
- CSV files are created in `ns-3-dev/lorawan_datasets/`
- Ensure simulation completes without errors

**Issue**: Different results than original
- Verify same random seed (RngSeedManager)
- Check all parameters match exactly
- Ensure same ns-3 version

## Dependencies

- **ns-3**: Core simulator (tested with ns-3.41+, works with 3.35+)
- **lorawan module**: ns-3 LoRaWAN module (included in ns-3-dev)
- **C++17 or higher**: Standard library features
- **CMake 3.10+**: Build system (via ns-3)

## Module Files Reference

```
lorawan-sim/
├── CMakeLists.txt                 # ns-3 build config (library + example)
├── README.md                      # This file
├── INTEGRATION.md                 # External integration guide (Qt, etc.)
├── REFACTORING_SUMMARY.md         # Detailed refactoring overview
├── build-and-test.sh             # Quick test script
│
├── model/                         # Core library files
│   ├── adr-ddqn.h                # DDQN-PER classes
│   ├── adr-ddqn.cc               # RL agent implementation (480 lines)
│   ├── environment.h             # Environment structures
│   ├── environment.cc            # Environment modeling (318 lines)
│   ├── packet-recorder.h         # Packet tracking structures
│   ├── packet-recorder.cc        # CSV generation (180 lines)
│   ├── simulation-runner.h       # Main simulation class
│   └── simulation-runner.cc      # Orchestration logic (544 lines)
│
├── examples/                      # Example programs
│   └── lorawan-sim-example.cc    # Main example (90 lines)
│
└── helper/                        # Future helper classes
    └── (empty for now)
```

## License

This module follows the same license as the ns-3 LoRaWAN module and ns-3 core.

## Related Documentation

- **INTEGRATION.md**: Qt integration, external linking, CMake examples
- **REFACTORING_SUMMARY.md**: Before/after comparison, benefits, migration guide
- **ns-3-dev/src/lorawan/**: Original LoRaWAN module documentation

## Acknowledgments

Refactored from `advanced-ddqn-per-adr-example.cc` (1,800+ lines) into modular components while preserving identical functionality.
