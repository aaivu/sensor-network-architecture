# DDQN-PER ADR Methodology for LoRaWAN

Based on "Enhanced Reinforcement Learning Algorithm Based-Transmission Parameter Selection for Optimization of Energy Consumption and Packet Delivery Ratio in LoRa Wireless Networks" (Zholamanov et al., 2024)

## Overview

This implementation combines:
1. **Pre-TX RSSI Sampling**: Real-time interference measurement at transmitter location
2. **DDQN-PER Algorithm**: Double Deep Q-Network with Prioritized Experience Replay
3. **Multi-objective Optimization**: Balancing Packet Delivery Ratio (PDR) and Energy Consumption (EC)

## Key Components

### 1. State Representation
- **SNR**: Signal-to-Noise Ratio at receiver
- **SF**: Current Spreading Factor (7-12)
- **TP**: Current Transmission Power (2-14 dBm)
- **PreTxRSSI**: Pre-transmission RSSI measurement (our enhancement)

### 2. Action Space
- INCREASE_SF: Increase spreading factor by 1
- DECREASE_SF: Decrease spreading factor by 1
- INCREASE_TP: Increase transmission power by 2 dBm
- DECREASE_TP: Decrease transmission power by 2 dBm
- NO_CHANGE: Maintain current parameters

### 3. Reward Function
```
R = w1 * PDR_reward + w2 * Energy_penalty
```
Where:
- PDR_reward = +1 for successful transmission, -1 for failure
- Energy_penalty = -(SF-7)*0.1 - (TP-2)*0.05
- w1 = 1.0 (PDR weight)
- w2 = 0.3 (Energy weight)

### 4. DDQN-PER Algorithm

#### Double Deep Q-Network (DDQN)
- **Main Network**: Used for action selection
- **Target Network**: Used for Q-value estimation
- **Benefit**: Prevents Q-value overestimation

#### Prioritized Experience Replay (PER)
- **Priority**: Based on TD-error magnitude
- **Sampling**: Higher priority experiences sampled more frequently
- **Benefit**: Faster learning from important experiences

## Implementation Architecture

### Neural Network Structure
```
Input Layer (4 neurons) -> Hidden Layer 1 (64 neurons) -> Hidden Layer 2 (32 neurons) -> Output Layer (5 neurons)
```

### Training Process
1. Observe state s_t = [SNR, SF, TP, PreTxRSSI]
2. Select action a_t using ε-greedy policy
3. Execute action and observe reward r_t and next state s_t+1
4. Store experience (s_t, a_t, r_t, s_t+1) in PER buffer
5. Sample prioritized batch from buffer
6. Compute TD-error and update Q-network
7. Periodically update target network

### Hyperparameters
- Learning Rate: 0.001
- Discount Factor (γ): 0.95
- Exploration Rate (ε): 1.0 → 0.01 (decay: 0.995)
- Batch Size: 32
- Target Network Update Frequency: 100 steps
- Replay Buffer Size: 10,000 experiences

## Key Advantages

### 1. Enhanced State Information
- Traditional ADR only uses SNR
- Our approach adds Pre-TX RSSI for better interference awareness

### 2. Faster Convergence
- PER prioritizes important learning experiences
- Reaches optimal performance in ~24 hours vs several days for traditional ADR

### 3. Multi-objective Optimization
- Balances PDR maximization with energy minimization
- Achieved 17.2% higher PDR than traditional ADR methods

### 4. Scalability
- Tested with 10-1000 nodes
- Maintains performance in dense deployments

## Performance Metrics

### Primary Metrics
- **PDR**: Packet Delivery Ratio = Packets_received / Packets_sent
- **Energy per Packet**: Total_energy / Packets_received

### Expected Results (from paper)
- PDR improvement: +17.2% vs traditional ADR
- Energy consumption: Stable ~0.18-0.19 mJ per node
- Convergence time: <24 hours

## Simulation Environment

### Network Setup
- Topology: Star (1 gateway, multiple end devices)
- Frequency: 868 MHz (EU band)
- Bandwidth: 125 kHz
- Coding Rate: 4/5

### Environmental Modeling
- Urban obstacles and shadowing
- WiFi interference modeling
- Hardware manufacturing tolerances
- Temporal dynamics and fading

## Files Generated

### Datasets
- Per-device CSV files with transmission parameters
- Pre-TX RSSI measurements for each transmission
- Reward values and action selections
- Success/failure outcomes

### Metrics
- PDR statistics
- Energy consumption analysis
- RL convergence metrics
- Parameter adaptation history

## Usage

```bash
# Run baseline simulation (current implementation)
./ns3 run "advanced-pre-tx-rssi-example --nDevices=20 --simulationTime=600"

# Future: Run RL-enhanced simulation
./ns3 run "rl-adr-example --nDevices=20 --simulationTime=600 --rlMode=ddqn-per"
```

## References

Zholamanov, B., et al. (2024). "Enhanced Reinforcement Learning Algorithm Based-Transmission Parameter Selection for Optimization of Energy Consumption and Packet Delivery Ratio in LoRa Wireless Networks."
