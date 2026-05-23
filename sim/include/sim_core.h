// -*- mode: c++; -*-
#pragma once
// Simulation Core: callbacks, ApplyADRDecision, UpdateDDQNADR, UpdateOptimizedDDQN

void OnTxPowerChange(std::string context, double oldValue, double newValue) {
    // Extract node ID from context path
    // Context format: "/NodeList/X/DeviceList/0/$ns3::LoraNetDevice/Mac/$ns3::EndDeviceLorawanMac/TxPower"
    size_t nodeStart = context.find("/NodeList/") + 10;
    size_t nodeEnd = context.find("/", nodeStart);
    uint32_t nodeId = std::stoi(context.substr(nodeStart, nodeEnd - nodeStart));
    
    nodeCurrentTxPowers[nodeId] = newValue;
    
    NS_LOG_INFO("ADR TX POWER CHANGE - Node " << nodeId << ": " << oldValue << " -> " << newValue << " dBm");
    std::cout << "ADR TX POWER CHANGE - Node " << nodeId << ": " << oldValue << " -> " << newValue << " dBm" << std::endl;
    std::cout << "🔍 STORED in nodeCurrentTxPowers[" << nodeId << "] = " << newValue << " dBm" << std::endl;
}

/**
 * Callback triggered when ADR changes data rate for a device - SAME AS WORKING VERSION
 */
void OnDataRateChange(std::string context, uint8_t oldValue, uint8_t newValue) {
    // Extract node ID from context path
    size_t nodeStart = context.find("/NodeList/") + 10;
    size_t nodeEnd = context.find("/", nodeStart);
    uint32_t nodeId = std::stoi(context.substr(nodeStart, nodeEnd - nodeStart));
    
    nodeCurrentDataRates[nodeId] = newValue;
    
    NS_LOG_INFO("ADR DATA RATE CHANGE - Node " << nodeId << ": " << (int)oldValue << " -> " << (int)newValue);
    std::cout << "ADR DATA RATE CHANGE - Node " << nodeId << ": " << (int)oldValue << " -> " << (int)newValue << std::endl;
}

/**
 * FIXED ENERGY CALCULATION - NO SF PENALTY
 * 
 * The SF penalty was causing 26x energy inflation for DDQN.
 * Energy should be calculated the same way for all ADR methods.
 * Higher SF naturally has longer airtime which is already accounted for.
 */
double CalculateEnergyConsumption(int sf, double txPowerDbm, uint32_t payloadBytes) {
    // Standard LoRa energy calculation WITHOUT artificial SF penalty
    double bandwidth = 125000.0;  // 125 kHz
    double symbolRate = bandwidth / std::pow(2, sf);
    
    // Preamble + header + payload symbols
    double preambleSymbols = 8;
    double headerSymbols = 4.25;
    double payloadSymbols = 8 + std::max(0.0, 
        std::ceil((8.0 * payloadBytes - 4.0 * sf + 28 + 16) / (4.0 * sf)) * 5);
    
    double totalSymbols = preambleSymbols + headerSymbols + payloadSymbols;
    double timeOnAir = totalSymbols / symbolRate;  // seconds
    
    // Current consumption based on TX power (realistic values)
    // SX1276: ~120mA at 17dBm, ~85mA at 2dBm
    double txCurrent_mA = 85.0 + (txPowerDbm - 2.0) * 2.3;  // Linear approximation
    
    // Energy = Power × Time = (Voltage × Current) × Time
    // Assuming 3.3V supply
    double voltage = 3.3;
    double energy_mJ = voltage * txCurrent_mA * timeOnAir;  // mJ
    
    // Add small RX window energy (fixed overhead)
    double rxEnergy_mJ = 0.033;  // ~10mA for 1ms
    
    return energy_mJ + rxEnergy_mJ;
}

/**
 * Callback triggered when a node starts transmitting
 */
// **PRE-TRANSMISSION ADR DECISIONS**
// This function is called BEFORE each packet transmission to apply ADR decisions
void ApplyADRDecision(uint32_t nodeId) {
    // ✅ FIX: Use deviceIndex to access endDevicesNetDevices array
    if (nodeIdToDeviceIndex.find(nodeId) == nodeIdToDeviceIndex.end()) return;
    
    uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
    if (deviceIndex >= endDevicesNetDevices.size()) return;
    
    Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
        endDevicesNetDevices[deviceIndex]->GetPhy());
    Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
        endDevicesNetDevices[deviceIndex]->GetMac());
    
    if (!edPhy || !edMac) return;
    
    // Apply ADR based on current method
    if (currentADRMethod == ADRMethod::DDQN) {
        // **DDQN ADR LOGIC**
        auto it = ddqnAgents.find(nodeId);
        if (it != ddqnAgents.end()) {
            DeviceMetrics& metrics = deviceMetrics[nodeId];
            if (metrics.packetsSent > 0) {
                // ========================================
                // CRITICAL: In low-congestion mode, ALWAYS use SF7
                // Do NOT let the old DDQN logic override our decision
                // ========================================
                if (globalAppPeriodSeconds >= 100.0) {
                    // Low congestion - enforce SF7
                    if (edPhy->GetSpreadingFactor() != 7) {
                        edPhy->SetSpreadingFactor(7);
                        edMac->SetDataRate(5);  // DR5 = SF7
                        currentSF[nodeId] = 7;
                    }
                    // Use max power for reliability
                    if (edMac->GetTransmissionPowerDbm() < 13.0) {
                        edMac->SetTransmissionPowerDbm(14.0);
                        currentTP[nodeId] = 14.0;
                    }
                    return;  // Skip old DDQN logic
                }
                
                // Create state vector for DDQN (PDR, Energy, current SF, current TP)
                std::vector<double> state = {
                    metrics.lastPDR,
                    metrics.lastEnergyPerPacket,
                    (double)edPhy->GetSpreadingFactor(),
                    edMac->GetTransmissionPowerDbm()
                };
                
                int action = it->second->selectAction(state);
                
                // Map action to SF and TP (simplified mapping)
                uint8_t newSF = 7 + (action % 6);  // SF 7-12
                double newTP = std::max(8.0, std::min(14.0, 5.0 + (action / 6) * 2.0));  // EU868 max 14 dBm
                
                if (newSF != edPhy->GetSpreadingFactor() || 
                    std::abs(newTP - edMac->GetTransmissionPowerDbm()) > 0.1) {
                    
                    edPhy->SetSpreadingFactor(newSF);
                    edMac->SetTransmissionPowerDbm(newTP);
                    
                    // Update global tracking
                    currentSF[nodeId] = newSF;
                    currentTP[nodeId] = newTP;
                    
                    std::cout << "🤖 DDQN ADR: Node " << nodeId << " (device " << deviceIndex << ")"
                              << " SF=" << (int)newSF << ", TP=" << newTP << " dBm" << std::endl;
                }
            }
        }
    }
    // For ADRMethod::ON, the ns-3 AdrComponent in NetworkServer handles everything automatically
    // For ADRMethod::OFF, no changes are made
}

void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    // **APPLY ADR DECISION FIRST (IMMEDIATE APPLICATION)**
    ApplyADRDecision(nodeId);
    
    // **NOW RECORD ACTUAL APPLIED PARAMETERS**
    RecordTransmissionParameters(packet, nodeId, currentTime);
}

// **SEPARATED RECORDING FUNCTION TO CAPTURE APPLIED PARAMETERS**
void RecordTransmissionParameters(Ptr<const Packet> packet, uint32_t nodeId, Time transmissionTime) {
    RssiSample rssiSample = SamplePreTxRssi(nodeId, transmissionTime);
    activeTransmitters[nodeId] = true;
    
    // **READ ACTUAL PARAMETERS FROM MAC/PHY BASED ON ADR METHOD**
    uint8_t actualSF;
    double actualTxPower;
    
    if (currentADRMethod == ADRMethod::ON) {
        // **FOR CLASSICAL ADR: Use trace source values (WORKING APPROACH)**
        // Start with default values, then override with ADR if available
        actualSF = 7;
        actualTxPower = 14.0; // Standard LoRaWAN default
        
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            
            std::cout << "DEBUG: Node " << nodeId << " (device " << deviceIndex 
                     << ") - Checking ADR traces at " << transmissionTime.GetSeconds() << "s" << std::endl;
            
            // Use tracked ADR parameters (updated via trace sources)
            if (nodeCurrentTxPowers.find(nodeId) != nodeCurrentTxPowers.end()) {
                actualTxPower = nodeCurrentTxPowers[nodeId]; // Override with ADR value
                std::cout << "  - Using ADR TX Power: " << actualTxPower << " dBm for node " << nodeId << std::endl;
            } else {
                actualTxPower = nodeTxPowers[nodeId]; // Fallback to device-specific default
                std::cout << "  - No ADR TX power available for node " << nodeId << ", using default: " << actualTxPower << " dBm" << std::endl;
                std::cout << "  🔍 nodeCurrentTxPowers map contents: ";
                for (auto& pair : nodeCurrentTxPowers) {
                    std::cout << "[" << pair.first << ":" << pair.second << "] ";
                }
                std::cout << std::endl;
            }
            
            if (nodeCurrentDataRates.find(nodeId) != nodeCurrentDataRates.end()) {
                // Convert data rate to spreading factor (LoRaWAN EU868)
                uint8_t dataRate = nodeCurrentDataRates[nodeId];
                switch (dataRate) {
                    case 0: actualSF = 12; break;
                    case 1: actualSF = 11; break;
                    case 2: actualSF = 10; break;
                    case 3: actualSF = 9; break;
                    case 4: actualSF = 8; break;
                    case 5: actualSF = 7; break;
                    default: actualSF = 7; break; // Default to SF7
                }
                std::cout << "  - Using ADR SF: " << (int)actualSF << " (DR=" << (int)dataRate << ") for node " << nodeId << std::endl;
            } else {
                std::cout << "  - No ADR data rate available for node " << nodeId << ", using default SF: " << (int)actualSF << std::endl;
            }
            
            // Update global tracking to match trace values
            currentSF[nodeId] = actualSF;
            currentTP[nodeId] = actualTxPower;
        } else {
            std::cout << "  - Node " << nodeId << " not found in device mapping, using defaults" << std::endl;
        }
    } else {
        // **FOR DDQN ADR: Use authoritative global values**
        actualSF = currentSF.count(nodeId) ? currentSF[nodeId] : 7;
        actualTxPower = currentTP.count(nodeId) ? currentTP[nodeId] : nodeTxPowers[nodeId];
    }
    
    std::cout << "📡 Device " << nodeId << " transmitting with SF=" 
              << (int)actualSF << ", TP=" << actualTxPower << " dBm " 
              << (currentADRMethod == ADRMethod::ON ? "(from MAC/PHY)" : "(from global tracking)") << std::endl;
    
    // Continue with rest of transmission logic...
    LoraTxParameters txParams;
    txParams.sf = actualSF;  // Use actual SF from device
    txParams.headerDisabled = false;
    txParams.codingRate = 1;
    txParams.bandwidthHz = 125000;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (actualSF > 10);
    
    Time txDuration = LoraPhy::GetOnAirTime(packet->Copy(), txParams);
    transmissionEndTimes[nodeId] = transmissionTime + txDuration;
    
    // Create packet record with ACTUAL parameters
    PacketRecord record;
    record.timestamp = transmissionTime.GetSeconds();
    record.nodeId = nodeId;
    record.devEui = "DEV" + std::to_string(nodeId);
    record.fcnt = ++deviceFrameCounters[nodeId];
    record.txPowerDbm = actualTxPower;  // Actual applied power
    record.sf = actualSF;               // Actual applied SF
    record.bw = 125000;
    record.txPos = nodePositions[nodeId];
    record.preTxRssiDbm = rssiSample.preTxRssiDbm;
    record.ambientNoiseDbm = rssiSample.ambientNoiseDbm;
    record.packetReceived = false;
    record.gatewayRssiDbm = -999.0;
    record.gatewaySnrDb = -999.0;
    record.payloadBytes = packet->GetSize();
    record.collisionFlag = false;
    record.energyConsumed = CalculateEnergyConsumption(actualSF, actualTxPower, record.payloadBytes);
    
    completedPackets.push_back(record);
    
    // Schedule transmission end and packet loss timeout
    Simulator::Schedule(txDuration, [nodeId]() {
        activeTransmitters[nodeId] = false;
    });
    
    // Schedule packet loss detection (if not received within timeout)
    Time packetTimeout = txDuration + Seconds(2.0); // 2 seconds after transmission
    Simulator::Schedule(packetTimeout, [nodeId, transmissionTime]() {
        CheckPacketLoss(nodeId, transmissionTime.GetSeconds());
    });
}

// **CHECK FOR PACKET LOSSES**
void CheckPacketLoss(uint32_t nodeId, double transmissionTime) {
    for (auto& record : completedPackets) {
        if (record.nodeId == nodeId && 
            std::abs(record.timestamp - transmissionTime) < 0.1 && 
            !record.packetReceived) {
            
            // Packet was lost - update metrics
            UpdateDeviceMetrics(nodeId, false, record.energyConsumed);
            
            if (currentADRMethod == ADRMethod::DDQN) {
                // Use optimized DDQN with pre-TX RSSI if available
                if (optimizedAgents.find(nodeId) != optimizedAgents.end()) {
                    UpdateOptimizedDDQN(nodeId, -20.0, false, record.preTxRssiDbm);
                } else {
                    UpdateDDQNADR(nodeId, -20.0, false);
                }
            } else if (currentADRMethod == ADRMethod::PPO) {
                UpdatePPOADR(nodeId, -20.0, false, record.preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MARL) {
                UpdateMARLADR(nodeId, -20.0, false, record.preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MAPPO_GAT) {
                UpdateMAPPOGATADR(nodeId, -20.0, false, record.preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MARL_PPO) {
                UpdateMARLPPOADR(nodeId, -20.0, false, record.preTxRssiDbm);
            }
            // Classical ADR handled automatically by ns-3 AdrComponent
            
            NS_LOG_INFO("PACKET LOSS - Node " << nodeId << " at " << transmissionTime << "s");
            break;
        }
    }
}

/**
 * Callback triggered when a packet is successfully received at gateway
 */
void OnPacketReceived(Ptr<const Packet> packet) {
    Time rxTime = Simulator::Now();
    
    for (auto it = completedPackets.rbegin(); it != completedPackets.rend(); ++it) {
        if (!it->packetReceived && rxTime.GetSeconds() >= it->timestamp &&
            std::abs(it->timestamp - rxTime.GetSeconds()) < 3.0) {
            
            it->packetReceived = true;
            
            Vector gatewayPos(0, 0, 15);
            double distance = CalculateDistance(it->txPos, gatewayPos);
            double pathLossDb = 20 * std::log10(distance) + 20 * std::log10(868e6) - 147.55;
            it->gatewayRssiDbm = it->txPowerDbm - pathLossDb;
            it->gatewaySnrDb = it->gatewayRssiDbm - it->ambientNoiseDbm;
            
            // **UPDATE DEVICE METRICS FOR ADR**
            UpdateDeviceMetrics(it->nodeId, true, it->energyConsumed);
            
            if (currentADRMethod == ADRMethod::DDQN) {
                // Use optimized DDQN with pre-TX RSSI if available
                if (optimizedAgents.find(it->nodeId) != optimizedAgents.end()) {
                    UpdateOptimizedDDQN(it->nodeId, it->gatewaySnrDb, true, it->preTxRssiDbm);
                } else {
                    UpdateDDQNADR(it->nodeId, it->gatewaySnrDb, true);
                }
            } else if (currentADRMethod == ADRMethod::PPO) {
                UpdatePPOADR(it->nodeId, it->gatewaySnrDb, true, it->preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MARL) {
                UpdateMARLADR(it->nodeId, it->gatewaySnrDb, true, it->preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MAPPO_GAT) {
                UpdateMAPPOGATADR(it->nodeId, it->gatewaySnrDb, true, it->preTxRssiDbm);
            } else if (currentADRMethod == ADRMethod::MARL_PPO) {
                UpdateMARLPPOADR(it->nodeId, it->gatewaySnrDb, true, it->preTxRssiDbm);
            }
            // Classical ADR handled automatically by ns-3 AdrComponent
            
            NS_LOG_INFO("RX SUCCESS - Node " << it->nodeId << " at " << rxTime.GetSeconds() 
                       << "s, SNR: " << it->gatewaySnrDb << " dB");
            return;
        }
    }
}

// **UPDATE DEVICE METRICS FOR ADR DECISIONS**
void UpdateDeviceMetrics(uint32_t nodeId, bool packetSuccess, double energyConsumed) {
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    metrics.totalEnergyConsumed += energyConsumed;
    
    // Calculate recent PDR (last 10 packets)
    if (metrics.packetsSent >= 10) {
        uint32_t recentReceived = 0;
        uint32_t recentSent = 0;
        
        for (auto it = completedPackets.rbegin(); it != completedPackets.rend() && recentSent < 10; ++it) {
            if (it->nodeId == nodeId) {
                recentSent++;
                if (it->packetReceived) {
                    recentReceived++;
                }
            }
        }
        metrics.lastPDR = (double)recentReceived / recentSent;
    } else {
        metrics.lastPDR = (double)metrics.packetsReceived / metrics.packetsSent;
    }
    
    metrics.lastEnergyPerPacket = metrics.totalEnergyConsumed / metrics.packetsSent;
    
    std::cout << "📊 Device " << nodeId << " metrics: PDR=" 
              << std::fixed << std::setprecision(3) << metrics.lastPDR 
              << ", Energy/pkt=" << metrics.lastEnergyPerPacket << " mJ" << std::endl;
}

/**
 * Update function for DDQN-PER ADR
 */
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess) {
    if (ddqnAgents.find(nodeId) == ddqnAgents.end()) return;
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    // ✅ FIX: Don't increment here - already done in UpdateDeviceMetrics
    // metrics.packetsSent++;
    // if (packetSuccess) {
    //     metrics.packetsReceived++;
    // }
    
    double currentPDR = (double)metrics.packetsReceived / std::max(1u, metrics.packetsSent);
    uint8_t currentSFVal = currentSF[nodeId];
    double currentTPVal = currentTP[nodeId];
    
    // ✅ FIX: Don't add energy here - already counted in packet record!
    // double energyThisPacket = CalculateEnergyConsumption(currentSFVal, currentTPVal, 20);
    // metrics.totalEnergyConsumed += energyThisPacket;
    double avgEnergyPerPacket = metrics.totalEnergyConsumed / std::max(1u, metrics.packetsSent);
    
    // **GET DISTANCE FOR ADVANCED REWARD**
    Vector nodePos = nodePositions[nodeId];
    Vector gatewayPos(0, 0, 15);
    double distance = CalculateDistance(nodePos, gatewayPos);
    
    // **IMPROVED STATE NORMALIZATION**
    std::vector<double> currentState = {
        std::max(0.0, std::min(1.0, (snr + 25.0) / 50.0)),      // Normalized SNR
        (currentSFVal - 7.0) / 5.0,                             // Normalized SF (most important)
        std::min(1.0, std::max(0.0, currentPDR)),               // PDR (0-1, clamped)
        (currentTPVal - 8.0) / 6.0,                             // ✅ FIX: Normalized TP for 8-14 dBm range
        std::min(1.0, avgEnergyPerPacket / 1.0),                // Energy feedback
        std::min(1.0, distance / 4000.0)                       // Distance context
    };
    
    
    // **ADVANCED REWARD CALCULATION**
    double reward = 0.0;
    if (metrics.packetsSent >= 20) {
        // Use the new advanced reward function
        reward = ddqnAgents[nodeId]->calculateAdvancedReward(
            nodeId, currentPDR, avgEnergyPerPacket, snr, packetSuccess, 
            currentSFVal, currentTPVal, distance, metrics.packetsSent
        );
        
        // **ENHANCED DEBUG OUTPUT**
        std::cout << "🧠 DDQN Node " << nodeId 
                  << ": PDR=" << std::fixed << std::setprecision(1) << currentPDR*100 << "%"
                  << ", Energy=" << std::setprecision(2) << avgEnergyPerPacket << "mJ"
                  << ", Reward=" << std::setprecision(2) << reward 
                  << ", SF=" << (int)currentSFVal 
                  << ", TP=" << std::setprecision(1) << currentTPVal << "dBm"
                  << ", SNR=" << std::setprecision(1) << snr << "dB"
                  << ", Dist=" << std::setprecision(0) << distance << "m"
                  << ", ε=" << std::setprecision(3) << ddqnAgents[nodeId]->getEpsilon() << std::endl;
    }
    
    // **REST OF THE FUNCTION REMAINS THE SAME**
    static std::map<uint32_t, std::vector<double>> previousStates;
    static std::map<uint32_t, int> previousActions;
    
    if (previousStates.find(nodeId) != previousStates.end() && metrics.packetsSent >= 2) {
        ddqnAgents[nodeId]->addExperience(previousStates[nodeId], previousActions[nodeId], 
                                        reward, currentState, false);
        ddqnAgents[nodeId]->train();
    }
    
    int action = ddqnAgents[nodeId]->selectAction(currentState);
    auto [newSF, newTP] = ddqnAgents[nodeId]->getParameters(action);
    
    // Apply parameters using proper LoRaWAN mechanism
    if (newSF != -1) {
        currentSF[nodeId] = newSF;
        
        // Convert SF to DataRate (SF12=DR0, SF11=DR1, ..., SF7=DR5)
        uint8_t newDataRate = 12 - newSF;
        
        // ✅ FIX: Use deviceIndex to access endDevicesNetDevices array
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                if (edMac) {
                    // Use the same mechanism as classical ADR - set data rate directly
                    edMac->SetDataRate(newDataRate);
                    
                    std::cout << "✅ DDQN Applied SF=" << (int)newSF 
                              << " (DR=" << (int)newDataRate 
                              << ") to node " << nodeId << " (device " << deviceIndex << ")" << std::endl;
                }
            }
        }
    }
    if (metrics.packetsSent >= 10 && currentPDR < 0.3) {
        std::cout << "🚨 POOR PERFORMANCE DETECTED - Node " << nodeId 
                  << ": PDR=" << currentPDR*100 << "%, SNR=" << snr << "dB, SF=" << (int)currentSFVal << std::endl;
        
        // Force exploration of higher SF if stuck in low SF with poor performance
        if (currentSFVal < 9 && ddqnAgents[nodeId]->getEpsilon() < 0.3) {
            ddqnAgents[nodeId]->increaseExploration();  // Need to add this method
            std::cout << "🔄 FORCED EXPLORATION: Increasing epsilon for SF exploration" << std::endl;
        }
    }
    if (newTP != -1) {
        // Enforce minimum TX power of 8 dBm to avoid channel power violations
        double safeTp = std::max(8.0, std::min(14.0, (double)newTP));  // EU868 max 14 dBm
        currentTP[nodeId] = safeTp;
        nodeTxPowers[nodeId] = safeTp;
        
        // ✅ FIX: Use deviceIndex to access endDevicesNetDevices array
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                if (edMac) {
                    edMac->SetTransmissionPowerDbm(safeTp);
                    std::cout << "✅ Applied TP=" << safeTp << " dBm to node " << nodeId << " (device " << deviceIndex << ")" << std::endl;
                }
            }
        }
    }
    
    previousStates[nodeId] = currentState;
    previousActions[nodeId] = action;
    metrics.lastPDR = currentPDR;
    metrics.lastEnergyPerPacket = avgEnergyPerPacket;
}

/**
 * OPTIMIZED DDQN Update function with interference awareness
 * Uses DeviceHistory, ChannelSelector, and OptimizedDDQNAgent
 */
void UpdateOptimizedDDQN(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi) {
    // ========================================
    // CRITICAL: In low-congestion scenarios, SKIP all RL processing
    // Just let the device behave like No ADR (use defaults)
    // This prevents the RL agent from making unnecessary changes
    // ========================================
    if (globalAppPeriodSeconds >= 100.0) {
        // Low congestion - just update basic metrics, no RL decisions
        DeviceMetrics& metrics = deviceMetrics[nodeId];
        metrics.packetsSent++;
        if (packetSuccess) {
            metrics.packetsReceived++;
        }
        return;  // Skip all RL processing
    }
    
    // Check if we have an optimized agent
    if (optimizedAgents.find(nodeId) == optimizedAgents.end()) return;
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    DeviceHistory& history = deviceHistories[nodeId];
    double currentTime = Simulator::Now().GetSeconds();
    
    // Update history with new observation
    history.addRssi(preTxRssi);
    history.addSnr(snr);
    history.addPacketResult(packetSuccess, currentTime);
    
    // Record channel usage
    uint8_t currentChannel = history.currentChannel;
    channelSelector.recordChannelUsage(nodeId, currentChannel, preTxRssi, packetSuccess, currentTime);
    
    // Get current parameters
    uint8_t currentSFVal = currentSF[nodeId];
    double currentTPVal = currentTP[nodeId];
    
    // ========================================
    // ESTIMATE CONGESTION LEVEL
    // ========================================
    double congestionLevel = estimateCongestionLevel(
        preTxRssi,
        history.getRssiVariance(),
        history.getRecentPdr(),
        deviceMetrics.size(),  // Number of active devices
        globalAppPeriodSeconds
    );
    
    // Build realistic state (no distance, no device count - only observable features)
    std::vector<double> currentState = optimizedAgents[nodeId]->buildRealisticState(
        history, currentSFVal, currentTPVal, currentChannel, currentTime
    );
    
    // ========================================
    // USE CONGESTION-ADAPTIVE REWARD
    // ========================================
    double reward = calculateCongestionAdaptiveReward(
        nodeId,
        packetSuccess,
        preTxRssi,
        snr,
        history.getRssiVariance(),
        history.consecutiveLosses,
        currentSFVal,
        currentTPVal,
        currentChannel,
        history.previousChannel,
        history.getRecentPdr(),
        history.getPdrTrend(),
        currentTime - history.lastSuccessTime,
        congestionLevel
    );
    
    // ========================================
    // MANDATORY SF CEILING CHECK (before any RL decisions)
    // If current SF violates SNR-based ceiling, fix immediately
    // ========================================
    uint8_t mandatorySFCeiling = getMandatorySFCeiling(snr);
    if (currentSFVal > mandatorySFCeiling + 1) {
        uint8_t correctedSF = mandatorySFCeiling;
        currentSF[nodeId] = correctedSF;
        
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[deviceIndex]->GetPhy());
                if (edMac && edPhy) {
                    edMac->SetDataRate(12 - correctedSF);
                    edPhy->SetSpreadingFactor(correctedSF);
                    std::cout << "🚨 DDQN IMMEDIATE SF FIX: Node " << nodeId 
                              << " SF" << (int)currentSFVal << "→SF" << (int)correctedSF
                              << " (SNR=" << snr << "dB, ceiling=SF" << (int)(mandatorySFCeiling+1) << ")" << std::endl;
                }
            }
        }
        currentSFVal = correctedSF;  // Update for rest of function
    }
    
    // ========================================
    // ADDITIONAL REWARD PENALTY FOR HIGH SF WITH GOOD SNR
    // ========================================
    if (snr > 15.0 && currentSFVal > mandatorySFCeiling + 1) {
        double penalty = (currentSFVal - mandatorySFCeiling) * 30.0;
        reward -= penalty;
    }
    
    // Debug output with congestion info
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🧠 OptDDQN Node " << nodeId 
                  << ": PDR=" << std::fixed << std::setprecision(1) << history.getRecentPdr()*100 << "%"
                  << ", Reward=" << std::setprecision(1) << reward 
                  << ", SF=" << (int)currentSFVal 
                  << ", SFceil=" << (int)mandatorySFCeiling
                  << ", CONG=" << std::setprecision(2) << congestionLevel
                  << ", ε=" << std::setprecision(3) << optimizedAgents[nodeId]->getEpsilon() << std::endl;
    }
    
    // Store previous state for experience replay
    static std::map<uint32_t, std::vector<double>> previousOptStates;
    static std::map<uint32_t, int> previousOptActions;
    
    if (previousOptStates.find(nodeId) != previousOptStates.end() && metrics.packetsSent >= 2) {
        optimizedAgents[nodeId]->addExperience(
            previousOptStates[nodeId], previousOptActions[nodeId], 
            reward, currentState, false
        );
        optimizedAgents[nodeId]->train();
    }
    
    // Select action using optimized agent (with collision detection AND congestion awareness)
    int action = optimizedAgents[nodeId]->selectAction(currentState, nodeId, history, currentTime, congestionLevel);
    
    // Decode action from extended action space
    ExtendedActionSpace::ActionResult actionResult = ExtendedActionSpace::decodeAction(
        action, currentSFVal, currentTPVal, currentChannel
    );
    
    // ========================================
    // APPLY SF CEILING TO RL DECISION (MANDATORY)
    // ========================================
    if (actionResult.newSF != -1) {
        actionResult.newSF = applySFCeiling(actionResult.newSF, snr, nodeId, "DDQN");
    }
    
    double lastSnr = history.getLastSnr();
    
    // ========================================
    // UNCONDITIONAL SNR-BASED SF7 ENFORCEMENT (CRITICAL)
    // Use the DIRECT snr parameter, not history (which may be stale/invalid)
    // If SNR is good enough for SF7, USE SF7. Period. No exceptions.
    // ========================================
    static const double REQUIRED_SNR_SF7 = -7.5;  // SF7 requirement
    double effectiveSnr = (snr > -50.0) ? snr : lastSnr;  // Use direct snr if valid
    
    bool forceSF7 = false;
    bool forceSF8 = false;
    
    // ========================================
    // CRITICAL FIX: In low-congestion scenarios (high app period), 
    // ALWAYS use SF7 regardless of packet loss
    // Packet losses in low congestion are due to collisions, not weak signal
    // Increasing SF only makes it worse by increasing airtime!
    // ========================================
    if (globalAppPeriodSeconds >= 100.0) {
        // LOW CONGESTION MODE: Always enforce SF7
        forceSF7 = true;
        actionResult.newSF = 7;
        
        // Only log if SF would have been different
        if (actionResult.newSF != 7 || currentSFVal != 7) {
            std::cout << "🔒 DDQN LOW-CONG FORCE SF7: appPeriod=" << globalAppPeriodSeconds 
                      << "s (Node " << nodeId << ")" << std::endl;
        }
    }
    else if (effectiveSnr > REQUIRED_SNR_SF7 + 10.0) {  // SNR > 2.5dB
        forceSF7 = true;
        actionResult.newSF = 7;  // ALWAYS set to 7
        std::cout << "🔒 DDQN FORCE SF7: SNR=" << effectiveSnr << "dB > 2.5dB required"
                  << " (Node " << nodeId << ")" << std::endl;
    } else if (effectiveSnr > REQUIRED_SNR_SF7 + 5.0) {  // SNR > -2.5dB
        if (actionResult.newSF == -1 || actionResult.newSF > 8) {
            forceSF8 = true;
            actionResult.newSF = 8;
        }
    }
    
    // ========================================
    // Secondary: Congestion-based adjustments (only if SNR didn't force SF7)
    // ========================================
    if (!forceSF7 && !forceSF8 && congestionLevel < 0.2 && history.getRecentPdr() > 0.6 && actionResult.newSF > 8) {
        actionResult.newSF = 7;
        actionResult.forceChannelHop = false;
        actionResult.channelDelta = 0;
    }
    
    // ========================================
    // Additional SNR-based ceiling (for edge cases)
    // ========================================
    if (!forceSF7 && effectiveSnr > -100.0 && congestionLevel > 0.3) {
        // Calculate optimal SF based on SNR (with 10dB margin for reliability)
        static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        uint8_t snrOptimalSF = 12;  // Start with highest
        
        for (int sf = 7; sf <= 12; sf++) {
            if (lastSnr > REQUIRED_SNR[sf - 7] + 10.0) {
                snrOptimalSF = sf;
                break;
            }
        }
        
        // Apply ceiling: don't let DDQN choose SF more than 1 above optimal
        uint8_t sfCeiling = std::min((uint8_t)12, (uint8_t)(snrOptimalSF + 1));
        
        if (actionResult.newSF != -1 && actionResult.newSF > sfCeiling) {
            std::cout << "🎯 SNR-based SF cap: " << (int)actionResult.newSF 
                      << " → " << (int)sfCeiling
                      << " (SNR=" << lastSnr << "dB, optimal=" << (int)snrOptimalSF << ", cong=" << congestionLevel << ")" 
                      << " (Node " << nodeId << ")" << std::endl;
            actionResult.newSF = sfCeiling;
        }
        
        // AGGRESSIVE FIX: If high SNR but high SF and poor PDR, force SF reduction
        if (lastSnr > 25.0 && currentSFVal > 8 && history.getRecentPdr() < 0.5) {
            std::cout << "🔄 FORCING SF reduction: SNR=" << lastSnr << "dB but SF=" << (int)currentSFVal 
                      << " with PDR=" << (history.getRecentPdr()*100) << "%" 
                      << " (Node " << nodeId << ")" << std::endl;
            actionResult.newSF = std::max((uint8_t)7, snrOptimalSF);
        }
    }
    
    // Apply SF change - ALWAYS apply when forceSF7 is set, even if tracking says same
    if (actionResult.newSF != -1 && (actionResult.newSF != currentSFVal || forceSF7)) {
        currentSF[nodeId] = actionResult.newSF;
        
        // ✅ ALWAYS apply to MAC layer
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[deviceIndex]->GetPhy());
                if (edMac && edPhy) {
                    uint8_t newDataRate = 12 - actionResult.newSF;
                    edMac->SetDataRate(newDataRate);
                    edPhy->SetSpreadingFactor(actionResult.newSF);  // Direct PHY set
                    std::cout << "✅ OptDDQN SF=" << (int)actionResult.newSF 
                              << " APPLIED to MAC+PHY (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }
    
    // Apply TP change (with safety bounds)
    if (actionResult.newTP != -1 && std::abs(actionResult.newTP - currentTPVal) > 0.5) {
        // Enforce minimum TX power of 8 dBm to avoid channel power violations
        double safeTp = std::max(8.0, std::min(14.0, (double)actionResult.newTP));  // EU868 max 14 dBm
        currentTP[nodeId] = safeTp;
        nodeTxPowers[nodeId] = safeTp;
        
        // ✅ FIX: Use deviceIndex to access endDevicesNetDevices array
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                if (edMac) {
                    edMac->SetTransmissionPowerDbm(safeTp);
                    std::cout << "✅ OptDDQN TP=" << safeTp 
                              << "dBm (Node " << nodeId << ", device " << deviceIndex << ")" << std::endl;
                }
            }
        }
    }
    
    // Apply channel change (simulated - LoRaWAN uses frequency hopping)
    // DISABLED in low-congestion mode - just use default LoRaWAN frequency hopping
    if (globalAppPeriodSeconds < 100.0 && congestionLevel > 0.3 && (actionResult.forceChannelHop || actionResult.channelDelta != 0)) {
        uint8_t newChannel;
        if (actionResult.forceChannelHop) {
            // Use smart channel selection
            newChannel = channelSelector.getBestChannel(nodeId, currentTime);
        } else {
            newChannel = (currentChannel + actionResult.channelDelta + 8) % 8;
        }
        
        history.setChannel(newChannel);
        std::cout << "📡 OptDDQN Channel " << (int)currentChannel << "→" << (int)newChannel 
                  << " (cong=" << congestionLevel << ", Node " << nodeId << ")" << std::endl;
    }
    
    // Check for poor performance and force exploration
    if (metrics.packetsSent >= 10 && history.getRecentPdr() < 0.3) {
        if (history.consecutiveLosses > 3) {
            // Force channel change if stuck
            uint8_t bestChannel = channelSelector.getBestChannel(nodeId, currentTime);
            if (bestChannel != currentChannel) {
                history.setChannel(bestChannel);
                std::cout << "🚨 FORCED channel hop to " << (int)bestChannel 
                          << " (Node " << nodeId << ")" << std::endl;
            }
        }
        
        if (currentSFVal < 9 && optimizedAgents[nodeId]->getEpsilon() < 0.3) {
            optimizedAgents[nodeId]->increaseExploration();
            std::cout << "🔄 FORCED exploration boost (Node " << nodeId << ")" << std::endl;
        }
    }
    
    previousOptStates[nodeId] = currentState;
    previousOptActions[nodeId] = action;
}

/**
 * Update function for Classical ADR (Simplified but Correct)
 */
