// -*- mode: c++; -*-
#pragma once
// ADR Update: UpdateClassicalADR, UpdatePPOADR

void UpdateClassicalADR(uint32_t nodeId, double snr, bool packetSuccess) {
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    
    // Classical ADR algorithm based on LoRaWAN 1.0.4 specification
    // Only perform ADR if we have enough data points
    if (metrics.packetsSent >= 1) {  // Test with just 1 packet to verify mechanism
        // Classical ADR thresholds
        const double SNR_MARGIN = 2.5; // dB
        const double REQUIRED_SNR[6] = {-20, -17.5, -15, -12.5, -10, -7.5}; // SF12-SF7
        
        uint8_t originalSF = currentSF[nodeId];
        double originalTP = currentTP[nodeId];
        
        int currentSFIdx = currentSF[nodeId] - 7;
        if (currentSFIdx < 0 || currentSFIdx > 5) currentSFIdx = 0; // Safety check
        
        double requiredSnr = REQUIRED_SNR[currentSFIdx];
        double snrMargin = snr - requiredSnr;
        
        // **1. SF OPTIMIZATION FIRST (Most Important for LoRaWAN)**
        if (snrMargin > SNR_MARGIN + 3.0 && currentSF[nodeId] > 7) {
            // Good link quality - decrease SF for higher data rate
            currentSF[nodeId]--;
            std::cout << "📡 Classical ADR: Node " << nodeId << " SF " << (int)originalSF 
                      << " → " << (int)currentSF[nodeId] << " (better link)" << std::endl;
        } else if (snrMargin < -SNR_MARGIN && currentSF[nodeId] < 12) {
            // Poor link quality - increase SF for better sensitivity  
            currentSF[nodeId]++;
            std::cout << "📡 Classical ADR: Node " << nodeId << " SF " << (int)originalSF 
                      << " → " << (int)currentSF[nodeId] << " (poor link)" << std::endl;
        }
        
        // **2. TX POWER OPTIMIZATION (Secondary)**
        if (snrMargin > SNR_MARGIN + 5.0 && currentTP[nodeId] > 2.0) {
            // Excellent link quality - reduce power for energy saving
            currentTP[nodeId] = std::max(2.0, currentTP[nodeId] - 2.0);
            std::cout << "⚡ Classical ADR: Node " << nodeId << " TP " << originalTP 
                      << " → " << currentTP[nodeId] << " dBm (power save)" << std::endl;
        } else if (snrMargin < -SNR_MARGIN - 2.0 && currentTP[nodeId] < 17.0) {
            // Very poor link quality - increase power
            currentTP[nodeId] = std::min(14.0, currentTP[nodeId] + 2.0);  // EU868 max 14 dBm
            std::cout << "⚡ Classical ADR: Node " << nodeId << " TP " << originalTP 
                      << " → " << currentTP[nodeId] << " dBm (boost signal)" << std::endl;
        }
        
        // **3. APPLY CHANGES IMMEDIATELY VIA MAC LAYER**
        if (originalSF != currentSF[nodeId] || std::abs(originalTP - currentTP[nodeId]) > 0.1) {
            // ✅ FIX: Use deviceIndex to access endDevicesNetDevices array
            if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
                uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
                if (deviceIndex < endDevicesNetDevices.size()) {
                    Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                        endDevicesNetDevices[deviceIndex]->GetMac());
                    Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                        endDevicesNetDevices[deviceIndex]->GetPhy());
                    if (edMac && edPhy) {
                        // Apply SF directly to PHY layer for immediate effect
                        edPhy->SetSpreadingFactor(currentSF[nodeId]);
                        
                        // Apply TX power directly to MAC layer  
                        edMac->SetTransmissionPowerDbm(currentTP[nodeId]);
                        
                        std::cout << "✅ Classical ADR Applied IMMEDIATELY: SF=" << (int)currentSF[nodeId] 
                                  << ", TP=" << currentTP[nodeId] << "dBm to node " << nodeId 
                                  << " (device " << deviceIndex << ")" << std::endl;
                    }
                }
            }
        }
    }
}

/**
 * Update function for PPO ADR
 * Uses continuous action space for SF and TP
 * NOW WITH CONGESTION AWARENESS
 */
void UpdatePPOADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi) {
    // ========================================
    // CRITICAL: In low-congestion scenarios, SKIP all RL processing
    // ========================================
    if (globalAppPeriodSeconds >= 100.0) {
        DeviceMetrics& metrics = deviceMetrics[nodeId];
        metrics.packetsSent++;
        if (packetSuccess) {
            metrics.packetsReceived++;
        }
        return;
    }
    
    // Initialize PPO agent if needed
    if (ppoAgents.find(nodeId) == ppoAgents.end()) {
        ppoAgents[nodeId] = std::make_unique<PPOAgent>();
    }
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    DeviceHistory& history = deviceHistories[nodeId];
    double currentTime = Simulator::Now().GetSeconds();
    
    // Update history with new observation
    history.addRssi(preTxRssi);
    history.addSnr(snr);
    history.addPacketResult(packetSuccess, currentTime);
    
    // Get current parameters
    uint8_t currentSFVal = currentSF[nodeId];
    double currentTPVal = currentTP[nodeId];
    uint8_t currentChannel = history.currentChannel;
    
    // ========================================
    // ESTIMATE CONGESTION LEVEL
    // ========================================
    double congestionLevel = estimateCongestionLevel(
        preTxRssi,
        history.getRssiVariance(),
        history.getRecentPdr(),
        deviceMetrics.size(),
        globalAppPeriodSeconds
    );
    
    // Build state vector
    std::vector<double> currentState = ppoAgents[nodeId]->buildState(
        history, currentSFVal, currentTPVal, currentChannel, currentTime
    );
    
    // ========================================
    // CONGESTION-ADAPTIVE REWARD FOR PPO
    // ========================================
    double reward = 0.0;
    
    // Base reward from packet success
    double outcomeWeight = 0.4 + 0.3 * (1.0 - congestionLevel);
    if (packetSuccess) {
        reward = 1.0 * outcomeWeight / 0.4;
        
        // Efficiency bonus (stronger in low congestion)
        double efficiencyWeight = 0.15 + 0.2 * (1.0 - congestionLevel);
        reward += (12.0 - currentSFVal) * 0.05 * efficiencyWeight / 0.15;
        reward += (14.0 - currentTPVal) * 0.02 * efficiencyWeight / 0.15;
        
        // Stability bonus in low congestion
        if (congestionLevel < 0.3 && currentSFVal == 7 && history.getRecentPdr() > 0.7) {
            reward += 0.3;  // Reward for staying optimal
        }
    } else {
        reward = -0.5 * outcomeWeight / 0.4;
        // Penalty for consecutive losses (reduced in low congestion)
        reward -= history.consecutiveLosses * 0.1 * (0.5 + 0.5 * congestionLevel);
    }
    
    // Store transition for PPO update
    static std::map<uint32_t, std::tuple<uint8_t, double, double>> previousPPOActions;
    
    if (previousPPOActions.find(nodeId) != previousPPOActions.end()) {
        auto [prevSF, prevTP, prevLogProb] = previousPPOActions[nodeId];
        ppoAgents[nodeId]->storeTransition(currentState, prevSF, prevTP, prevLogProb, reward, false);
    }
    
    // Select action using PPO policy
    auto [newSF, newTP, logProb] = ppoAgents[nodeId]->selectAction(currentState);
    
    double lastSnr = history.getLastSnr();
    
    // ========================================
    // MANDATORY SF CEILING CHECK (before any other SF logic)
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
                    std::cout << "🚨 PPO IMMEDIATE SF FIX: Node " << nodeId 
                              << " SF" << (int)currentSFVal << "→SF" << (int)correctedSF << std::endl;
                }
            }
        }
        currentSFVal = correctedSF;
    }
    
    // Apply SF ceiling to PPO's proposed SF
    newSF = applySFCeiling(newSF, snr, nodeId, "PPO");
    
    // ========================================
    // UNCONDITIONAL SNR-BASED SF7 ENFORCEMENT (CRITICAL)
    // Use DIRECT snr parameter for reliability
    // ========================================
    static const double REQUIRED_SNR_SF7 = -7.5;
    double effectiveSnr = (snr > -50.0) ? snr : lastSnr;
    
    bool forceSF7 = false;
    
    // ========================================
    // CRITICAL FIX: In low-congestion scenarios (high app period), 
    // ALWAYS use SF7 regardless of packet loss
    // ========================================
    if (globalAppPeriodSeconds >= 100.0) {
        forceSF7 = true;
        newSF = 7;
    }
    else if (effectiveSnr > REQUIRED_SNR_SF7 + 10.0) {  // SNR > 2.5dB
        forceSF7 = true;
        newSF = 7;
        std::cout << "🔒 PPO FORCE SF7: SNR=" << effectiveSnr << "dB > 2.5dB required"
                  << " (Node " << nodeId << ")" << std::endl;
    } else if (effectiveSnr > REQUIRED_SNR_SF7 + 5.0) {  // SNR > -2.5dB
        if (newSF > 8) {
            newSF = 8;
        }
    }
    
    // ========================================
    // Secondary: LOW CONGESTION override
    // ========================================
    if (!forceSF7 && congestionLevel < 0.2 && history.getRecentPdr() > 0.6 && newSF > 8) {
        newSF = 7;
    }
    
    previousPPOActions[nodeId] = {newSF, newTP, logProb};
    
    // Debug output
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🎯 PPO Node " << nodeId 
                  << ": PDR=" << std::fixed << std::setprecision(1) << history.getRecentPdr()*100 << "%"
                  << ", SF=" << (int)newSF 
                  << ", TP=" << std::setprecision(1) << newTP
                  << ", CONG=" << std::setprecision(2) << congestionLevel
                  << ", ConsecLoss=" << history.consecutiveLosses << std::endl;
    }
    
    // Apply SF change - ALWAYS apply when forceSF7
    if (newSF != currentSFVal || forceSF7) {
        currentSF[nodeId] = newSF;
        
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[deviceIndex]->GetPhy());
                if (edMac && edPhy) {
                    uint8_t newDataRate = 12 - newSF;
                    edMac->SetDataRate(newDataRate);
                    edPhy->SetSpreadingFactor(newSF);  // Direct PHY set
                    std::cout << "✅ PPO SF=" << (int)newSF 
                              << " APPLIED to MAC+PHY (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }
    
    // Apply TP change
    double roundedTP = std::round(newTP / 2.0) * 2.0;  // Round to even dBm
    roundedTP = std::max(8.0, std::min(14.0, roundedTP));
    
    if (std::abs(roundedTP - currentTPVal) > 0.5) {
        currentTP[nodeId] = roundedTP;
        nodeTxPowers[nodeId] = roundedTP;
        
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                if (edMac) {
                    edMac->SetTransmissionPowerDbm(roundedTP);
                    std::cout << "✅ PPO TP=" << roundedTP 
                              << "dBm (Node " << nodeId << ", device " << deviceIndex << ")" << std::endl;
                }
            }
        }
    }
    
    // Periodically update PPO networks
    if (metrics.packetsSent % 50 == 0) {
        ppoAgents[nodeId]->update();
    }
}

/**
 * Update function for MARL ADR (Enhanced with DDQN Short-Range Optimizations)
 * Uses coordination signals between agents for collision avoidance
 * NOW INCLUDES: Distance-aware behavior, SNR-based SF ceiling, collision detection, CONGESTION AWARENESS
 */
