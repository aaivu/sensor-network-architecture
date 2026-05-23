// -*- mode: c++; -*-
#pragma once
// ADR Update: UpdateMARLADR

void UpdateMARLADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi) {
    // ========================================
    // CRITICAL: In low-congestion scenarios, SKIP all RL processing
    // ========================================
    if (globalAppPeriodSeconds >= 500.0) {
        DeviceMetrics& metrics = deviceMetrics[nodeId];
        metrics.packetsSent++;
        if (packetSuccess) {
            metrics.packetsReceived++;
        }
        return;
    }
    
    // Initialize MARL agent if needed
    if (marlAgents.find(nodeId) == marlAgents.end()) {
        marlAgents[nodeId] = std::make_unique<MARLAgent>();
    }
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    DeviceHistory& history = deviceHistories[nodeId];
    double currentTime = Simulator::Now().GetSeconds();
    
    // Update history with new observation
    history.addRssi(preTxRssi);
    history.addSnr(snr);
    history.addPacketResult(packetSuccess, currentTime);
    
    // Update global channel tracking for coordination
    deviceChannels[nodeId] = history.currentChannel;
    
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
    
    // ========================================
    // DISTANCE-AWARE BEHAVIOR (from DDQN)
    // ========================================
    double distance = 1000.0;  // Default to medium range
    if (nodePositions.find(nodeId) != nodePositions.end()) {
        Vector nodePos = nodePositions[nodeId];
        // Gateway is at origin (0, 0, 15)
        distance = std::sqrt(nodePos.x * nodePos.x + nodePos.y * nodePos.y);
    }
    
    // Distance-based behavior flags
    bool isShortRange = (distance < 500.0);
    bool isMediumRange = (distance >= 500.0 && distance < 1200.0);
    bool isLongRange = (distance >= 1200.0);
    
    // ========================================
    // COLLISION DETECTION (from DDQN)
    // ========================================
    CollisionIndicators collision = CollisionIndicators::analyze(
        preTxRssi,
        snr,
        packetSuccess,
        history.getRssiVariance(),
        history.consecutiveLosses,
        history.getRecentPdr()
    );
    
    // Collect all device histories, SFs, and channels for coordination
    std::map<uint32_t, uint8_t> allSFs;
    for (const auto& kv : currentSF) {
        allSFs[kv.first] = kv.second;
    }
    
    // Build coordinated state vector
    std::vector<double> currentState = marlAgents[nodeId]->buildCoordinatedState(
        history, currentSFVal, currentTPVal, currentChannel, currentTime,
        nodeId, deviceHistories, allSFs, deviceChannels
    );
    
    // ========================================
    // CONGESTION-ADAPTIVE Reward Calculation for MARL
    // Controlled by g_rewardSchedule:
    //   0 = adaptive (default): weights vary with congestion level C
    //   1 = uniform fixed:  w1=w3=w4=0.333, w2=0 (no interference term)
    //   2 = outcome-only:   w1=1, w2=w3=w4=0
    // ========================================
    double reward = 0.0;
    int sameChannelCount = 0;
    for (const auto& kv : deviceChannels) {
        if (kv.first != nodeId && kv.second == currentChannel) {
            sameChannelCount++;
        }
    }

    // --- select weights based on schedule ---
    double outcomeWeight, efficiencyWeight, stabilityWeight, interferenceWeight;
    if (g_rewardSchedule == 2) {          // outcome-only
        outcomeWeight     = 1.0;
        efficiencyWeight  = 0.0;
        stabilityWeight   = 0.0;
        interferenceWeight = 0.0;
    } else if (g_rewardSchedule == 1) {   // uniform fixed (w2=0)
        outcomeWeight     = 1.0 / 3.0;
        efficiencyWeight  = 1.0 / 3.0;
        stabilityWeight   = 1.0 / 3.0;
        interferenceWeight = 0.0;
    } else {                               // adaptive (default)
        outcomeWeight     = 0.4 + 0.3 * (1.0 - congestionLevel);   // 0.4–0.7
        efficiencyWeight  = 0.15 + 0.2 * (1.0 - congestionLevel);  // 0.15–0.35
        stabilityWeight   = 0.2 + 0.15 * (1.0 - congestionLevel);  // 0.2–0.35
        interferenceWeight = congestionLevel * 0.25;                // 0.0–0.25
    }

    if (packetSuccess) {
        reward = 1.0 * outcomeWeight / 0.4;
        
        // Interference avoidance bonus (w2): only when interferenceWeight > 0
        if (interferenceWeight > 0.0 && congestionLevel > 0.3) {
            reward += (3 - std::min(3, sameChannelCount)) * 0.1 * (interferenceWeight / 0.25);
        }
        
        // Distance-aware efficiency bonus (w4)
        if (efficiencyWeight > 0.0) {
            if (isShortRange) {
                reward += (12.0 - currentSFVal) * 0.15 * efficiencyWeight / 0.15;
                if (currentSFVal == 7) reward += 0.3 * efficiencyWeight / 0.15;
                if (currentSFVal == 8) reward += 0.15 * efficiencyWeight / 0.15;
            } else if (isMediumRange) {
                reward += (12.0 - currentSFVal) * 0.08 * efficiencyWeight / 0.15;
            } else {
                reward += (12.0 - currentSFVal) * 0.03;
            }
        }

        // Stability bonus (w3)
        if (stabilityWeight > 0.0) {
            double stabilityThreshold = (g_rewardSchedule == 0) ? 0.3 : 0.5;
            if (congestionLevel < stabilityThreshold && currentSFVal == 7 && history.getRecentPdr() > 0.7) {
                reward += 0.4 * stabilityWeight / 0.2;
            }
        }
        
        // Bonus for good SNR margin (schedule-independent)
        double lastSnr = history.getLastSnr();
        static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        if (lastSnr > -100.0 && currentSFVal >= 7 && currentSFVal <= 12) {
            double requiredSnr = REQUIRED_SNR[currentSFVal - 7];
            double snrMargin = lastSnr - requiredSnr;
            if (snrMargin > 15.0 && isShortRange) {
                reward += 0.2;
            }
        }
    } else {
        reward = -0.5 * outcomeWeight / 0.4;
        
        // Extra penalty if on congested channel (interference term)
        if (interferenceWeight > 0.0 && congestionLevel > 0.3) {
            reward -= sameChannelCount * 0.15 * (interferenceWeight / 0.25);
        }
        reward -= history.consecutiveLosses * 0.1 * (0.5 + 0.5 * congestionLevel);
        
        // Collision-based diagnosis (only relevant in high congestion)
        if (congestionLevel > 0.3) {
            if (collision.likelyCollision && isShortRange) {
                reward -= 0.1;
            } else if (collision.likelyWeakSignal && isShortRange) {
                reward -= 0.3;
            }
        }
        
        // Penalty for high SNR + High SF + Loss
        double lastSnr = history.getLastSnr();
        if (lastSnr > 20.0 && currentSFVal > 8 && isShortRange && congestionLevel > 0.3) {
            reward -= 0.25;
        }
    }
    
    // Store experience
    static std::map<uint32_t, std::vector<double>> previousMARLStates;
    static std::map<uint32_t, int> previousMARLActions;
    
    if (previousMARLStates.find(nodeId) != previousMARLStates.end() && metrics.packetsSent >= 2) {
        marlAgents[nodeId]->addExperience(
            previousMARLStates[nodeId], previousMARLActions[nodeId], 
            reward, currentState, false
        );
        marlAgents[nodeId]->train();
    }
    
    // ========================================
    // MANDATORY SF CEILING CHECK (before action selection)
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
                    std::cout << "🚨 MARL IMMEDIATE SF FIX: Node " << nodeId 
                              << " SF" << (int)currentSFVal << "→SF" << (int)correctedSF << std::endl;
                }
            }
        }
        currentSFVal = correctedSF;
    }
    
    // Select action using coordination-aware policy
    int action = marlAgents[nodeId]->selectAction(currentState, currentChannel, deviceChannels, nodeId);
    
    // Decode action from extended action space
    ExtendedActionSpace::ActionResult actionResult = ExtendedActionSpace::decodeAction(
        action, currentSFVal, currentTPVal, currentChannel
    );
    
    // Apply SF ceiling to MARL's proposed SF
    if (actionResult.newSF != -1) {
        actionResult.newSF = applySFCeiling(actionResult.newSF, snr, nodeId, "MARL");
    }
    
    double lastSnr = history.getLastSnr();
    
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
        actionResult.newSF = 7;
    }
    else if (effectiveSnr > REQUIRED_SNR_SF7 + 10.0) {  // SNR > 2.5dB
        forceSF7 = true;
        actionResult.newSF = 7;  // ALWAYS set to 7
        std::cout << "🔒 MARL FORCE SF7: SNR=" << effectiveSnr << "dB > 2.5dB required"
                  << " (Node " << nodeId << ")" << std::endl;
    } else if (effectiveSnr > REQUIRED_SNR_SF7 + 5.0) {  // SNR > -2.5dB
        if (actionResult.newSF == -1 || actionResult.newSF > 8) {
            actionResult.newSF = 8;
        }
    }
    
    // ========================================
    // Secondary: LOW CONGESTION override
    // ========================================
    if (!forceSF7 && congestionLevel < 0.2 && history.getRecentPdr() > 0.6 && actionResult.newSF > 8) {
        actionResult.newSF = 7;
        actionResult.forceChannelHop = false;
        actionResult.channelDelta = 0;
    }
    
    // ========================================
    // SNR-BASED SF CEILING for edge cases (skip if already forced SF7)
    // ========================================
    if (!forceSF7 && effectiveSnr > -100.0 && congestionLevel > 0.3) {
        static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        uint8_t snrOptimalSF = 12;
        
        for (int sf = 7; sf <= 12; sf++) {
            // Use stricter margin at short range
            double margin = isShortRange ? 8.0 : 10.0;
            if (lastSnr > REQUIRED_SNR[sf - 7] + margin) {
                snrOptimalSF = sf;
                break;
            }
        }
        
        // Apply stricter ceiling at short range
        uint8_t sfCeiling;
        if (isShortRange) {
            sfCeiling = snrOptimalSF;  // No buffer at short range - enforce optimal
        } else if (isMediumRange) {
            sfCeiling = std::min((uint8_t)12, (uint8_t)(snrOptimalSF + 1));
        } else {
            sfCeiling = std::min((uint8_t)12, (uint8_t)(snrOptimalSF + 2));  // More flexibility at long range
        }
        
        if (actionResult.newSF != -1 && actionResult.newSF > sfCeiling) {
            std::cout << "🎯 MARL SNR-cap: SF " << (int)actionResult.newSF 
                      << " → " << (int)sfCeiling
                      << " (SNR=" << lastSnr << "dB, optimal=" << (int)snrOptimalSF 
                      << ", dist=" << (int)distance << "m, cong=" << congestionLevel << ")" 
                      << " (Node " << nodeId << ")" << std::endl;
            actionResult.newSF = sfCeiling;
        }
        
        // High SNR + High SF + Low PDR detection - force reduction
        if (lastSnr > 25.0 && currentSFVal > 8 && history.getRecentPdr() < 0.6 && isShortRange) {
            std::cout << "🔄 MARL forcing SF reduction: SNR=" << lastSnr 
                      << "dB but SF=" << (int)currentSFVal 
                      << " with PDR=" << (history.getRecentPdr()*100) << "%" 
                      << " (Node " << nodeId << ")" << std::endl;
            actionResult.newSF = std::max((uint8_t)7, snrOptimalSF);
        }
    }
    
    // Distance-aware SF floor (prevent too high SF at short range) - only in high congestion
    if (isShortRange && actionResult.newSF > 9 && congestionLevel > 0.3) {
        // At short range, SF9+ is almost never optimal
        if (lastSnr > 15.0) {
            actionResult.newSF = std::min((int)actionResult.newSF, 8);
            std::cout << "🎯 MARL short-range SF floor: capped to SF8 (Node " << nodeId << ")" << std::endl;
        }
    }
    
    // Debug output with congestion info
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🤝 MARL Node " << nodeId 
                  << ": PDR=" << std::fixed << std::setprecision(1) << history.getRecentPdr()*100 << "%"
                  << ", SF=" << (int)currentSFVal 
                  << ", CH=" << (int)currentChannel
                  << ", Peers=" << sameChannelCount
                  << ", Dist=" << (int)distance << "m"
                  << ", CONG=" << std::setprecision(2) << congestionLevel
                  << ", ε=" << std::setprecision(3) << marlAgents[nodeId]->getEpsilon() << std::endl;
    }
    
    // Apply SF change - ALWAYS apply when forceSF7
    if (actionResult.newSF != -1 && (actionResult.newSF != currentSFVal || forceSF7)) {
        currentSF[nodeId] = actionResult.newSF;
        
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
                    std::cout << "✅ MARL SF=" << (int)actionResult.newSF 
                              << " APPLIED to MAC+PHY (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }
    
    // Apply TP change
    if (actionResult.newTP != -1 && std::abs(actionResult.newTP - currentTPVal) > 0.5) {
        double safeTp = std::max(8.0, std::min(14.0, (double)actionResult.newTP));
        currentTP[nodeId] = safeTp;
        nodeTxPowers[nodeId] = safeTp;
        
        if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
            uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
            if (deviceIndex < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[deviceIndex]->GetMac());
                if (edMac) {
                    edMac->SetTransmissionPowerDbm(safeTp);
                    std::cout << "✅ MARL TP=" << safeTp 
                              << "dBm (Node " << nodeId << ", device " << deviceIndex << ")" << std::endl;
                }
            }
        }
    }
    
    // Apply channel change with coordination awareness
    if (actionResult.forceChannelHop || actionResult.channelDelta != 0) {
        uint8_t newChannel;
        if (actionResult.forceChannelHop) {
            // Find least congested channel
            std::vector<int> channelCounts(8, 0);
            for (const auto& kv : deviceChannels) {
                if (kv.first != nodeId) {
                    channelCounts[kv.second]++;
                }
            }
            newChannel = 0;
            for (int i = 1; i < 8; i++) {
                if (channelCounts[i] < channelCounts[newChannel]) {
                    newChannel = i;
                }
            }
        } else {
            newChannel = (currentChannel + actionResult.channelDelta + 8) % 8;
        }
        
        history.setChannel(newChannel);
        deviceChannels[nodeId] = newChannel;
        std::cout << "📡 MARL Channel " << (int)currentChannel << "→" << (int)newChannel 
                  << " (Node " << nodeId << ")" << std::endl;
    }
    
    // ENHANCED: Emergency recovery for short-range devices
    if (metrics.packetsSent >= 10 && history.getRecentPdr() < 0.3) {
        if (history.consecutiveLosses > 3) {
            marlAgents[nodeId]->increaseExploration();
            std::cout << "🔄 MARL exploration boost (Node " << nodeId << ")" << std::endl;
            
            // NEW: At short range with poor PDR, force SF reduction + channel hop
            if (isShortRange && currentSFVal > 8) {
                currentSF[nodeId] = 7;
                if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
                    uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
                    if (deviceIndex < endDevicesNetDevices.size()) {
                        Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                            endDevicesNetDevices[deviceIndex]->GetMac());
                        if (edMac) {
                            edMac->SetDataRate(5);  // SF7
                            std::cout << "🚨 MARL emergency SF7 reset (short-range, Node " << nodeId << ")" << std::endl;
                        }
                    }
                }
                
                // Also hop to least congested channel
                std::vector<int> channelCounts(8, 0);
                for (const auto& kv : deviceChannels) {
                    if (kv.first != nodeId) {
                        channelCounts[kv.second]++;
                    }
                }
                uint8_t bestChannel = 0;
                for (int i = 1; i < 8; i++) {
                    if (channelCounts[i] < channelCounts[bestChannel]) {
                        bestChannel = i;
                    }
                }
                history.setChannel(bestChannel);
                deviceChannels[nodeId] = bestChannel;
                std::cout << "🚨 MARL emergency channel hop to " << (int)bestChannel 
                          << " (Node " << nodeId << ")" << std::endl;
            }
        }
    }
    
    previousMARLStates[nodeId] = currentState;
    previousMARLActions[nodeId] = action;
}

/**
 * Update function for MARL-PPO ADR
 *
 * Design: Independent PPO per device, augmented with the same 12-dim
 * coordinated state (includes n_same_ch/3 coordination signal) and
 * coordination-aware channel-hop pre-selection as MARL-DDQN.
 * Uses softmax actor + GAE-based clipped PPO objective instead of
 * Q-network + PER replay.
 */
