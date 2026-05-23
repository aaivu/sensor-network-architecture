// -*- mode: c++; -*-
#pragma once
// ADR Update: UpdateMARLPPOADR

void UpdateMARLPPOADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi) {
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

    // Initialize MARL-PPO agent if needed
    if (marlPpoAgents.find(nodeId) == marlPpoAgents.end()) {
        marlPpoAgents[nodeId] = std::make_unique<MARLPPOAgent>();
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
    double distance = 1000.0;
    if (nodePositions.find(nodeId) != nodePositions.end()) {
        Vector nodePos = nodePositions[nodeId];
        distance = std::sqrt(nodePos.x * nodePos.x + nodePos.y * nodePos.y);
    }

    bool isShortRange  = (distance < 500.0);
    bool isMediumRange = (distance >= 500.0 && distance < 1200.0);
    bool isLongRange   = (distance >= 1200.0);

    // ========================================
    // COLLISION DETECTION (from DDQN)
    // ========================================
    CollisionIndicators collision = CollisionIndicators::analyze(
        preTxRssi, snr, packetSuccess,
        history.getRssiVariance(),
        history.consecutiveLosses,
        history.getRecentPdr()
    );

    // Collect all device SFs for coordination state
    std::map<uint32_t, uint8_t> allSFs;
    for (const auto& kv : currentSF) {
        allSFs[kv.first] = kv.second;
    }

    // Build coordinated state vector (12-dim, includes coordination signal)
    std::vector<double> currentState = marlPpoAgents[nodeId]->buildCoordinatedState(
        history, currentSFVal, currentTPVal, currentChannel, currentTime,
        nodeId, deviceHistories, allSFs, deviceChannels
    );

    // ========================================
    // CONGESTION-ADAPTIVE Reward Calculation (mirrors MARL exactly)
    // ========================================
    double reward = 0.0;
    int sameChannelCount = 0;
    for (const auto& kv : deviceChannels) {
        if (kv.first != nodeId && kv.second == currentChannel) {
            sameChannelCount++;
        }
    }

    double outcomeWeight = 0.4 + 0.3 * (1.0 - congestionLevel);

    if (packetSuccess) {
        reward = 1.0 * outcomeWeight / 0.4;

        if (congestionLevel > 0.3) {
            reward += (3 - std::min(3, sameChannelCount)) * 0.1;
        }

        double efficiencyWeight = 0.15 + 0.2 * (1.0 - congestionLevel);
        if (isShortRange) {
            reward += (12.0 - currentSFVal) * 0.15 * efficiencyWeight / 0.15;
            if (currentSFVal == 7) reward += 0.3;
            if (currentSFVal == 8) reward += 0.15;
        } else if (isMediumRange) {
            reward += (12.0 - currentSFVal) * 0.08 * efficiencyWeight / 0.15;
        } else {
            reward += (12.0 - currentSFVal) * 0.03;
        }

        if (congestionLevel < 0.3 && currentSFVal == 7 && history.getRecentPdr() > 0.7) {
            reward += 0.4;
        }

        double lastSnrR = history.getLastSnr();
        static const double REQUIRED_SNR_R[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        if (lastSnrR > -100.0 && currentSFVal >= 7 && currentSFVal <= 12) {
            double requiredSnr = REQUIRED_SNR_R[currentSFVal - 7];
            double snrMargin = lastSnrR - requiredSnr;
            if (snrMargin > 15.0 && isShortRange) {
                reward += 0.2;
            }
        }
    } else {
        reward = -0.5 * outcomeWeight / 0.4;

        if (congestionLevel > 0.3) {
            reward -= sameChannelCount * 0.15;
        }
        reward -= history.consecutiveLosses * 0.1 * (0.5 + 0.5 * congestionLevel);

        if (congestionLevel > 0.3) {
            if (collision.likelyCollision && isShortRange) {
                reward -= 0.1;
            } else if (collision.likelyWeakSignal && isShortRange) {
                reward -= 0.3;
            }
        }

        double lastSnrR = history.getLastSnr();
        if (lastSnrR > 20.0 && currentSFVal > 8 && isShortRange && congestionLevel > 0.3) {
            reward -= 0.25;
        }
    }

    // ========================================
    // PPO trajectory store + update
    // ========================================
    static std::map<uint32_t, std::vector<double>> previousMARLPPOStates;
    static std::map<uint32_t, int>    previousMARLPPOActions;
    static std::map<uint32_t, double> previousMARLPPOLogProbs;

    if (previousMARLPPOStates.find(nodeId) != previousMARLPPOStates.end() && metrics.packetsSent >= 2) {
        marlPpoAgents[nodeId]->storeTransition(
            previousMARLPPOStates[nodeId],
            previousMARLPPOActions[nodeId],
            previousMARLPPOLogProbs[nodeId],
            reward, false
        );
        marlPpoAgents[nodeId]->update();
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
                    std::cout << "🚨 MARL-PPO IMMEDIATE SF FIX: Node " << nodeId
                              << " SF" << (int)currentSFVal << "→SF" << (int)correctedSF << std::endl;
                }
            }
        }
        currentSFVal = correctedSF;
    }

    // Select action using coordination-aware PPO policy
    double logProb = 0.0;
    int action = marlPpoAgents[nodeId]->selectAction(
        currentState, currentChannel, deviceChannels, nodeId, logProb
    );

    // Decode action from extended action space
    ExtendedActionSpace::ActionResult actionResult = ExtendedActionSpace::decodeAction(
        action, currentSFVal, currentTPVal, currentChannel
    );

    // Apply SF ceiling to proposed SF
    if (actionResult.newSF != -1) {
        actionResult.newSF = applySFCeiling(actionResult.newSF, snr, nodeId, "MARL-PPO");
    }

    double lastSnr = history.getLastSnr();

    // ========================================
    // UNCONDITIONAL SNR-BASED SF7 ENFORCEMENT
    // ========================================
    static const double REQUIRED_SNR_SF7 = -7.5;
    double effectiveSnr = (snr > -50.0) ? snr : lastSnr;

    bool forceSF7 = false;

    if (globalAppPeriodSeconds >= 100.0) {
        forceSF7 = true;
        actionResult.newSF = 7;
    } else if (effectiveSnr > REQUIRED_SNR_SF7 + 10.0) {
        forceSF7 = true;
        actionResult.newSF = 7;
        std::cout << "🔒 MARL-PPO FORCE SF7: SNR=" << effectiveSnr << "dB > 2.5dB required"
                  << " (Node " << nodeId << ")" << std::endl;
    } else if (effectiveSnr > REQUIRED_SNR_SF7 + 5.0) {
        if (actionResult.newSF == -1 || actionResult.newSF > 8) {
            actionResult.newSF = 8;
        }
    }

    // Low-congestion override
    if (!forceSF7 && congestionLevel < 0.2 && history.getRecentPdr() > 0.6 && actionResult.newSF > 8) {
        actionResult.newSF = 7;
        actionResult.forceChannelHop = false;
        actionResult.channelDelta = 0;
    }

    // SNR-based SF ceiling for edge cases
    if (!forceSF7 && effectiveSnr > -100.0 && congestionLevel > 0.3) {
        static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        uint8_t snrOptimalSF = 12;

        for (int sf = 7; sf <= 12; sf++) {
            double margin = isShortRange ? 8.0 : 10.0;
            if (lastSnr > REQUIRED_SNR[sf - 7] + margin) {
                snrOptimalSF = sf;
                break;
            }
        }

        uint8_t sfCeiling;
        if (isShortRange) {
            sfCeiling = snrOptimalSF;
        } else if (isMediumRange) {
            sfCeiling = std::min((uint8_t)12, (uint8_t)(snrOptimalSF + 1));
        } else {
            sfCeiling = std::min((uint8_t)12, (uint8_t)(snrOptimalSF + 2));
        }

        if (actionResult.newSF != -1 && actionResult.newSF > sfCeiling) {
            std::cout << "🎯 MARL-PPO SNR-cap: SF " << (int)actionResult.newSF
                      << " → " << (int)sfCeiling
                      << " (SNR=" << lastSnr << "dB, Node " << nodeId << ")" << std::endl;
            actionResult.newSF = sfCeiling;
        }

        if (lastSnr > 25.0 && currentSFVal > 8 && history.getRecentPdr() < 0.6 && isShortRange) {
            actionResult.newSF = std::max((uint8_t)7, snrOptimalSF);
        }
    }

    // Distance-aware SF floor
    if (isShortRange && actionResult.newSF > 9 && congestionLevel > 0.3) {
        if (lastSnr > 15.0) {
            actionResult.newSF = std::min((int)actionResult.newSF, 8);
            std::cout << "🎯 MARL-PPO short-range SF floor: capped to SF8 (Node " << nodeId << ")" << std::endl;
        }
    }

    // Debug output
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🤝 MARL-PPO Node " << nodeId
                  << ": PDR=" << std::fixed << std::setprecision(1) << history.getRecentPdr()*100 << "%"
                  << ", SF=" << (int)currentSFVal
                  << ", CH=" << (int)currentChannel
                  << ", Peers=" << sameChannelCount
                  << ", Dist=" << (int)distance << "m"
                  << ", CONG=" << std::setprecision(2) << congestionLevel
                  << ", ε=" << std::setprecision(3) << marlPpoAgents[nodeId]->getEpsilon() << std::endl;
    }

    // Apply SF change
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
                    edPhy->SetSpreadingFactor(actionResult.newSF);
                    std::cout << "✅ MARL-PPO SF=" << (int)actionResult.newSF
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
                    std::cout << "✅ MARL-PPO TP=" << safeTp
                              << "dBm (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }

    // Apply channel change with coordination awareness
    if (actionResult.forceChannelHop || actionResult.channelDelta != 0) {
        uint8_t newChannel;
        if (actionResult.forceChannelHop) {
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
        std::cout << "📡 MARL-PPO Channel " << (int)currentChannel << "→" << (int)newChannel
                  << " (Node " << nodeId << ")" << std::endl;
    }

    // Emergency recovery
    if (metrics.packetsSent >= 10 && history.getRecentPdr() < 0.3) {
        if (history.consecutiveLosses > 3) {
            marlPpoAgents[nodeId]->increaseExploration();
            std::cout << "🔄 MARL-PPO exploration boost (Node " << nodeId << ")" << std::endl;

            if (isShortRange && currentSFVal > 8) {
                currentSF[nodeId] = 7;
                if (nodeIdToDeviceIndex.find(nodeId) != nodeIdToDeviceIndex.end()) {
                    uint32_t deviceIndex = nodeIdToDeviceIndex[nodeId];
                    if (deviceIndex < endDevicesNetDevices.size()) {
                        Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                            endDevicesNetDevices[deviceIndex]->GetMac());
                        if (edMac) {
                            edMac->SetDataRate(5);
                            std::cout << "🚨 MARL-PPO emergency SF7 reset (Node " << nodeId << ")" << std::endl;
                        }
                    }
                }

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
                std::cout << "🚨 MARL-PPO emergency channel hop to " << (int)bestChannel
                          << " (Node " << nodeId << ")" << std::endl;
            }
        }
    }

    previousMARLPPOStates[nodeId]   = currentState;
    previousMARLPPOActions[nodeId]  = action;
    previousMARLPPOLogProbs[nodeId] = logProb;
}

/**
 * Update function for MAPPO-GAT ADR
 *
 * Design (CTDE):
 *  - Actor input  : [local_state || GAT(on-channel neighbours)] (28 dims)
 *  - Critic input : [local_state || mean(all_states)]            (24 dims)
 *  - Action space : 48 discrete actions  (SF / TP / channel, same as MARL)
 *  - Neighbours   : devices sharing the same LoRa channel (collision-prone)
 *  - SF ceiling + SF7 enforcement mirror the MARL / PPO logic for fair comparison
 */
