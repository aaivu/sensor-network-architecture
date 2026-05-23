// -*- mode: c++; -*-
#pragma once
// ADR Update: UpdateMAPPOGATADR

void UpdateMAPPOGATADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi) {
    // In low-congestion scenarios skip RL processing (same policy as other agents)
    if (globalAppPeriodSeconds >= 500.0) {
        DeviceMetrics& metrics = deviceMetrics[nodeId];
        metrics.packetsSent++;
        if (packetSuccess) metrics.packetsReceived++;
        return;
    }

    if (mappoGatAgents.find(nodeId) == mappoGatAgents.end()) return;

    DeviceMetrics& metrics   = deviceMetrics[nodeId];
    DeviceHistory& history   = deviceHistories[nodeId];
    double currentTime       = Simulator::Now().GetSeconds();

    history.addRssi(preTxRssi);
    history.addSnr(snr);
    history.addPacketResult(packetSuccess, currentTime);
    deviceChannels[nodeId] = history.currentChannel;

    uint8_t currentSFVal  = currentSF[nodeId];
    double  currentTPVal  = currentTP[nodeId];
    uint8_t currentChannel = history.currentChannel;

    // ── Congestion estimation ──────────────────────────────────────────
    double congestionLevel = estimateCongestionLevel(
        preTxRssi, history.getRssiVariance(), history.getRecentPdr(),
        deviceMetrics.size(), globalAppPeriodSeconds);

    // ── Build local state vector (12 dims, identical to other agents) ──
    double timeSinceLast = currentTime - history.lastSuccessTime;
    std::vector<double> localState = {
        std::max(0.0, std::min(1.0, (history.getLastRssi()  + 120.0) / 40.0)),
        std::max(0.0, std::min(1.0,  history.getRssiVariance() / 20.0)),
        std::max(0.0, std::min(1.0, (history.getLastSnr()   + 20.0)  / 50.0)),
        std::max(0.0, std::min(1.0, (history.getSnrTrend()  + 5.0)   / 10.0)),
        history.getRecentPdr(),
        std::max(0.0, std::min(1.0, (history.getPdrTrend()  + 0.5))),
        std::max(0.0, std::min(1.0,  history.consecutiveLosses / 5.0)),
        std::max(0.0, std::min(1.0,  timeSinceLast / 300.0)),
        (currentSFVal - 7.0) / 5.0,
        (currentTPVal - 2.0) / 12.0,
        currentChannel / 7.0,
        std::fmod(currentTime / 3600.0, 24.0) / 24.0
    };

    // ── Collect neighbour states (devices on the same channel) ─────────
    std::vector<std::vector<double>> neighborStates;
    for (const auto& kv : deviceHistories) {
        uint32_t otherId = kv.first;
        if (otherId == nodeId) continue;
        if (deviceChannels.count(otherId) && deviceChannels.at(otherId) == currentChannel) {
            const DeviceHistory& oh = kv.second;
            double otherTime = currentTime - oh.lastSuccessTime;
            uint8_t otherSF  = currentSF.count(otherId) ? currentSF.at(otherId) : 7;
            double  otherTP  = currentTP.count(otherId) ? currentTP.at(otherId) : 14.0;
            uint8_t otherCH  = deviceChannels.count(otherId) ? deviceChannels.at(otherId) : 0;
            neighborStates.push_back({
                std::max(0.0, std::min(1.0, (oh.getLastRssi()  + 120.0) / 40.0)),
                std::max(0.0, std::min(1.0,  oh.getRssiVariance() / 20.0)),
                std::max(0.0, std::min(1.0, (oh.getLastSnr()   + 20.0)  / 50.0)),
                std::max(0.0, std::min(1.0, (oh.getSnrTrend()  + 5.0)   / 10.0)),
                oh.getRecentPdr(),
                std::max(0.0, std::min(1.0, (oh.getPdrTrend()  + 0.5))),
                std::max(0.0, std::min(1.0,  oh.consecutiveLosses / 5.0)),
                std::max(0.0, std::min(1.0,  otherTime / 300.0)),
                (otherSF - 7.0) / 5.0,
                (otherTP - 2.0) / 12.0,
                otherCH / 7.0,
                std::fmod(currentTime / 3600.0, 24.0) / 24.0
            });
        }
    }

    // ── Collect all states for centralized critic ──────────────────────
    std::vector<std::vector<double>> allStates;
    for (const auto& kv : deviceHistories) {
        if (kv.first == nodeId) { allStates.push_back(localState); continue; }
        const DeviceHistory& oh = kv.second;
        double otherTime = currentTime - oh.lastSuccessTime;
        uint8_t otherSF  = currentSF.count(kv.first) ? currentSF.at(kv.first) : 7;
        double  otherTP  = currentTP.count(kv.first) ? currentTP.at(kv.first) : 14.0;
        uint8_t otherCH  = deviceChannels.count(kv.first) ? deviceChannels.at(kv.first) : 0;
        allStates.push_back({
            std::max(0.0, std::min(1.0, (oh.getLastRssi()  + 120.0) / 40.0)),
            std::max(0.0, std::min(1.0,  oh.getRssiVariance() / 20.0)),
            std::max(0.0, std::min(1.0, (oh.getLastSnr()   + 20.0)  / 50.0)),
            std::max(0.0, std::min(1.0, (oh.getSnrTrend()  + 5.0)   / 10.0)),
            oh.getRecentPdr(),
            std::max(0.0, std::min(1.0, (oh.getPdrTrend()  + 0.5))),
            std::max(0.0, std::min(1.0,  oh.consecutiveLosses / 5.0)),
            std::max(0.0, std::min(1.0,  otherTime / 300.0)),
            (otherSF - 7.0) / 5.0,
            (otherTP - 2.0) / 12.0,
            otherCH / 7.0,
            std::fmod(currentTime / 3600.0, 24.0) / 24.0
        });
    }

    // ── Build actor / critic inputs ────────────────────────────────────
    std::vector<double> gatInput    = mappoGatAgents[nodeId]->buildGATInput(localState, neighborStates);
    std::vector<double> criticInput = mappoGatAgents[nodeId]->buildCriticInput(localState, allStates);

    // ── Congestion-adaptive reward (reuse helper from MARL) ─────────────
    int sameChannelCount = 0;
    for (const auto& kv : deviceChannels)
        if (kv.first != nodeId && kv.second == currentChannel) sameChannelCount++;

    double distance = 1000.0;
    if (nodePositions.count(nodeId)) {
        Vector p = nodePositions.at(nodeId);
        distance = std::sqrt(p.x * p.x + p.y * p.y);
    }
    bool isShortRange = (distance < 500.0);

    double outcomeWeight = 0.4 + 0.3 * (1.0 - congestionLevel);
    double reward = 0.0;

    if (packetSuccess) {
        reward = 1.0 * outcomeWeight / 0.4;
        if (congestionLevel > 0.3)
            reward += (3 - std::min(3, sameChannelCount)) * 0.1;
        double effW = 0.15 + 0.2 * (1.0 - congestionLevel);
        if (isShortRange) {
            reward += (12.0 - currentSFVal) * 0.15 * effW / 0.15;
            if (currentSFVal == 7) reward += 0.3;
        } else {
            reward += (12.0 - currentSFVal) * 0.05 * effW / 0.15;
        }
        if (congestionLevel < 0.3 && currentSFVal == 7 && history.getRecentPdr() > 0.7)
            reward += 0.4;
    } else {
        reward = -0.5 * outcomeWeight / 0.4;
        if (congestionLevel > 0.3) reward -= sameChannelCount * 0.15;
        reward -= history.consecutiveLosses * 0.1 * (0.5 + 0.5 * congestionLevel);
    }

    // ── Mandatory SF ceiling ───────────────────────────────────────────
    uint8_t mandatorySFCeiling = getMandatorySFCeiling(snr);
    if (currentSFVal > mandatorySFCeiling + 1) {
        uint8_t correctedSF = mandatorySFCeiling;
        currentSF[nodeId] = correctedSF;
        if (nodeIdToDeviceIndex.count(nodeId)) {
            uint32_t di = nodeIdToDeviceIndex.at(nodeId);
            if (di < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[di]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[di]->GetPhy());
                if (edMac && edPhy) {
                    edMac->SetDataRate(12 - correctedSF);
                    edPhy->SetSpreadingFactor(correctedSF);
                    std::cout << "🚨 MAPPO-GAT IMMEDIATE SF FIX: Node " << nodeId
                              << " SF" << (int)currentSFVal << "→SF" << (int)correctedSF << std::endl;
                }
            }
        }
        currentSFVal = correctedSF;
    }

    // ── Store previous transition, then select new action ─────────────
    static std::map<uint32_t, std::vector<double>> prevGATInputs;
    static std::map<uint32_t, std::vector<double>> prevCriticInputs;
    static std::map<uint32_t, int>                 prevActions;
    static std::map<uint32_t, double>              prevLogProbs;

    if (prevGATInputs.count(nodeId) && metrics.packetsSent >= 2) {
        mappoGatAgents[nodeId]->storeTransition(
            prevGATInputs[nodeId], prevCriticInputs[nodeId],
            prevActions[nodeId], prevLogProbs[nodeId],
            reward, false);
    }

    double logProb = 0.0;
    int action = mappoGatAgents[nodeId]->selectAction(gatInput, logProb);

    // Decode action using the same ExtendedActionSpace as MARL
    ExtendedActionSpace::ActionResult ar = ExtendedActionSpace::decodeAction(
        action, currentSFVal, currentTPVal, currentChannel);

    // Apply SF ceiling to proposed SF
    if (ar.newSF != -1) ar.newSF = applySFCeiling(ar.newSF, snr, nodeId, "MAPPO-GAT");

    // Unconditional SF7 enforcement (mirrors MARL / PPO logic)
    double effectiveSnr = (snr > -50.0) ? snr : history.getLastSnr();
    bool forceSF7 = false;

    if (globalAppPeriodSeconds >= 100.0) {
        forceSF7 = true; ar.newSF = 7;
    } else if (effectiveSnr > -7.5 + 10.0) {
        forceSF7 = true; ar.newSF = 7;
        std::cout << "🔒 MAPPO-GAT FORCE SF7: SNR=" << effectiveSnr
                  << "dB (Node " << nodeId << ")" << std::endl;
    } else if (effectiveSnr > -7.5 + 5.0) {
        if (ar.newSF == -1 || ar.newSF > 8) ar.newSF = 8;
    }

    if (!forceSF7 && congestionLevel < 0.2 &&
        history.getRecentPdr() > 0.6 && ar.newSF > 8) {
        ar.newSF = 7;
    }

    // Apply SF change
    if (ar.newSF != -1 && (ar.newSF != (int)currentSFVal || forceSF7)) {
        currentSF[nodeId] = ar.newSF;
        if (nodeIdToDeviceIndex.count(nodeId)) {
            uint32_t di = nodeIdToDeviceIndex.at(nodeId);
            if (di < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[di]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[di]->GetPhy());
                if (edMac && edPhy) {
                    edMac->SetDataRate(12 - ar.newSF);
                    edPhy->SetSpreadingFactor(ar.newSF);
                    std::cout << "✅ MAPPO-GAT SF=" << ar.newSF
                              << " APPLIED (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }

    // Apply TP change
    if (ar.newTP != -1 && std::abs(ar.newTP - currentTPVal) > 0.5) {
        double safeTp = std::max(8.0, std::min(14.0, (double)ar.newTP));
        currentTP[nodeId] = safeTp;
        nodeTxPowers[nodeId] = safeTp;
        if (nodeIdToDeviceIndex.count(nodeId)) {
            uint32_t di = nodeIdToDeviceIndex.at(nodeId);
            if (di < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[di]->GetMac());
                if (edMac) {
                    edMac->SetTransmissionPowerDbm(safeTp);
                    std::cout << "✅ MAPPO-GAT TP=" << safeTp
                              << "dBm (Node " << nodeId << ")" << std::endl;
                }
            }
        }
    }

    // Apply channel change (via logical hop, only in high-congestion mode)
    if (globalAppPeriodSeconds < 100.0 && congestionLevel > 0.3 &&
        (ar.forceChannelHop || ar.channelDelta != 0)) {
        uint8_t newChannel;
        if (ar.forceChannelHop) {
            std::vector<int> counts(8, 0);
            for (const auto& kv : deviceChannels)
                if (kv.first != nodeId) counts[kv.second]++;
            newChannel = 0;
            for (int i = 1; i < 8; i++)
                if (counts[i] < counts[newChannel]) newChannel = i;
        } else {
            newChannel = (currentChannel + ar.channelDelta + 8) % 8;
        }
        history.setChannel(newChannel);
        deviceChannels[nodeId] = newChannel;
        std::cout << "📡 MAPPO-GAT Channel " << (int)currentChannel
                  << "→" << (int)newChannel << " (Node " << nodeId << ")" << std::endl;
    }

    // Debug output
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🌐 MAPPO-GAT Node " << nodeId
                  << ": PDR=" << std::fixed << std::setprecision(1)
                  << history.getRecentPdr() * 100 << "%"
                  << ", SF="  << (int)currentSFVal
                  << ", CH="  << (int)currentChannel
                  << ", Nbrs=" << neighborStates.size()
                  << ", CONG=" << std::setprecision(2) << congestionLevel
                  << ", ε="   << std::setprecision(3)
                  << mappoGatAgents[nodeId]->getEpsilon() << std::endl;
    }

    // Periodic PPO network update
    if (metrics.packetsSent % 32 == 0)
        mappoGatAgents[nodeId]->update();

    // Emergency exploration for stuck devices
    if (metrics.packetsSent >= 10 && history.getRecentPdr() < 0.3 &&
        history.consecutiveLosses > 3)
        mappoGatAgents[nodeId]->increaseExploration();

    prevGATInputs[nodeId]    = gatInput;
    prevCriticInputs[nodeId] = criticInput;
    prevActions[nodeId]      = action;
    prevLogProbs[nodeId]     = logProb;
}

