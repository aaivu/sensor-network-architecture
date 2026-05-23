// -*- mode: c++; -*-
#pragma once
// Reward: congestion estimation, SF ceiling, reward functions, global ablation vars

double estimateCongestionLevel(
    double preTxRssi,
    double rssiVariance,
    double recentPdr,
    uint32_t activeDeviceCount,
    double appPeriod)
{
    double congestion = 0.0;
    
    // Factor 1: Network density is PRIMARY indicator
    // 20 devices at 60s = 0.33 pps = MEDIUM congestion
    // 30 devices at 10s = 3 pps = VERY HIGH congestion
    double trafficIntensity = activeDeviceCount / std::max(1.0, appPeriod);
    
    if (trafficIntensity > 2.0) {
        congestion += 0.6;  // Very high traffic
    } else if (trafficIntensity > 0.5) {
        congestion += 0.45;  // High traffic
    } else if (trafficIntensity > 0.2) {
        congestion += 0.35;  // Medium traffic (NEW - critical for 20 devices @ 60s)
    } else if (trafficIntensity > 0.1) {
        congestion += 0.2;  // Low-medium traffic
    }
    
    // Factor 2: Pre-TX RSSI indicates active transmitters
    if (preTxRssi > -100.0) {
        congestion += 0.2 * std::min(1.0, (preTxRssi + 110.0) / 10.0);
    } else if (preTxRssi > -105.0) {
        congestion += 0.1 * std::min(1.0, (preTxRssi + 115.0) / 15.0);
    }
    
    // Factor 3: RSSI variance indicates time-varying interference
    if (rssiVariance > 5.0) {
        congestion += 0.15 * std::min(1.0, rssiVariance / 15.0);
    }
    
    // Factor 4: Low PDR with good signal = collision indicator
    // This is the strongest indicator of congestion-based losses
    if (recentPdr < 0.3 && preTxRssi > -105.0) {
        congestion += 0.25;  // Strong collision indicator
    } else if (recentPdr < 0.5 && preTxRssi > -108.0) {
        congestion += 0.15;
    }
    
    return std::min(1.0, congestion);
}

/**
 * MANDATORY SF ceiling based on SNR - returns maximum SF allowed
 * Uses SNR requirements with margin for reliability
 */
uint8_t getMandatorySFCeiling(double snr) {
    // SNR requirements with margin for reliability
    // SF7: -7.5dB required + 10dB margin = SNR > 2.5dB
    // SF8: -10dB required + 10dB margin = SNR > 0dB
    // etc.
    static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
    static const double MARGIN = 10.0;
    
    if (snr < -50.0) return 12;  // Invalid SNR, allow any SF
    
    for (int sf = 7; sf <= 12; sf++) {
        if (snr > REQUIRED_SNR[sf - 7] + MARGIN) {
            return sf;  // This is the minimum SF that works reliably
        }
    }
    return 12;  // Poor SNR, need SF12
}

/**
 * Apply SF ceiling UNCONDITIONALLY - enforces SNR-based limits
 * Returns the enforced SF (may be lower than proposed)
 */
uint8_t applySFCeiling(uint8_t proposedSF, double snr, uint32_t nodeId, const std::string& agentName) {
    uint8_t optimalSF = getMandatorySFCeiling(snr);
    
    // Allow at most 1 SF above optimal (for margin)
    uint8_t ceiling = std::min((uint8_t)12, (uint8_t)(optimalSF + 1));
    
    if (proposedSF > ceiling) {
        std::cout << "🚫 " << agentName << " SF CEILING: Node " << nodeId 
                  << " proposed SF" << (int)proposedSF 
                  << " → SF" << (int)ceiling
                  << " (SNR=" << snr << "dB, optimal=SF" << (int)optimalSF << ")" << std::endl;
        return ceiling;
    }
    return proposedSF;
}

// Global app period for congestion estimation (set in main)
double globalAppPeriodSeconds = 30.0;

// ============================================================================
// ABLATION / SENSITIVITY EXPERIMENT CONTROLS
// These are set from command-line in main() so the Python runner can sweep
// them without recompiling.
//
//   --coordDivisor N   : denominator used when normalising the per-channel
//                        device-count coordination signal:
//                          n_same_channel / N   (default N = 3)
//   --rewardSchedule S : reward weight schedule for MARL-DDQN
//                          0 = congestion-adaptive  (paper default)
//                          1 = uniform fixed         w1=w3=w4=0.333, w2=0
//                          2 = outcome-only          w1=1, w2=w3=w4=0
//   --rngSeed K        : RNG seed passed to RngSeedManager::SetSeed() so
//                        multiple independent runs can be averaged.
// ============================================================================
double   g_coordDivisor   = 3.0;
int      g_rewardSchedule = 0;
uint32_t g_rngSeed        = 12345;

/**
 * CONGESTION-ADAPTIVE REWARD FUNCTION
 * Adjusts reward strategy based on detected network congestion level
 */
double calculateCongestionAdaptiveReward(
    uint32_t nodeId,
    bool packetSuccess,
    double preTxRssi,
    double rxSnr,
    double rssiVariance,
    uint32_t consecutiveLosses,
    uint8_t currentSF,
    double currentTP,
    uint8_t currentChannel,
    uint8_t previousChannel,
    double recentPdr,
    double pdrTrend,
    double timeSinceLastSuccess,
    double congestionLevel)  // 0.0 = no congestion, 1.0 = high congestion
{
    double reward = 0.0;
    
    // ========================================
    // COMPONENT 1: Packet Outcome (ALWAYS PRIMARY)
    // Weight increases when congestion is low
    // ========================================
    double outcomeWeight = 0.4 + 0.3 * (1.0 - congestionLevel);  // 0.4-0.7
    
    if (packetSuccess) {
        reward += 80.0 * outcomeWeight / 0.4;  // Scale to maintain magnitude
        
        // Bonus for recovery after losses
        if (consecutiveLosses > 0) {
            reward += std::min(50.0, (double)consecutiveLosses * 10.0);
        }
    } else {
        reward -= 40.0 * outcomeWeight / 0.4;
        
        // Extra penalty for repeated losses
        if (consecutiveLosses > 2) {
            reward -= std::min(30.0, (double)(consecutiveLosses - 2) * 10.0);
        }
    }
    
    // ========================================
    // COMPONENT 2: Interference Avoidance
    // ONLY SIGNIFICANT WHEN CONGESTION IS DETECTED
    // ========================================
    double interferenceWeight = congestionLevel * 0.25;  // 0.0-0.25
    
    bool highInterference = (preTxRssi > -100.0);
    bool changedChannel = (currentChannel != previousChannel);
    
    if (congestionLevel > 0.3) {  // Only apply when meaningful congestion
        if (highInterference) {
            if (changedChannel && packetSuccess) {
                reward += 40.0 * interferenceWeight / 0.25;
            } else if (!changedChannel && !packetSuccess) {
                reward -= 20.0 * interferenceWeight / 0.25;
            }
        }
        
        if (rssiVariance > 10.0 && packetSuccess) {
            reward += 15.0 * interferenceWeight / 0.25;
        }
    }
    
    // ========================================
    // COMPONENT 3: PDR Stability (MORE IMPORTANT WHEN LOW CONGESTION)
    // ========================================
    double stabilityWeight = 0.2 + 0.15 * (1.0 - congestionLevel);  // 0.2-0.35
    
    if (pdrTrend > 0.1) {
        reward += 30.0 * stabilityWeight / 0.2;
    } else if (pdrTrend < -0.1) {
        reward -= 20.0 * stabilityWeight / 0.2;
    }
    
    if (recentPdr >= 0.9) {
        reward += 25.0 * stabilityWeight / 0.2;
    } else if (recentPdr >= 0.7) {
        reward += 10.0 * stabilityWeight / 0.2;
    } else if (recentPdr < 0.3) {
        reward -= 25.0 * stabilityWeight / 0.2;
    }
    
    // ========================================
    // COMPONENT 4: Energy Efficiency
    // MORE IMPORTANT WHEN LOW CONGESTION (can afford to optimize)
    // ========================================
    double efficiencyWeight = 0.15 + 0.2 * (1.0 - congestionLevel);  // 0.15-0.35
    
    if (packetSuccess && recentPdr >= 0.6) {
        reward += (12 - currentSF) * 3.0 * efficiencyWeight / 0.15;
        reward += (14 - currentTP) / 2.0 * efficiencyWeight / 0.15;
    }
    
    // ========================================
    // COMPONENT 5: STABILITY BONUS (NEW)
    // Reward for NOT changing parameters when things are working
    // ========================================
    if (congestionLevel < 0.3 && packetSuccess && recentPdr > 0.7) {
        // Low congestion, successful, good PDR - reward stability
        if (!changedChannel && currentSF == 7) {
            reward += 20.0;  // Bonus for staying optimal
        }
    }
    
    // ========================================
    // COMPONENT 6: SNR-SF Mismatch Penalty (REDUCED WHEN LOW CONGESTION)
    // In low congestion, SF choices matter less
    // ========================================
    static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
    
    if (rxSnr > 15.0 && congestionLevel > 0.3) {
        uint8_t optimalSF = 7;
        for (int sf = 7; sf <= 12; sf++) {
            double required = REQUIRED_SNR[sf - 7];
            if (rxSnr > required + 10.0) {
                optimalSF = sf;
                break;
            }
        }
        
        int sfExcess = (int)currentSF - (int)optimalSF;
        if (sfExcess > 1) {
            double penalty = sfExcess * 20.0 * congestionLevel;  // Scale by congestion
            reward -= penalty;
        }
    }
    
    // ========================================
    // COMPONENT 7: Recovery Speed
    // ========================================
    if (timeSinceLastSuccess > 60.0 && !packetSuccess) {
        reward -= 30.0;
    }
    
    // ========================================
    // COMPONENT 8: SF-SPECIFIC SHAPING (NEW)
    // Strongly encourage appropriate SF for link quality
    // ========================================
    if (rxSnr > -50.0) {  // Valid SNR
        // Calculate optimal SF using getMandatorySFCeiling logic
        uint8_t optimalSF = 12;
        for (int sf = 7; sf <= 12; sf++) {
            if (rxSnr > REQUIRED_SNR[sf - 7] + 10.0) {
                optimalSF = sf;
                break;
            }
        }
        
        // Strong reward for using optimal or optimal+1 SF
        if (currentSF == optimalSF) {
            reward += 40.0;  // Perfect choice
        } else if (currentSF == optimalSF + 1) {
            reward += 20.0;  // Acceptable margin
        } else if (currentSF < optimalSF) {
            reward -= 10.0;  // Too aggressive (might fail at edge)
        } else {
            // Using SF too high - penalize proportionally
            int sfExcess = currentSF - optimalSF - 1;
            if (sfExcess > 0) {
                reward -= sfExcess * 25.0;  // -25 per unnecessary SF level
            }
        }
    }
    
    return reward;
}

/**
 * INTERFERENCE-AWARE REWARD FUNCTION (LEGACY - for comparison)
 * Rewards smart interference avoidance behavior
 */
double calculateInterferenceAwareReward(
    uint32_t nodeId,
    bool packetSuccess,
    double preTxRssi,
    double rxSnr,
    double rssiVariance,
    uint32_t consecutiveLosses,
    uint8_t currentSF,
    double currentTP,
    uint8_t currentChannel,
    uint8_t previousChannel,
    double recentPdr,
    double pdrTrend,
    double timeSinceLastSuccess)
{
    double reward = 0.0;
    
    // ========================================
    // COMPONENT 1: Packet Outcome (40%)
    // ========================================
    if (packetSuccess) {
        reward += 80.0;
        
        // Bonus for recovery after losses
        if (consecutiveLosses > 0) {
            reward += std::min(50.0, (double)consecutiveLosses * 10.0);
        }
    } else {
        reward -= 40.0;
        
        // Extra penalty for repeated losses
        if (consecutiveLosses > 2) {
            reward -= std::min(30.0, (double)(consecutiveLosses - 2) * 10.0);
        }
    }
    
    // ========================================
    // COMPONENT 2: Interference Avoidance (25%)
    // ========================================
    bool highInterference = (preTxRssi > -100.0);
    bool changedChannel = (currentChannel != previousChannel);
    
    if (highInterference) {
        if (changedChannel && packetSuccess) {
            // Successfully avoided interference
            reward += 40.0;
            std::cout << "🎯 Smart channel hop rewarded! Node " << nodeId << std::endl;
        } else if (!changedChannel && !packetSuccess) {
            // Detected interference but didn't avoid
            reward -= 20.0;
        }
    }
    
    // High RSSI variance = unstable environment
    if (rssiVariance > 10.0 && packetSuccess) {
        reward += 15.0;  // Managed to succeed despite variance
    }
    
    // ========================================
    // COMPONENT 3: PDR Trend (20%)
    // ========================================
    if (pdrTrend > 0.1) {
        reward += 30.0;  // PDR improving
    } else if (pdrTrend < -0.1) {
        reward -= 20.0;  // PDR degrading
    }
    
    if (recentPdr >= 0.9) {
        reward += 25.0;
    } else if (recentPdr >= 0.7) {
        reward += 10.0;
    } else if (recentPdr < 0.3) {
        reward -= 25.0;
    }
    
    // ========================================
    // COMPONENT 4: Energy Efficiency (15%)
    // ========================================
    if (packetSuccess && recentPdr >= 0.6) {
        reward += (12 - currentSF) * 3.0;
        reward += (14 - currentTP) / 2.0;
    }
    
    // ========================================
    // COMPONENT 5: Recovery Speed
    // ========================================
    if (timeSinceLastSuccess > 60.0 && !packetSuccess) {
        reward -= 30.0;
    }
    
    // ========================================
    // COMPONENT 6: SNR-SF Mismatch Penalty (NEW)
    // Strongly penalize high SF when SNR is excellent
    // ========================================
    static const double REQUIRED_SNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0}; // SF7-SF12
    
    if (rxSnr > 15.0) {  // Good link quality
        // Calculate optimal SF for this SNR (with 10dB margin)
        uint8_t optimalSF = 7;
        for (int sf = 7; sf <= 12; sf++) {
            double required = REQUIRED_SNR[sf - 7];
            if (rxSnr > required + 10.0) {
                optimalSF = sf;
                break;
            }
        }
        
        // Penalize SF that's too high for the SNR
        int sfExcess = (int)currentSF - (int)optimalSF;
        if (sfExcess > 1) {
            double penalty = sfExcess * 20.0;  // 20 points per unnecessary SF level
            reward -= penalty;
            
            if (sfExcess >= 2) {
                std::cout << "⚠️ SNR-SF MISMATCH: Node " << nodeId
                          << " has SNR=" << rxSnr << "dB but using SF" << (int)currentSF 
                          << " (optimal=" << (int)optimalSF << "), penalty=" << penalty << std::endl;
            }
        }
        
        // Bonus for using appropriate SF with good SNR
        if (sfExcess <= 0 && packetSuccess) {
            reward += 15.0;  // Reward efficient SF choice
        }
    }
    
    // ========================================
    // COMPONENT 7: Collision Detection from Airtime (NEW)
    // High SF + good SNR + packet loss = likely collision from long airtime
    // ========================================
    if (!packetSuccess && currentSF >= 9 && rxSnr > 10.0) {
        // Lost packet despite good SNR and high SF = likely collision due to long airtime
        reward -= 35.0;
        std::cout << "💥 Likely collision from long airtime: Node " << nodeId
                  << ", SF=" << (int)currentSF << ", SNR=" << rxSnr << "dB" << std::endl;
    }
    
    return reward;
}

// Global optimized agents map
std::map<uint32_t, std::unique_ptr<OptimizedDDQNAgent>> optimizedAgents;
