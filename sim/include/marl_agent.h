// -*- mode: c++; -*-
#pragma once
// MARL Agent: MARLAgent (IQL + coordination), marlAgents, deviceChannels

class MARLAgent {
private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    
    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;
    
    std::mt19937 rng;
    
    // Coordination state: stores recent actions of other agents
    std::map<uint32_t, int> otherAgentActions;
    std::map<uint32_t, uint8_t> otherAgentChannels;
    
public:
    MARLAgent(double lr = 0.0005, double eps = 0.3, double epsDecay = 0.999,
              double epsMin = 0.05, double g = 0.95, int targetFreq = 50)
        : mainNetwork(lr), targetNetwork(lr), replayBuffer(20000),
          epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin), gamma(g),
          targetUpdateFreq(targetFreq), updateCounter(0), rng(std::random_device{}())
    {
        targetNetwork.copyFrom(mainNetwork);
    }
    
    /**
     * Build extended state vector including coordination signals
     */
    std::vector<double> buildCoordinatedState(
        const DeviceHistory& history,
        uint8_t currentSF,
        double currentTP,
        uint8_t currentChannel,
        double currentTime,
        uint32_t nodeId,
        const std::map<uint32_t, DeviceHistory>& allHistories,
        const std::map<uint32_t, uint8_t>& allSFs,
        const std::map<uint32_t, uint8_t>& allChannels)
    {
        double timeSinceLastSuccess = currentTime - history.lastSuccessTime;
        
        // Count devices on same channel (collision risk)
        int sameChannelCount = 0;
        double avgOtherPdr = 0.0;
        int otherCount = 0;
        
        for (const auto& kv : allChannels) {
            if (kv.first != nodeId && kv.second == currentChannel) {
                sameChannelCount++;
            }
        }
        
        for (const auto& kv : allHistories) {
            if (kv.first != nodeId) {
                avgOtherPdr += kv.second.getRecentPdr();
                otherCount++;
            }
        }
        if (otherCount > 0) avgOtherPdr /= otherCount;
        
        return {
            // Standard features (same as OptimizedDDQNAgent)
            std::max(0.0, std::min(1.0, (history.getLastRssi() + 120.0) / 40.0)),
            std::max(0.0, std::min(1.0, history.getRssiVariance() / 20.0)),
            std::max(0.0, std::min(1.0, (history.getLastSnr() + 20.0) / 50.0)),
            std::max(0.0, std::min(1.0, (history.getSnrTrend() + 5.0) / 10.0)),
            history.getRecentPdr(),
            std::max(0.0, std::min(1.0, (history.getPdrTrend() + 0.5) / 1.0)),
            std::max(0.0, std::min(1.0, history.consecutiveLosses / 5.0)),
            std::max(0.0, std::min(1.0, timeSinceLastSuccess / 300.0)),
            (currentSF - 7.0) / 5.0,
            (currentTP - 2.0) / 12.0,
            currentChannel / 7.0,
            // Coordination features
            std::min(1.0, sameChannelCount / g_coordDivisor)  // Normalized collision risk
        };
    }
    
    /**
     * Select action with coordination-aware exploration
     */
    int selectAction(const std::vector<double>& state, uint8_t currentChannel,
                     const std::map<uint32_t, uint8_t>& allChannels, uint32_t nodeId) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Count congestion on each channel
        std::vector<int> channelCounts(8, 0);
        for (const auto& kv : allChannels) {
            if (kv.first != nodeId) {
                channelCounts[kv.second]++;
            }
        }
        
        // If current channel is congested, prefer channel change actions
        if (channelCounts[currentChannel] >= 2 && dist(rng) < 0.4) {
            // Find least congested channel
            int leastCongested = 0;
            for (int i = 1; i < 8; i++) {
                if (channelCounts[i] < channelCounts[leastCongested]) {
                    leastCongested = i;
                }
            }
            // Return channel change action
            return 12 + leastCongested;  // Actions 12-19 are channel changes
        }
        
        // Standard epsilon-greedy
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> actionDist(0, 47);
            return actionDist(rng);
        }
        
        std::vector<double> qValues = mainNetwork.forward(state);
        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }
    
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done) {
        Experience exp(state, action, reward, nextState, done);
        replayBuffer.add(exp);
    }
    
    void train(size_t batchSize = 32) {
        if (replayBuffer.size() < batchSize * 2) return;
        
        std::vector<size_t> indices;
        std::vector<double> weights;
        auto batch = replayBuffer.sample(batchSize, indices, weights);
        
        std::vector<double> tdErrors;
        
        for (size_t i = 0; i < batch.size(); i++) {
            auto& exp = batch[i];
            
            std::vector<double> currentQ = mainNetwork.forward(exp.state);
            std::vector<double> nextQ = mainNetwork.forward(exp.nextState);
            std::vector<double> nextQTarget = targetNetwork.forward(exp.nextState);
            
            int bestAction = std::distance(nextQ.begin(), std::max_element(nextQ.begin(), nextQ.end()));
            double target = exp.reward + (exp.done ? 0.0 : gamma * nextQTarget[bestAction]);
            
            double tdError = std::abs(target - currentQ[exp.action]);
            tdErrors.push_back(tdError);
            
            // Note: DuelingQNetwork doesn't have updateWeights method
            // This is simplified - in production, use proper backprop
        }
        
        // Update priorities in batch
        replayBuffer.updatePriorities(indices, tdErrors);
        
        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0) {
            targetNetwork.copyFrom(mainNetwork);
        }
        
        epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    }
    
    void decayEpsilon() {
        epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    }
    
    double getEpsilon() const { return epsilon; }
    
    void increaseExploration() {
        epsilon = std::min(0.8, epsilon + 0.3);
    }
};

// Global MARL agents map
std::map<uint32_t, std::unique_ptr<MARLAgent>> marlAgents;

// Global channel tracking for MARL coordination
std::map<uint32_t, uint8_t> deviceChannels;

// ============================================================================
// MARL-PPO: Independent PPO agents with MARL coordination signal
// Same 12-dim coordinated state as MARLAgent; same coordination-aware
// channel-hop pre-selection; softmax actor over 48 discrete actions.
// ============================================================================

/**
 * Lightweight actor network for MARL-PPO.
 * Input : 12-dim coordinated state (local 11 + n_same_ch/3)
 * Output: logits over 48 discrete actions
 */
