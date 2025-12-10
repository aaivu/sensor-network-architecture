/*
 * advanced-ddqn-per-adr-example.cc
 * 
 * This file combines the DDQN-PER ADR from ddqn-per-adr-example.cc
 * with the realistic environment from advanced-pre-tx-rssi-example.cc.
 * 
 * It allows for three modes of operation for direct comparison:
 * 1. No ADR (`--adr=off`)
 * 2. Classical LoRaWAN ADR (`--adr=on`)
 * 3. DDQN-PER based ADR (`--adr=ddqn`)
 * 
 * Key Features:
 * - DDQN architecture with main and target networks
 * - Prioritized Experience Replay buffer
 * - Multi-objective reward function (PDR vs Energy Consumption)
 * - Real-time interference calculation at transmitter location
 * - Comprehensive CSV logging for ML dataset generation
 */

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/end-device-lora-phy.h"
#include "ns3/end-device-lorawan-mac.h"
#include "ns3/gateway-lora-phy.h"
#include "ns3/gateway-lorawan-mac.h"
#include "ns3/log.h"
#include "ns3/lora-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/periodic-sender-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"
#include "ns3/lora-utils.h"
#include "ns3/string.h"
#include "ns3/lora-channel.h"
#include "ns3/network-server-helper.h"
#include "ns3/forwarder-helper.h"
#include "ns3/buildings-propagation-loss-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-net-device.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <queue>
#include <memory>
#include <cmath>
#include <random>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("AdvancedDDQNPERADRExample");

// ============================================================================
// DDQN-PER IMPLEMENTATION
// ============================================================================

/**
 * Experience structure for Prioritized Experience Replay
 */
struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> nextState;
    bool done;
    double priority;
    double tdError;
    
    Experience(const std::vector<double>& s, int a, double r, 
              const std::vector<double>& ns, bool d, double p = 1.0)
        : state(s), action(a), reward(r), nextState(ns), done(d), priority(p), tdError(0.0) {}
};

/**
 * Simple Neural Network implementation for Q-value approximation
 */
class SimpleQNetwork {
private:
    static const int INPUT_SIZE = 6;
    static const int HIDDEN1_SIZE = 128;
    static const int HIDDEN2_SIZE = 64;
    static const int OUTPUT_SIZE = 15;
    
    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    std::vector<double> lastInput, lastHidden1, lastHidden2, lastOutput;
    double learningRate;
    
    double relu(double x) { return std::max(0.0, x); }
    double reluDerivative(double x) { return x > 0 ? 1.0 : 0.0; }
    
public:
    SimpleQNetwork(double lr = 0.001) : learningRate(lr) {
        initializeWeights();
    }
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0.0, 0.1);
        
        weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN1_SIZE));
        weights2.resize(HIDDEN1_SIZE, std::vector<double>(HIDDEN2_SIZE));
        weights3.resize(HIDDEN2_SIZE, std::vector<double>(OUTPUT_SIZE));
        
        bias1.resize(HIDDEN1_SIZE);
        bias2.resize(HIDDEN2_SIZE);
        bias3.resize(OUTPUT_SIZE);
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                weights1[i][j] = dis(gen);
            }
        }
        
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            bias1[i] = dis(gen);
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                weights2[i][j] = dis(gen);
            }
        }
        
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            bias2[i] = dis(gen);
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weights3[i][j] = dis(gen);
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias3[i] = dis(gen);
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        lastInput = input;
        
        lastHidden1.resize(HIDDEN1_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            lastHidden1[i] = bias1[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                lastHidden1[i] += input[j] * weights1[j][i];
            }
            lastHidden1[i] = relu(lastHidden1[i]);
        }
        
        lastHidden2.resize(HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            lastHidden2[i] = bias2[i];
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                lastHidden2[i] += lastHidden1[j] * weights2[j][i];
            }
            lastHidden2[i] = relu(lastHidden2[i]);
        }
        
        lastOutput.resize(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            lastOutput[i] = bias3[i];
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                lastOutput[i] += lastHidden2[j] * weights3[j][i];
            }
        }
        
        return lastOutput;
    }
    
    void updateWeights(const std::vector<double>& targetOutput) {
        std::vector<double> outputError(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            outputError[i] = targetOutput[i] - lastOutput[i];
        }
        
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weights3[i][j] += learningRate * outputError[j] * lastHidden2[i];
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias3[i] += learningRate * outputError[i];
        }
        
        std::vector<double> hidden2Error(HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            hidden2Error[i] = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                hidden2Error[i] += outputError[j] * weights3[i][j];
            }
            hidden2Error[i] *= reluDerivative(lastHidden2[i]);
        }
        
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                weights2[i][j] += learningRate * hidden2Error[j] * lastHidden1[i];
            }
        }
        
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            bias2[i] += learningRate * hidden2Error[i];
        }
        
        std::vector<double> hidden1Error(HIDDEN1_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            hidden1Error[i] = 0.0;
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                hidden1Error[i] += hidden2Error[j] * weights2[i][j];
            }
            hidden1Error[i] *= reluDerivative(lastHidden1[i]);
        }
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                weights1[i][j] += learningRate * hidden1Error[j] * lastInput[i];
            }
        }
        
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            bias1[i] += learningRate * hidden1Error[i];
        }
    }
    
    void copyFrom(const SimpleQNetwork& other) {
        weights1 = other.weights1;
        weights2 = other.weights2;
        weights3 = other.weights3;
        bias1 = other.bias1;
        bias2 = other.bias2;
        bias3 = other.bias3;
    }
};

/**
 * Prioritized Experience Replay Buffer
 */
class PrioritizedReplayBuffer {
private:
    std::vector<Experience> buffer;
    std::vector<double> priorities;
    size_t maxSize, currentSize, position;
    double alpha, beta, epsilon;
    
public:
    PrioritizedReplayBuffer(size_t maxSize, double alpha = 0.6, double beta = 0.4, double epsilon = 1e-6)
        : maxSize(maxSize), currentSize(0), position(0), alpha(alpha), beta(beta), epsilon(epsilon) {
        buffer.reserve(maxSize);
        priorities.reserve(maxSize);
    }
    
    void add(const Experience& experience) {
        double priority = priorities.empty() ? 1.0 : *std::max_element(priorities.begin(), priorities.end());
        
        if (currentSize < maxSize) {
            buffer.push_back(experience);
            priorities.push_back(priority);
            currentSize++;
        } else {
            buffer[position] = experience;
            priorities[position] = priority;
        }
        
        position = (position + 1) % maxSize;
    }
    
    std::vector<Experience> sample(size_t batchSize, std::vector<size_t>& indices, std::vector<double>& weights) {
        if (currentSize == 0) return {};
        
        std::vector<double> probs(currentSize);
        double totalPriority = 0.0;
        
        for (size_t i = 0; i < currentSize; i++) {
            probs[i] = std::pow(priorities[i] + epsilon, alpha);
            totalPriority += probs[i];
        }
        
        for (size_t i = 0; i < currentSize; i++) {
            probs[i] /= totalPriority;
        }
        
        std::vector<Experience> batch;
        indices.clear();
        weights.clear();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        double maxWeight = 0.0;
        for (size_t i = 0; i < batchSize && i < currentSize; i++) {
            double r = dis(gen);
            double cumSum = 0.0;
            size_t idx = 0;
            
            for (size_t j = 0; j < currentSize; j++) {
                cumSum += probs[j];
                if (r <= cumSum) {
                    idx = j;
                    break;
                }
            }
            
            batch.push_back(buffer[idx]);
            indices.push_back(idx);
            
            double weight = std::pow(currentSize * probs[idx], -beta);
            weights.push_back(weight);
            maxWeight = std::max(maxWeight, weight);
        }
        
        for (double& weight : weights) {
            weight /= maxWeight;
        }
        
        return batch;
    }
    
    void updatePriorities(const std::vector<size_t>& indices, const std::vector<double>& tdErrors) {
        for (size_t i = 0; i < indices.size(); i++) {
            priorities[indices[i]] = std::abs(tdErrors[i]) + epsilon;
        }
    }
    
    size_t size() const { return currentSize; }
};

/**
 * DDQN-PER Agent for ADR
 */
class DDQNPERADRAgent {
private:
    SimpleQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    double epsilon, epsilonDecay, epsilonMin, gamma;
    int targetUpdateFreq, updateCounter;
    std::vector<int> sfActions = {7, 8, 9, 10, 11, 12};
    std::vector<int> tpActions = {8, 11, 14};  // ✅ FIX: Higher minimum power (8 dBm instead of 2)
    double pdrWeight, energyWeight;
    std::mt19937 rng;
    double getRequiredSNR(int sf) {
        // Required SNR for each SF (approximate values in dB)
        std::map<int, double> requiredSNR = {
            {7, -7.5}, {8, -10}, {9, -12.5}, {10, -15}, {11, -17.5}, {12, -20}
        };
        return requiredSNR[sf];
    }
    
    int getOptimalSFForDistance(double distance) {
        // Heuristic: closer devices can use lower SF for better data rate
        if (distance < 300) return 7;
        else if (distance < 600) return 8;
        else if (distance < 900) return 9;
        else if (distance < 1200) return 10;
        else if (distance < 1500) return 11;
        else return 12;
    }
    
public:

    DDQNPERADRAgent(double lr = 0.001, double eps = 1.0, double epsDecay = 0.995, 
                   double epsMin = 0.01, double gam = 0.95, int targetFreq = 100,
                   double pdrW = 1.0, double energyW = 0.5)
        : mainNetwork(lr), targetNetwork(lr), replayBuffer(10000),
          epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin), gamma(gam),
          targetUpdateFreq(targetFreq), updateCounter(0),
          pdrWeight(pdrW), energyWeight(energyW), rng(std::random_device{}()) {
        
        targetNetwork.copyFrom(mainNetwork);
    }
    void increaseExploration() {
        epsilon = std::min(0.8, epsilon + 0.3);  // Boost exploration when stuck
        std::cout << "🔄 Epsilon boosted to " << epsilon << " for better exploration" << std::endl;
    }
    int selectAction(const std::vector<double>& state) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        if (dis(rng) < epsilon) {
            std::uniform_int_distribution<int> actionDis(0, 11);
            return actionDis(rng);
        } else {
            std::vector<double> qValues = mainNetwork.forward(state);
            return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
        }
    }
    
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done) {
        Experience exp(state, action, reward, nextState, done);
        replayBuffer.add(exp);
    }
    
    void train(size_t batchSize = 32) {
        if (replayBuffer.size() < batchSize) return;
        
        std::vector<size_t> indices;
        std::vector<double> weights;
        std::vector<Experience> batch = replayBuffer.sample(batchSize, indices, weights);
        
        std::vector<double> tdErrors;
        
        for (size_t i = 0; i < batch.size(); i++) {
            const Experience& exp = batch[i];
            
            std::vector<double> currentQ = mainNetwork.forward(exp.state);
            std::vector<double> nextQMain = mainNetwork.forward(exp.nextState);
            std::vector<double> nextQTarget = targetNetwork.forward(exp.nextState);
            
            int bestAction = std::distance(nextQMain.begin(), 
                                         std::max_element(nextQMain.begin(), nextQMain.end()));
            
            double targetQ = exp.reward;
            if (!exp.done) {
                targetQ += gamma * nextQTarget[bestAction];
            }
            
            double tdError = targetQ - currentQ[exp.action];
            tdErrors.push_back(tdError);
            
            std::vector<double> targetOutput = currentQ;
            targetOutput[exp.action] = targetQ;
            
            for (double& val : targetOutput) {
                val *= weights[i];
            }
            
            mainNetwork.updateWeights(targetOutput);
        }
        
        replayBuffer.updatePriorities(indices, tdErrors);
        
        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0) {
            targetNetwork.copyFrom(mainNetwork);
        }
        
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }
    
    std::pair<int, int> getParameters(int action) {
        // Action space: 0-5 = SF changes, 6-11 = SF+TP combinations, 12-14 = TP only
        if (action < 6) {
            // Pure SF actions (75% of action space)
            return {sfActions[action], -1};  // Change SF, keep current TP
        } else if (action < 12) {
            // SF + TP combinations (25% of action space)
            int sfIdx = (action - 6) / 2;  // 0,0,1,1,2,2
            int tpIdx = (action - 6) % 2;  // 0,1,0,1,0,1
            return {sfActions[sfIdx + 3], tpActions[tpIdx]};  // SF 10-12 with low/high TP
        } else {
            // Pure TP actions (minimal, for edge cases)
            return {-1, tpActions[action - 12]};
        }
    }
    
    double calculateReward(double pdr, double energyConsumption, double prevPdr, double prevEnergy) {
        double pdrImprovement = pdr - prevPdr;
        double energyReduction = prevEnergy - energyConsumption;
        
        double reward = pdrWeight * pdrImprovement + energyWeight * energyReduction;
        
        if (pdr > 0.8 && energyConsumption < 0.2) {
            reward += 1.0;
        }
        
        return reward;
    }
    
   /**
    * SIMPLIFIED REWARD FUNCTION - Clear credit assignment
    * Removes complex penalty scaling that was causing learning issues
    */
   double calculateAdvancedReward(uint32_t nodeId, double pdr, double energyConsumption, 
                              double snr, bool packetSuccess, int currentSF, double currentTP,
                              double distance, uint32_t packetsSent) {
    double reward = 0.0;
    
    // ========================================
    // Component 1: Immediate packet outcome (PRIMARY - 50%)
    // ========================================
    if (packetSuccess) {
        reward += 100.0;  // Strong positive signal
    } else {
        reward -= 30.0;   // Moderate negative (allow exploration)
    }
    
    // ========================================
    // Component 2: PDR trend (SECONDARY - 30%)
    // Only after enough samples for reliable PDR
    // ========================================
    if (packetsSent >= 10) {
        if (pdr >= 0.8) {
            reward += 50.0;  // Excellent
        } else if (pdr >= 0.6) {
            reward += 30.0;  // Good
        } else if (pdr >= 0.4) {
            reward += 10.0;  // Acceptable
        } else if (pdr >= 0.2) {
            reward -= 10.0;  // Poor
        } else {
            reward -= 30.0;  // Very poor - need to change something
        }
    }
    
    // ========================================
    // Component 3: Energy efficiency (TERTIARY - 20%)
    // Only reward efficiency if packet succeeded
    // ========================================
    if (packetSuccess) {
        // Reward lower SF if packet succeeded (more efficient)
        reward += (12 - currentSF) * 5.0;  // SF7 = +25, SF12 = 0
        
        // Reward lower TX power if packet succeeded
        reward += (14 - currentTP) / 2.0;  // 14dBm = 0, 8dBm = +3
    }
    
    // ========================================
    // Component 4: SNR margin guidance (GUIDANCE)
    // Help agent understand when to change SF
    // ========================================
    double requiredSNR[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};  // SF7-SF12
    int sfIdx = std::max(0, std::min(5, currentSF - 7));
    double snrMargin = snr - requiredSNR[sfIdx];
    
    if (snr > -100.0) {  // Valid SNR measurement
        if (snrMargin > 15.0 && currentSF > 7) {
            // Too much margin - could use lower SF for efficiency
            reward -= 15.0;
        } else if (snrMargin < -3.0 && currentSF < 12) {
            // Not enough margin - should use higher SF
            reward -= 25.0;
        } else if (snrMargin >= 3.0 && snrMargin <= 12.0) {
            // Good margin - optimal operation
            reward += 20.0;
        }
    }
    
    // ========================================
    // Debug output (less verbose)
    // ========================================
    if (packetsSent % 50 == 0) {
        std::cout << "📊 Node " << nodeId 
                  << ": reward=" << std::fixed << std::setprecision(1) << reward
                  << ", PDR=" << std::setprecision(1) << pdr*100 << "%"
                  << ", SF=" << currentSF 
                  << ", margin=" << std::setprecision(1) << snrMargin << "dB" << std::endl;
    }
    
    return reward;
}
    
    double getEpsilon() const { return epsilon; }
};

// ============================================================================
// ENHANCED DDQN COMPONENTS FOR SHORT-RANGE OPTIMIZATION
// ============================================================================

/**
 * DEVICE HISTORY - Tracks observable metrics for realistic state representation
 * Only uses features available on real LoRaWAN devices
 */
struct DeviceHistory {
    std::deque<double> rssiHistory;      // Last 20 RSSI readings
    std::deque<double> snrHistory;       // Last 20 SNR values
    std::deque<bool> packetResults;      // Last 20 packet outcomes
    std::deque<double> timestamps;       // Timestamps of packets
    uint32_t consecutiveLosses = 0;
    double lastSuccessTime = 0.0;
    uint8_t currentChannel = 0;
    uint8_t previousChannel = 0;
    
    void addRssi(double rssi) {
        rssiHistory.push_back(rssi);
        if (rssiHistory.size() > 20) rssiHistory.pop_front();
    }
    
    void addSnr(double snr) {
        snrHistory.push_back(snr);
        if (snrHistory.size() > 20) snrHistory.pop_front();
    }
    
    void addPacketResult(bool success, double timestamp) {
        packetResults.push_back(success);
        timestamps.push_back(timestamp);
        if (packetResults.size() > 20) {
            packetResults.pop_front();
            timestamps.pop_front();
        }
        
        if (success) {
            consecutiveLosses = 0;
            lastSuccessTime = timestamp;
        } else {
            consecutiveLosses++;
        }
    }
    
    void setChannel(uint8_t channel) {
        previousChannel = currentChannel;
        currentChannel = channel;
    }
    
    double getRssiVariance() const {
        if (rssiHistory.size() < 2) return 0.0;
        double mean = 0.0;
        for (double r : rssiHistory) mean += r;
        mean /= rssiHistory.size();
        
        double variance = 0.0;
        for (double r : rssiHistory) variance += (r - mean) * (r - mean);
        return variance / rssiHistory.size();
    }
    
    double getSnrTrend() const {
        if (snrHistory.size() < 5) return 0.0;
        // Compare recent average to older average
        double recentAvg = 0.0, oldAvg = 0.0;
        size_t mid = snrHistory.size() / 2;
        for (size_t i = 0; i < mid; i++) oldAvg += snrHistory[i];
        for (size_t i = mid; i < snrHistory.size(); i++) recentAvg += snrHistory[i];
        oldAvg /= mid;
        recentAvg /= (snrHistory.size() - mid);
        return recentAvg - oldAvg;
    }
    
    double getRecentPdr() const {
        if (packetResults.empty()) return 0.5;
        int successes = std::count(packetResults.begin(), packetResults.end(), true);
        return (double)successes / packetResults.size();
    }
    
    double getPdrTrend() const {
        if (packetResults.size() < 10) return 0.0;
        // Compare last 5 to previous 5
        int recent = 0, old = 0;
        size_t n = packetResults.size();
        for (size_t i = n - 5; i < n; i++) recent += packetResults[i] ? 1 : 0;
        for (size_t i = n - 10; i < n - 5; i++) old += packetResults[i] ? 1 : 0;
        return (recent - old) / 5.0;
    }
    
    double getLastRssi() const {
        return rssiHistory.empty() ? -110.0 : rssiHistory.back();
    }
    
    double getLastSnr() const {
        return snrHistory.empty() ? 0.0 : snrHistory.back();
    }
};

// Global storage for device histories
std::map<uint32_t, DeviceHistory> deviceHistories;

/**
 * COLLISION DETECTION HEURISTIC
 * Infers collision probability from observable metrics
 */
struct CollisionIndicators {
    double collisionProbability;
    bool likelyCollision;
    bool likelyWeakSignal;
    bool likelyInterference;
    
    static CollisionIndicators analyze(
        double preTxRssi,
        double rxSnr,
        bool packetSuccess,
        double rssiVariance,
        uint32_t consecutiveLosses,
        double recentPdr)
    {
        CollisionIndicators result;
        result.collisionProbability = 0.0;
        result.likelyCollision = false;
        result.likelyWeakSignal = false;
        result.likelyInterference = false;
        
        if (!packetSuccess) {
            // Case 1: High pre-TX RSSI + packet loss = likely collision
            if (preTxRssi > -105.0) {
                result.collisionProbability += 0.4;
                result.likelyCollision = true;
            }
            
            // Case 2: High RSSI variance + loss = interference
            if (rssiVariance > 8.0) {
                result.collisionProbability += 0.3;
                result.likelyInterference = true;
            }
            
            // Case 3: Good SNR history but sudden loss = collision
            if (rxSnr > 5.0 && consecutiveLosses < 3) {
                result.collisionProbability += 0.2;
                result.likelyCollision = true;
            }
            
            // Case 4: Low SNR = weak signal, not collision
            if (rxSnr < -5.0) {
                result.likelyWeakSignal = true;
                result.collisionProbability *= 0.5;
            }
            
            // Case 5: Pattern of intermittent losses = periodic interference
            if (recentPdr > 0.3 && recentPdr < 0.7 && rssiVariance > 5.0) {
                result.likelyInterference = true;
            }
        }
        
        result.collisionProbability = std::min(1.0, result.collisionProbability);
        return result;
    }
};

/**
 * CHANNEL SELECTOR - Tracks interference per channel and recommends best channel
 */
class ChannelSelector {
private:
    static const int NUM_CHANNELS = 8;
    
    struct ChannelStats {
        std::deque<double> rssiHistory;
        std::deque<bool> successHistory;
        double avgRssi = -110.0;
        double successRate = 0.5;
        double lastUsedTime = 0.0;
        
        void addSample(double rssi, bool success, double timestamp) {
            rssiHistory.push_back(rssi);
            successHistory.push_back(success);
            lastUsedTime = timestamp;
            
            if (rssiHistory.size() > 10) {
                rssiHistory.pop_front();
                successHistory.pop_front();
            }
            
            // Update averages
            avgRssi = 0.0;
            int successes = 0;
            for (size_t i = 0; i < rssiHistory.size(); i++) {
                avgRssi += rssiHistory[i];
                if (successHistory[i]) successes++;
            }
            avgRssi /= rssiHistory.size();
            successRate = (double)successes / successHistory.size();
        }
    };
    
    std::map<uint32_t, std::array<ChannelStats, NUM_CHANNELS>> deviceChannelStats;
    
public:
    void recordChannelUsage(uint32_t nodeId, uint8_t channel, double rssi, bool success, double timestamp) {
        if (channel >= NUM_CHANNELS) return;
        deviceChannelStats[nodeId][channel].addSample(rssi, success, timestamp);
    }
    
    uint8_t getBestChannel(uint32_t nodeId, double currentTime) {
        auto& stats = deviceChannelStats[nodeId];
        
        uint8_t bestChannel = 0;
        double bestScore = -1000.0;
        
        for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
            // Score = success rate + (lower RSSI is better) + time diversity
            double timeSinceUsed = currentTime - stats[ch].lastUsedTime;
            double timeBonus = std::min(1.0, timeSinceUsed / 60.0);
            
            double score = stats[ch].successRate * 100.0 
                         - (stats[ch].avgRssi + 110.0)
                         + timeBonus * 10.0;
            
            if (score > bestScore) {
                bestScore = score;
                bestChannel = ch;
            }
        }
        
        return bestChannel;
    }
    
    bool shouldChangeChannel(uint32_t nodeId, uint8_t currentChannel) {
        auto& stats = deviceChannelStats[nodeId];
        
        if (stats[currentChannel].successHistory.size() < 5) return false;
        
        if (stats[currentChannel].successRate < 0.4) {
            return true;
        }
        
        if (stats[currentChannel].avgRssi > -100.0) {
            return true;
        }
        
        return false;
    }
};

// Global channel selector instance
ChannelSelector channelSelector;

/**
 * EXTENDED ACTION SPACE with Channel Selection
 * 48 actions total for SF, TP, and channel changes
 */
class ExtendedActionSpace {
public:
    static const int NUM_ACTIONS = 48;
    
    struct ActionResult {
        int newSF;           // -1 means keep current
        int newTP;           // -1 means keep current
        int channelDelta;    // -1, 0, +1, or specific channel
        bool forceChannelHop;
    };
    
    static ActionResult decodeAction(int action, uint8_t currentSF, double currentTP, uint8_t currentChannel) {
        ActionResult result = {-1, -1, 0, false};
        
        if (action < 6) {
            // Actions 0-5: Pure SF change (SF 7-12)
            result.newSF = 7 + action;
        } else if (action < 12) {
            // Actions 6-11: Pure TP change (2, 5, 8, 11, 14, 17 dBm)
            int tpLevels[] = {2, 5, 8, 11, 14, 17};
            result.newTP = tpLevels[action - 6];
        } else if (action < 20) {
            // Actions 12-19: Channel change (channels 0-7)
            int targetChannel = action - 12;
            result.channelDelta = targetChannel - currentChannel;
            result.forceChannelHop = true;
        } else if (action < 36) {
            // Actions 20-35: SF + Channel combinations
            int idx = action - 20;
            int sfIdx = idx / 3;   // 0-5 -> SF 7-12
            int chAction = idx % 3; // 0=stay, 1=+1, 2=random hop
            
            result.newSF = 7 + sfIdx;
            if (chAction == 1) result.channelDelta = 1;
            else if (chAction == 2) result.forceChannelHop = true;
        } else {
            // Actions 36-47: SF + TP combinations for edge cases
            int idx = action - 36;
            int sfIdx = idx / 2;   // 0-5 -> SF 7-12
            int tpIdx = idx % 2;   // 0=low (8), 1=high (14)
            
            result.newSF = 7 + sfIdx;
            result.newTP = tpIdx ? 14 : 8;
        }
        
        return result;
    }
};

/**
 * DUELING DQN NETWORK
 * Separates state value and advantage estimation for better learning
 */
class DuelingQNetwork {
private:
    static const int INPUT_SIZE = 12;      // Realistic state size
    static const int HIDDEN_SIZE = 64;
    static const int VALUE_SIZE = 32;
    static const int ADVANTAGE_SIZE = 32;
    static const int OUTPUT_SIZE = 48;     // ExtendedActionSpace::NUM_ACTIONS
    
    // Shared feature layers
    std::vector<std::vector<double>> sharedWeights1, sharedWeights2;
    std::vector<double> sharedBias1, sharedBias2;
    
    // Value stream
    std::vector<std::vector<double>> valueWeights1, valueWeights2;
    std::vector<double> valueBias1, valueBias2;
    
    // Advantage stream
    std::vector<std::vector<double>> advWeights1, advWeights2;
    std::vector<double> advBias1, advBias2;
    
    double learningRate;
    
    double leakyRelu(double x, double alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }
    
public:
    DuelingQNetwork(double lr = 0.0005) : learningRate(lr) {
        initializeWeights();
    }
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        auto heInit = [&](int fanIn) {
            return std::normal_distribution<double>(0.0, std::sqrt(2.0 / fanIn));
        };
        
        // Shared layers: INPUT -> 64 -> 64
        sharedWeights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        sharedWeights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        sharedBias1.resize(HIDDEN_SIZE, 0.0);
        sharedBias2.resize(HIDDEN_SIZE, 0.0);
        
        auto dist1 = heInit(INPUT_SIZE);
        for (auto& row : sharedWeights1)
            for (auto& w : row) w = dist1(gen);
        
        auto dist2 = heInit(HIDDEN_SIZE);
        for (auto& row : sharedWeights2)
            for (auto& w : row) w = dist2(gen);
        
        // Value stream: 64 -> 32 -> 1
        valueWeights1.resize(HIDDEN_SIZE, std::vector<double>(VALUE_SIZE));
        valueWeights2.resize(VALUE_SIZE, std::vector<double>(1));
        valueBias1.resize(VALUE_SIZE, 0.0);
        valueBias2.resize(1, 0.0);
        
        for (auto& row : valueWeights1)
            for (auto& w : row) w = dist2(gen);
        auto dist3 = heInit(VALUE_SIZE);
        for (auto& row : valueWeights2)
            for (auto& w : row) w = dist3(gen);
        
        // Advantage stream: 64 -> 32 -> NUM_ACTIONS
        advWeights1.resize(HIDDEN_SIZE, std::vector<double>(ADVANTAGE_SIZE));
        advWeights2.resize(ADVANTAGE_SIZE, std::vector<double>(OUTPUT_SIZE));
        advBias1.resize(ADVANTAGE_SIZE, 0.0);
        advBias2.resize(OUTPUT_SIZE, 0.0);
        
        for (auto& row : advWeights1)
            for (auto& w : row) w = dist2(gen);
        for (auto& row : advWeights2)
            for (auto& w : row) w = dist3(gen);
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        // Shared layers
        std::vector<double> hidden1(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden1[j] = sharedBias1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                hidden1[j] += input[i] * sharedWeights1[i][j];
            }
            hidden1[j] = leakyRelu(hidden1[j]);
        }
        
        std::vector<double> hidden2(HIDDEN_SIZE);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[j] = sharedBias2[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                hidden2[j] += hidden1[i] * sharedWeights2[i][j];
            }
            hidden2[j] = leakyRelu(hidden2[j]);
        }
        
        // Value stream
        std::vector<double> valueHidden(VALUE_SIZE);
        for (int j = 0; j < VALUE_SIZE; j++) {
            valueHidden[j] = valueBias1[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                valueHidden[j] += hidden2[i] * valueWeights1[i][j];
            }
            valueHidden[j] = leakyRelu(valueHidden[j]);
        }
        
        double stateValue = valueBias2[0];
        for (int i = 0; i < VALUE_SIZE; i++) {
            stateValue += valueHidden[i] * valueWeights2[i][0];
        }
        
        // Advantage stream
        std::vector<double> advHidden(ADVANTAGE_SIZE);
        for (int j = 0; j < ADVANTAGE_SIZE; j++) {
            advHidden[j] = advBias1[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                advHidden[j] += hidden2[i] * advWeights1[i][j];
            }
            advHidden[j] = leakyRelu(advHidden[j]);
        }
        
        std::vector<double> advantages(OUTPUT_SIZE);
        double advantageMean = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            advantages[j] = advBias2[j];
            for (int i = 0; i < ADVANTAGE_SIZE; i++) {
                advantages[j] += advHidden[i] * advWeights2[i][j];
            }
            advantageMean += advantages[j];
        }
        advantageMean /= OUTPUT_SIZE;
        
        // Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        std::vector<double> qValues(OUTPUT_SIZE);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            qValues[j] = stateValue + advantages[j] - advantageMean;
        }
        
        return qValues;
    }
    
    void copyFrom(const DuelingQNetwork& other) {
        sharedWeights1 = other.sharedWeights1;
        sharedWeights2 = other.sharedWeights2;
        sharedBias1 = other.sharedBias1;
        sharedBias2 = other.sharedBias2;
        valueWeights1 = other.valueWeights1;
        valueWeights2 = other.valueWeights2;
        valueBias1 = other.valueBias1;
        valueBias2 = other.valueBias2;
        advWeights1 = other.advWeights1;
        advWeights2 = other.advWeights2;
        advBias1 = other.advBias1;
        advBias2 = other.advBias2;
    }
};

/**
 * OPTIMIZED DDQN AGENT FOR SHORT RANGE
 * Uses realistic state representation and extended action space
 */
class OptimizedDDQNAgent {
private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    
    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;
    
    // Curiosity-driven exploration
    std::map<std::string, int> stateVisitCounts;
    double curiosityBonus = 0.1;
    
    std::mt19937 rng;
    
    std::string discretizeState(const std::vector<double>& state) {
        std::string key;
        for (double v : state) {
            key += std::to_string((int)(v * 10)) + "_";
        }
        return key;
    }
    
public:
    OptimizedDDQNAgent(double lr = 0.0005, double eps = 0.3, double epsDecay = 0.999,
                      double epsMin = 0.05, double g = 0.95, int targetFreq = 50)
        : mainNetwork(lr), targetNetwork(lr), replayBuffer(20000),
          epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin), gamma(g),
          targetUpdateFreq(targetFreq), updateCounter(0), rng(std::random_device{}())
    {
        targetNetwork.copyFrom(mainNetwork);
    }
    
    /**
     * Build realistic state vector from observable features only
     */
    std::vector<double> buildRealisticState(
        const DeviceHistory& history,
        uint8_t currentSF,
        double currentTP,
        uint8_t currentChannel,
        double currentTime)
    {
        double timeSinceLastSuccess = currentTime - history.lastSuccessTime;
        
        return {
            // Interference indicators
            std::max(0.0, std::min(1.0, (history.getLastRssi() + 120.0) / 40.0)),
            std::max(0.0, std::min(1.0, history.getRssiVariance() / 20.0)),
            
            // Link quality indicators
            std::max(0.0, std::min(1.0, (history.getLastSnr() + 20.0) / 50.0)),
            std::max(0.0, std::min(1.0, (history.getSnrTrend() + 5.0) / 10.0)),
            
            // Performance indicators
            history.getRecentPdr(),
            std::max(0.0, std::min(1.0, (history.getPdrTrend() + 0.5) / 1.0)),
            std::max(0.0, std::min(1.0, history.consecutiveLosses / 5.0)),
            
            // Temporal features
            std::max(0.0, std::min(1.0, timeSinceLastSuccess / 300.0)),
            
            // Current state
            (currentSF - 7.0) / 5.0,
            (currentTP - 2.0) / 12.0,
            currentChannel / 7.0,
            
            // Time of day (cyclical - could indicate traffic patterns)
            std::fmod(currentTime / 3600.0, 24.0) / 24.0
        };
    }
    
    int selectAction(const std::vector<double>& state, uint32_t nodeId, 
                     const DeviceHistory& history, double currentTime) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // Check for collision indicators
        CollisionIndicators collision = CollisionIndicators::analyze(
            history.getLastRssi(),
            history.getLastSnr(),
            history.packetResults.empty() ? true : history.packetResults.back(),
            history.getRssiVariance(),
            history.consecutiveLosses,
            history.getRecentPdr()
        );
        
        // If likely collision, bias towards channel change (50% of time)
        if (collision.likelyCollision && dist(rng) < 0.5) {
            // Random channel hop (actions 12-19)
            std::uniform_int_distribution<int> channelDist(12, 19);
            int action = channelDist(rng);
            std::cout << "🔀 Collision avoidance: Node " << nodeId << " changing channel" << std::endl;
            return action;
        }
        
        // If likely interference, prefer SF increase + channel hop
        if (collision.likelyInterference && dist(rng) < 0.4) {
            // SF + channel combination (actions 20-35)
            std::uniform_int_distribution<int> comboDist(20, 35);
            return comboDist(rng);
        }
        
        // Epsilon-greedy with curiosity bonus
        if (dist(rng) < epsilon) {
            // Smart exploration
            std::vector<double> qValues = mainNetwork.forward(state);
            
            std::string stateKey = discretizeState(state);
            double bonus = curiosityBonus / (1.0 + stateVisitCounts[stateKey]);
            
            // Softmax exploration with bonus
            std::vector<double> probs(qValues.size());
            double maxQ = *std::max_element(qValues.begin(), qValues.end());
            double sumExp = 0.0;
            
            for (size_t i = 0; i < qValues.size(); i++) {
                probs[i] = std::exp((qValues[i] + bonus - maxQ) / 0.5);
                sumExp += probs[i];
            }
            for (double& p : probs) p /= sumExp;
            
            std::discrete_distribution<> actionDist(probs.begin(), probs.end());
            int action = actionDist(rng);
            
            stateVisitCounts[stateKey]++;
            return action;
        }
        
        // Exploitation
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
            const auto& exp = batch[i];
            
            auto nextQMain = mainNetwork.forward(exp.nextState);
            auto nextQTarget = targetNetwork.forward(exp.nextState);
            
            int bestAction = std::distance(nextQMain.begin(),
                                          std::max_element(nextQMain.begin(), nextQMain.end()));
            
            double target = exp.reward;
            if (!exp.done) {
                target += gamma * nextQTarget[bestAction];
            }
            
            auto currentQ = mainNetwork.forward(exp.state);
            double tdError = target - currentQ[exp.action];
            tdErrors.push_back(std::abs(tdError));
        }
        
        replayBuffer.updatePriorities(indices, tdErrors);
        
        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0) {
            targetNetwork.copyFrom(mainNetwork);
        }
        
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }
    
    double getEpsilon() const { return epsilon; }
    
    void increaseExploration() {
        epsilon = std::min(0.5, epsilon + 0.2);
    }
};

/**
 * INTERFERENCE-AWARE REWARD FUNCTION
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
    
    return reward;
}

// Global optimized agents map
std::map<uint32_t, std::unique_ptr<OptimizedDDQNAgent>> optimizedAgents;

// ============================================================================
// ENHANCED ENVIRONMENT & SIMULATION LOGIC
// ============================================================================

// Global tracking structures
std::map<uint32_t, bool> activeTransmitters;
std::map<uint32_t, double> nodeTxPowers;
std::map<uint32_t, Vector> nodePositions;
std::map<uint32_t, Time> transmissionEndTimes;

// Enhanced environmental modeling structures
struct UrbanObstacle {
    Vector position;
    double width;
    double height;
    double attenuationDb;
};
std::vector<UrbanObstacle> urbanObstacles;
std::map<uint32_t, double> nodeAntennaGains;
std::map<uint32_t, double> nodeNoiseFigures;
std::map<uint32_t, double> nodeHardwareVariance;
NodeContainer wifiInterferers;
bool enableEnvironmentalModeling = true;

// Packet tracking for correlation between TX and RX events
struct PacketRecord {
    double timestamp;
    uint32_t nodeId;
    std::string devEui;
    uint32_t fcnt;
    double txPowerDbm;
    uint8_t sf;
    uint32_t bw;
    Vector txPos;
    double preTxRssiDbm;
    double ambientNoiseDbm;
    bool packetReceived;
    double gatewayRssiDbm;
    double gatewaySnrDb;
    uint32_t payloadBytes;
    bool collisionFlag;
    double energyConsumed;
};
std::vector<PacketRecord> completedPackets;
std::map<uint32_t, uint32_t> deviceFrameCounters;

// Global simulation parameters
Ptr<LoraChannel> globalChannel;
enum class ADRMethod { OFF, ON, DDQN };
ADRMethod currentADRMethod = ADRMethod::OFF;

// Global references for ADR functionality
std::vector<Ptr<LoraNetDevice>> endDevicesNetDevices;
std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>> ddqnAgents;

// ADR parameter tracking (updated via trace sources) - SAME AS WORKING VERSION
std::map<uint32_t, double> nodeCurrentTxPowers; // nodeId -> current ADR-adjusted TX power
std::map<uint32_t, uint8_t> nodeCurrentDataRates; // nodeId -> current ADR-adjusted data rate
std::map<uint32_t, uint32_t> nodeIdToDeviceIndex; // nodeId -> device index in endDevicesNetDevices

// Device performance tracking for DDQN
struct DeviceMetrics {
    uint32_t packetsSent = 0;
    uint32_t packetsReceived = 0;
    double totalEnergyConsumed = 0.0;
    double lastPDR = 0.0;
    double lastEnergyPerPacket = 0.0;
};
std::map<uint32_t, DeviceMetrics> deviceMetrics;

// Track current SF and TP for each device
std::map<uint32_t, uint8_t> currentSF;
std::map<uint32_t, double> currentTP;

// **FUNCTION DECLARATIONS**
void ApplyADRDecision(uint32_t nodeId);
void UpdateDeviceMetrics(uint32_t nodeId, bool packetSuccess, double energyConsumed);
void CheckPacketLoss(uint32_t nodeId, double transmissionTime);
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId);
void RecordTransmissionParameters(Ptr<const Packet> packet, uint32_t nodeId, Time transmissionTime);
void OnPacketReceived(Ptr<const Packet> packet);
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
void UpdateOptimizedDDQN(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);

// Forward declarations
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
void UpdateOptimizedDDQN(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void WriteCSVOutput(const std::string& filename);
void WriteEnvironmentVisualization(const std::string& baseFilename, double radius, uint32_t nDevices);

/**
 * Set up simplified urban environment with virtual obstacles
 */
void SetupUrbanEnvironment(double radius) {
    if (!enableEnvironmentalModeling) return;
    
    // Use fixed seed for consistent obstacle placement
    uint32_t obstaclesPerSide = static_cast<uint32_t>(radius / 80.0);
    double obstacleSpacing = (2.0 * radius) / obstaclesPerSide;
    
    Ptr<UniformRandomVariable> obstacleWidthRand = CreateObject<UniformRandomVariable>();
    obstacleWidthRand->SetAttribute("Min", DoubleValue(20.0));
    obstacleWidthRand->SetAttribute("Max", DoubleValue(120.0));
    obstacleWidthRand->SetStream(1001);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> obstacleHeightRand = CreateObject<UniformRandomVariable>();
    obstacleHeightRand->SetAttribute("Min", DoubleValue(8.0));
    obstacleHeightRand->SetAttribute("Max", DoubleValue(80.0));
    obstacleHeightRand->SetStream(1002);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> attenuationRand = CreateObject<UniformRandomVariable>();
    attenuationRand->SetAttribute("Min", DoubleValue(8.0));
    attenuationRand->SetAttribute("Max", DoubleValue(35.0));
    attenuationRand->SetStream(1003);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> placementProbability = CreateObject<UniformRandomVariable>();
    placementProbability->SetAttribute("Min", DoubleValue(0.0));
    placementProbability->SetAttribute("Max", DoubleValue(1.0));
    placementProbability->SetStream(1004);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> positionJitter = CreateObject<UniformRandomVariable>();
    positionJitter->SetAttribute("Min", DoubleValue(-15.0));
    positionJitter->SetAttribute("Max", DoubleValue(15.0));
    positionJitter->SetStream(1005);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < obstaclesPerSide; i++) {
        for (uint32_t j = 0; j < obstaclesPerSide; j++) {
            if (placementProbability->GetValue() < 0.25) continue;
            
            double centerX = -radius + i * obstacleSpacing + positionJitter->GetValue();
            double centerY = -radius + j * obstacleSpacing + positionJitter->GetValue();
            double distFromCenter = std::sqrt(centerX * centerX + centerY * centerY);
            
            if (distFromCenter < 150.0) continue;
            
            UrbanObstacle obstacle;
            obstacle.position = Vector(centerX, centerY, 0);
            obstacle.width = obstacleWidthRand->GetValue();
            obstacle.height = obstacleHeightRand->GetValue();
            obstacle.attenuationDb = attenuationRand->GetValue();
            
            urbanObstacles.push_back(obstacle);
        }
    }
    
    NS_LOG_INFO("Created " << urbanObstacles.size() << " virtual obstacles for urban modeling");
    std::cout << "Urban environment: Created " << urbanObstacles.size() << " virtual obstacles" << std::endl;
}

/**
 * Structure to hold RSSI sampling results
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

/**
 * Set up simplified WiFi interferer modeling
 */
void SetupWiFiInterferers(double radius, uint32_t nInterferers) {
    if (!enableEnvironmentalModeling || nInterferers == 0) return;
    
    wifiInterferers.Create(nInterferers);
    
    MobilityHelper wifiMobility;
    Ptr<ListPositionAllocator> wifiAllocator = CreateObject<ListPositionAllocator>();
    
    Ptr<UniformRandomVariable> wifiXRand = CreateObject<UniformRandomVariable>();
    wifiXRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiXRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiXRand->SetStream(2001);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> wifiYRand = CreateObject<UniformRandomVariable>();
    wifiYRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiYRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiYRand->SetStream(2002);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < nInterferers; i++) {
        Vector wifiPos(wifiXRand->GetValue(), wifiYRand->GetValue(), 3.0);
        wifiAllocator->Add(wifiPos);
    }
    
    wifiMobility.SetPositionAllocator(wifiAllocator);
    wifiMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    wifiMobility.Install(wifiInterferers);
    
    NS_LOG_INFO("Created " << nInterferers << " WiFi interferers (simplified model)");
    std::cout << "Interference modeling: Created " << nInterferers << " WiFi interferers" << std::endl;
}

/**
 * Initialize hardware variability for each device
 */
void InitializeHardwareVariability(uint32_t nDevices) {
    if (!enableEnvironmentalModeling) return;
    
    Ptr<NormalRandomVariable> antennaGainVar = CreateObject<NormalRandomVariable>();
    antennaGainVar->SetAttribute("Mean", DoubleValue(0.0));
    antennaGainVar->SetAttribute("Variance", DoubleValue(2.0));
    antennaGainVar->SetStream(3001);  // Fixed stream for consistency
    
    Ptr<NormalRandomVariable> noiseFigureVar = CreateObject<NormalRandomVariable>();
    noiseFigureVar->SetAttribute("Mean", DoubleValue(6.0));
    noiseFigureVar->SetAttribute("Variance", DoubleValue(1.5));
    noiseFigureVar->SetStream(3002);  // Fixed stream for consistency
    
    Ptr<NormalRandomVariable> hardwareVar = CreateObject<NormalRandomVariable>();
    hardwareVar->SetAttribute("Mean", DoubleValue(0.0));
    hardwareVar->SetAttribute("Variance", DoubleValue(1.0));
    hardwareVar->SetStream(3003);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < nDevices; i++) {
        nodeAntennaGains[i] = antennaGainVar->GetValue();
        nodeNoiseFigures[i] = noiseFigureVar->GetValue();
        nodeHardwareVariance[i] = hardwareVar->GetValue();
        
        Ptr<NormalRandomVariable> txPowerVar = CreateObject<NormalRandomVariable>();
        txPowerVar->SetAttribute("Mean", DoubleValue(0.0));
        txPowerVar->SetAttribute("Variance", DoubleValue(0.5));
        txPowerVar->SetStream(3100 + i);  // Fixed stream for consistency
        
        if (nodeTxPowers.find(i) != nodeTxPowers.end()) {
            nodeTxPowers[i] += txPowerVar->GetValue();
        }
    }
    
    NS_LOG_INFO("Initialized hardware variability for " << nDevices << " devices");
    std::cout << "Hardware modeling: Added manufacturing tolerances for " << nDevices << " devices" << std::endl;
}

/**
 * Core function: Simulate reading RSSI register at transmitter location
 */
RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    Vector txPos = nodePositions[txNodeId];
    
    Ptr<UniformRandomVariable> noiseVar = CreateObject<UniformRandomVariable>();
    
    if (enableEnvironmentalModeling) {
        double urbanNoiseIncrease = 5.0;
        double timeVaryingNoise = 2.0 * std::sin(currentTime.GetSeconds() / 60.0);
        
        noiseVar->SetAttribute("Min", DoubleValue(-115.0 + urbanNoiseIncrease));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0 + urbanNoiseIncrease + timeVaryingNoise));
    } else {
        noiseVar->SetAttribute("Min", DoubleValue(-115.0));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0));
    }
    
    double ambientNoise = noiseVar->GetValue();
    
    if (enableEnvironmentalModeling && nodeNoiseFigures.find(txNodeId) != nodeNoiseFigures.end()) {
        ambientNoise += (nodeNoiseFigures[txNodeId] - 6.0);
    }
    
    double totalPowerLinear = DbmToW(ambientNoise);
    
    for (const auto& pair : activeTransmitters) {
        uint32_t otherNodeId = pair.first;
        bool isTransmitting = pair.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) continue;
        if (currentTime >= transmissionEndTimes[otherNodeId]) continue;
        
        Vector interfererPos = nodePositions[otherNodeId];
        double interfererTxPower = nodeTxPowers[otherNodeId];
        
        if (enableEnvironmentalModeling) {
            if (nodeAntennaGains.find(otherNodeId) != nodeAntennaGains.end()) {
                interfererTxPower += nodeAntennaGains[otherNodeId];
            }
            if (nodeAntennaGains.find(txNodeId) != nodeAntennaGains.end()) {
                interfererTxPower += nodeAntennaGains[txNodeId];
            }
        }
        
        Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel>();
        txMobility->SetPosition(txPos);
        
        Ptr<ConstantPositionMobilityModel> interfererMobility = CreateObject<ConstantPositionMobilityModel>();
        interfererMobility->SetPosition(interfererPos);
        
        double rxPowerDbm = globalChannel->GetRxPower(interfererTxPower, interfererMobility, txMobility);
        
        if (enableEnvironmentalModeling && nodeHardwareVariance.find(txNodeId) != nodeHardwareVariance.end()) {
            rxPowerDbm += nodeHardwareVariance[txNodeId];
        }
        
        totalPowerLinear += DbmToW(rxPowerDbm);
    }
    
    if (enableEnvironmentalModeling && wifiInterferers.GetN() > 0) {
        Ptr<UniformRandomVariable> wifiInterferenceVar = CreateObject<UniformRandomVariable>();
        wifiInterferenceVar->SetAttribute("Min", DoubleValue(-130.0));
        wifiInterferenceVar->SetAttribute("Max", DoubleValue(-110.0));
        
        double wifiTrafficLevel = 0.5 + 0.5 * std::sin(currentTime.GetSeconds() / 30.0);
        double wifiInterferenceDbm = wifiInterferenceVar->GetValue() + 10.0 * std::log10(wifiTrafficLevel);
        
        totalPowerLinear += DbmToW(wifiInterferenceDbm);
    }
    
    if (enableEnvironmentalModeling) {
        Ptr<UniformRandomVariable> adjacentChannelVar = CreateObject<UniformRandomVariable>();
        adjacentChannelVar->SetAttribute("Min", DoubleValue(-140.0));
        adjacentChannelVar->SetAttribute("Max", DoubleValue(-120.0));
        
        double adjacentInterferenceDbm = adjacentChannelVar->GetValue();
        totalPowerLinear += DbmToW(adjacentInterferenceDbm);
    }
    
    double preTxRssiDbm = WToDbm(totalPowerLinear);
    
    Ptr<NormalRandomVariable> rssiNoise = CreateObject<NormalRandomVariable>();
    rssiNoise->SetAttribute("Mean", DoubleValue(0.0));
    
    if (enableEnvironmentalModeling) {
        rssiNoise->SetAttribute("Variance", DoubleValue(1.5));
    } else {
        rssiNoise->SetAttribute("Variance", DoubleValue(1.0));
    }
    
    double noisyRssiDbm = preTxRssiDbm + rssiNoise->GetValue();
    
    RssiSample sample;
    sample.preTxRssiDbm = noisyRssiDbm;
    sample.ambientNoiseDbm = ambientNoise;
    return sample;
}

/**
 * Callback triggered when ADR changes TX power for a device - SAME AS WORKING VERSION
 */
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
    if (nodeId >= endDevicesNetDevices.size()) return;
    
    Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
        endDevicesNetDevices[nodeId]->GetPhy());
    Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
        endDevicesNetDevices[nodeId]->GetMac());
    
    if (!edPhy || !edMac) return;
    
    // Apply ADR based on current method
    if (currentADRMethod == ADRMethod::DDQN) {
        // **DDQN ADR LOGIC**
        auto it = ddqnAgents.find(nodeId);
        if (it != ddqnAgents.end()) {
            DeviceMetrics& metrics = deviceMetrics[nodeId];
            if (metrics.packetsSent > 0) {
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
                double newTP = 5.0 + (action / 6) * 2.0;  // Power levels
                
                if (newSF != edPhy->GetSpreadingFactor() || 
                    std::abs(newTP - edMac->GetTransmissionPowerDbm()) > 0.1) {
                    
                    edPhy->SetSpreadingFactor(newSF);
                    edMac->SetTransmissionPowerDbm(newTP);
                    
                    // Update global tracking
                    currentSF[nodeId] = newSF;
                    currentTP[nodeId] = newTP;
                    
                    std::cout << "🤖 DDQN ADR: Device " << nodeId 
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
        
        // ✅ FIX: Apply immediately, not after delay
        if (nodeId < endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                // Use the same mechanism as classical ADR - set data rate directly
                edMac->SetDataRate(newDataRate);
                
                std::cout << "✅ DDQN Applied SF=" << (int)newSF 
                          << " (DR=" << (int)newDataRate 
                          << ") to device " << nodeId << std::endl;
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
        currentTP[nodeId] = newTP;
        nodeTxPowers[nodeId] = newTP;
        
        if (nodeId < endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                edMac->SetTransmissionPowerDbm(newTP);
                std::cout << "✅ Applied TP=" << newTP << " dBm to device " << nodeId << " (via MAC)" << std::endl;
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
    
    // Build realistic state (no distance, no device count - only observable features)
    std::vector<double> currentState = optimizedAgents[nodeId]->buildRealisticState(
        history, currentSFVal, currentTPVal, currentChannel, currentTime
    );
    
    // Calculate interference-aware reward
    double reward = calculateInterferenceAwareReward(
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
        currentTime - history.lastSuccessTime
    );
    
    // Debug output
    if (metrics.packetsSent % 20 == 0) {
        std::cout << "🧠 OptDDQN Node " << nodeId 
                  << ": PDR=" << std::fixed << std::setprecision(1) << history.getRecentPdr()*100 << "%"
                  << ", Reward=" << std::setprecision(1) << reward 
                  << ", SF=" << (int)currentSFVal 
                  << ", CH=" << (int)currentChannel
                  << ", ConsecLoss=" << history.consecutiveLosses
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
    
    // Select action using optimized agent (with collision detection)
    int action = optimizedAgents[nodeId]->selectAction(currentState, nodeId, history, currentTime);
    
    // Decode action from extended action space
    ExtendedActionSpace::ActionResult actionResult = ExtendedActionSpace::decodeAction(
        action, currentSFVal, currentTPVal, currentChannel
    );
    
    // Apply SF change
    if (actionResult.newSF != -1 && actionResult.newSF != currentSFVal) {
        currentSF[nodeId] = actionResult.newSF;
        
        if (nodeId < endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                uint8_t newDataRate = 12 - actionResult.newSF;
                edMac->SetDataRate(newDataRate);
                std::cout << "✅ OptDDQN SF=" << actionResult.newSF 
                          << " (Node " << nodeId << ")" << std::endl;
            }
        }
    }
    
    // Apply TP change
    if (actionResult.newTP != -1 && std::abs(actionResult.newTP - currentTPVal) > 0.5) {
        currentTP[nodeId] = actionResult.newTP;
        nodeTxPowers[nodeId] = actionResult.newTP;
        
        if (nodeId < endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                edMac->SetTransmissionPowerDbm(actionResult.newTP);
                std::cout << "✅ OptDDQN TP=" << actionResult.newTP 
                          << "dBm (Node " << nodeId << ")" << std::endl;
            }
        }
    }
    
    // Apply channel change (simulated - LoRaWAN uses frequency hopping)
    if (actionResult.forceChannelHop || actionResult.channelDelta != 0) {
        uint8_t newChannel;
        if (actionResult.forceChannelHop) {
            // Use smart channel selection
            newChannel = channelSelector.getBestChannel(nodeId, currentTime);
        } else {
            newChannel = (currentChannel + actionResult.channelDelta + 8) % 8;
        }
        
        history.setChannel(newChannel);
        std::cout << "📡 OptDDQN Channel " << (int)currentChannel << "→" << (int)newChannel 
                  << " (Node " << nodeId << ")" << std::endl;
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
            currentTP[nodeId] = std::min(17.0, currentTP[nodeId] + 2.0);
            std::cout << "⚡ Classical ADR: Node " << nodeId << " TP " << originalTP 
                      << " → " << currentTP[nodeId] << " dBm (boost signal)" << std::endl;
        }
        
        // **3. APPLY CHANGES IMMEDIATELY VIA MAC LAYER**
        if (originalSF != currentSF[nodeId] || std::abs(originalTP - currentTP[nodeId]) > 0.1) {
            if (nodeId < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    endDevicesNetDevices[nodeId]->GetMac());
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[nodeId]->GetPhy());
                if (edMac && edPhy) {
                    // Apply SF directly to PHY layer for immediate effect
                    edPhy->SetSpreadingFactor(currentSF[nodeId]);
                    
                    // Apply TX power directly to MAC layer  
                    edMac->SetTransmissionPowerDbm(currentTP[nodeId]);
                    
                    std::cout << "✅ Classical ADR Applied IMMEDIATELY: SF=" << (int)currentSF[nodeId] 
                              << ", TP=" << currentTP[nodeId] << "dBm to device " << nodeId << std::endl;
                }
            }
        }
    }
}

/**
 * Run a single simulation
 */
void RunSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                  double radius, const std::string& csvFileName, ADRMethod adrMethod, uint32_t nWifiInterferers);

/**
 * Get optimal initial SF based on distance (heuristic warm-start)
 * This prevents DDQN from starting with random/bad parameters
 */
uint8_t getInitialSFForDistance(double distance) {
    if (distance < 300) return 7;
    if (distance < 500) return 8;
    if (distance < 800) return 9;
    if (distance < 1200) return 10;
    if (distance < 1600) return 11;
    return 12;
}

/**
 * Get optimal initial TX power based on distance
 */
double getInitialTPForDistance(double distance) {
    if (distance < 400) return 8;
    if (distance < 800) return 11;
    return 14;
}

/**
 * Write environment visualization CSV for plotting
 */
void WriteEnvironmentVisualization(const std::string& baseFilename, double radius, uint32_t nDevices) {
    std::string envFilename = "lorawan_datasets/" + baseFilename + "_environment.csv";
    std::ofstream envFile(envFilename);
    
    // Header
    envFile << "type,x,y,z,width,height,attenuation,id,label\n";
    
    // Write gateway
    envFile << "gateway,0,0,15,0,0,0,GW,Gateway\n";
    
    // Write end devices (iterate through actual node positions)
    for (const auto& nodePair : nodePositions) {
        uint32_t nodeId = nodePair.first;
        Vector pos = nodePair.second;
        
        // Skip gateway node (ID 0 at position (0,0,15))
        if (pos.x == 0 && pos.y == 0 && pos.z == 15) continue;
        
        double txPower = (nodeTxPowers.find(nodeId) != nodeTxPowers.end()) ? nodeTxPowers[nodeId] : 14.0;
        envFile << "end_device," << pos.x << "," << pos.y << "," << pos.z 
                << ",0,0," << txPower << "," << nodeId << ",Device_" << nodeId << "\n";
    }
    
    // Write urban obstacles
    for (size_t i = 0; i < urbanObstacles.size(); i++) {
        const UrbanObstacle& obstacle = urbanObstacles[i];
        envFile << "obstacle," << obstacle.position.x << "," << obstacle.position.y << ",0"
                << "," << obstacle.width << "," << obstacle.height << "," << obstacle.attenuationDb
                << ",OBS_" << i << ",Building_" << i << "\n";
    }
    
    // Write WiFi interferers
    for (uint32_t i = 0; i < wifiInterferers.GetN(); i++) {
        Ptr<MobilityModel> mobility = wifiInterferers.Get(i)->GetObject<MobilityModel>();
        if (mobility) {
            Vector pos = mobility->GetPosition();
            envFile << "wifi_interferer," << pos.x << "," << pos.y << "," << pos.z
                    << ",0,0,0,WIFI_" << i << ",WiFi_" << i << "\n";
        }
    }
    
    envFile.close();
    std::cout << "Environment visualization data saved to: " << envFilename << std::endl;
}

/**
 * Write comprehensive CSV output
 */
void WriteCSVOutput(const std::string& filename) {
    std::string folderName = "lorawan_datasets";
    system(("mkdir -p " + folderName).c_str());
    
    std::string baseFilename = filename;
    size_t lastDot = baseFilename.find_last_of('.');
    if (lastDot != std::string::npos) {
        baseFilename = baseFilename.substr(0, lastDot);
    }
    
    std::map<uint32_t, std::vector<PacketRecord>> devicePackets;
    for (const auto& record : completedPackets) {
        devicePackets[record.nodeId].push_back(record);
    }
    
    Vector gatewayPos(0, 0, 15);
    for (const auto& devicePair : devicePackets) {
        uint32_t deviceId = devicePair.first;
        const std::vector<PacketRecord>& packets = devicePair.second;
        
        std::string deviceFile = folderName + "/" + baseFilename + "_device_" + 
                               std::to_string(deviceId) + "_dataset.csv";
        std::ofstream deviceCsv(deviceFile);
        
        deviceCsv << "timestamp,devEUI,fcnt,tx_power_dBm,sf,bw,cr,tx_x,tx_y,tx_z,"
                 << "pre_tx_rssi_dBm,ambient_noise_dBm,expected_rx_rssi_dBm,snr_db,"
                 << "gateway_id,packet_received,energy_consumed_mJ,distance_m\n";
        
        for (const auto& record : packets) {
            double distance = CalculateDistance(record.txPos, gatewayPos);
            
            deviceCsv << std::fixed << std::setprecision(6)
                     << record.timestamp << ","
                     << record.devEui << ","
                     << record.fcnt << ","
                     << record.txPowerDbm << ","
                     << static_cast<int>(record.sf) << ","
                     << record.bw << ","
                     << 1 << ","
                     << record.txPos.x << ","
                     << record.txPos.y << ","
                     << record.txPos.z << ","
                     << record.preTxRssiDbm << ","
                     << record.ambientNoiseDbm << ","
                     << (record.packetReceived ? record.gatewayRssiDbm : -999.0) << ","
                     << (record.packetReceived ? record.gatewaySnrDb : -999.0) << ","
                     << 1 << ","
                     << (record.packetReceived ? 1 : 0) << ","
                     << record.energyConsumed << ","
                     << distance << "\n";
        }
        
        deviceCsv.close();
    }
    
    int totalPackets = completedPackets.size();
    int receivedPackets = 0;
    for (const auto& record : completedPackets) {
        if (record.packetReceived) {
            receivedPackets++;
        }
    }
    double successRate = (totalPackets > 0) ? (double)receivedPackets / totalPackets * 100.0 : 0.0;
    
    // Generate environment visualization CSV
    WriteEnvironmentVisualization(baseFilename, 1000.0, devicePackets.size());
    
    std::cout << "\n=== REALISTIC ns-3 LoRaWAN SIMULATION RESULTS ===" << std::endl;
    std::cout << "Total packets transmitted: " << totalPackets << std::endl;
    std::cout << "Successfully received: " << receivedPackets << " (" << successRate << "%)" << std::endl;
    std::cout << "Per-device datasets saved to: lorawan_datasets/ folder" << std::endl;
}

int main(int argc, char* argv[]) {
    uint32_t nDevices = 10;
    double simulationTime = 300.0;
    double appPeriodSeconds = 30.0;
    double radius = 1000.0;
    std::string csvFileName = "advanced_pre_tx_rssi_dataset.csv";
    std::string adrModeStr = "off";
    bool environmentalModeling = true;
    uint32_t nWifiInterferers = 12;
    
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.AddValue("adr", "ADR mode: 'off', 'on', 'ddqn', or 'all'", adrModeStr);
    cmd.AddValue("environmental", "Enable environmental effects modeling", environmentalModeling);
    cmd.AddValue("wifiInterferers", "Number of WiFi interfering nodes", nWifiInterferers);
    cmd.Parse(argc, argv);
    
    enableEnvironmentalModeling = environmentalModeling;
    
    LogComponentEnable("AdvancedDDQNPERADRExample", LOG_LEVEL_INFO);
    
    std::cout << "\n=== ENHANCED ns-3 LoRaWAN Simulation ===" << std::endl;
    std::cout << "Environmental modeling: " << (enableEnvironmentalModeling ? "ENABLED" : "DISABLED") << std::endl;
    if (enableEnvironmentalModeling) {
        std::cout << "  - Urban obstacle modeling" << std::endl;
        std::cout << "  - WiFi interference modeling (" << nWifiInterferers << " nodes)" << std::endl;
        std::cout << "  - Hardware variability modeling" << std::endl;
    }
    
    if (adrModeStr == "all") {
        std::cout << "\n========================================" << std::endl;
        std::cout << "RUNNING ALL ADR METHODS COMPARISON" << std::endl;
        std::cout << "Simulation Time: " << simulationTime << " seconds" << std::endl;
        std::cout << "Number of Devices: " << nDevices << std::endl;
        std::cout << "App Period: " << appPeriodSeconds << " seconds" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "\n[1/3] Running simulation for NO ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "no_adr_" + csvFileName, ADRMethod::OFF, nWifiInterferers);
        
        std::cout << "\n[2/3] Running simulation for CLASSICAL ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "adr_" + csvFileName, ADRMethod::ON, nWifiInterferers);
        
        std::cout << "\n[3/3] Running simulation for DDQN-PER ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "ddqn_adr_" + csvFileName, ADRMethod::DDQN, nWifiInterferers);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL ADR METHODS COMPARISON COMPLETED" << std::endl;
        std::cout << "All three methods used the same:" << std::endl;
        std::cout << "  - Simulation time: " << simulationTime << "s" << std::endl;
        std::cout << "  - Device count: " << nDevices << std::endl;
        std::cout << "  - RNG seed: 12345 (fixed)" << std::endl;
        std::cout << "  - Packet interval: " << appPeriodSeconds << "s" << std::endl;
        std::cout << "Output files:" << std::endl;
        std::cout << "  - no_adr_" << csvFileName << std::endl;
        std::cout << "  - adr_" << csvFileName << std::endl;
        std::cout << "  - ddqn_adr_" << csvFileName << std::endl;
        std::cout << "========================================\n" << std::endl;
    } else {
        ADRMethod adrMethod;
        std::string filePrefix;
        if (adrModeStr == "on") {
            adrMethod = ADRMethod::ON;
            filePrefix = "adr_";
        } else if (adrModeStr == "ddqn") {
            adrMethod = ADRMethod::DDQN;
            filePrefix = "ddqn_adr_";
        } else {
            adrMethod = ADRMethod::OFF;
            filePrefix = "no_adr_";
        }
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, filePrefix + csvFileName, adrMethod, nWifiInterferers);
    }
    
    return 0;
}

void RunSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                  double radius, const std::string& csvFileName, ADRMethod adrMethod, uint32_t nWifiInterferers) {
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "Starting simulation with parameters:" << std::endl;
    std::cout << "  Duration: " << simulationTime << " seconds" << std::endl;
    std::cout << "  App Period: " << appPeriodSeconds << " seconds" << std::endl;
    std::cout << "  Devices: " << nDevices << std::endl;
    std::cout << "  ADR Mode: " << (adrMethod == ADRMethod::ON ? "CLASSICAL" : (adrMethod == ADRMethod::DDQN ? "DDQN-PER (Optimized)" : "OFF")) << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    endDevicesNetDevices.clear();
    ddqnAgents.clear();
    optimizedAgents.clear();  // Clear optimized agents
    deviceMetrics.clear();
    deviceHistories.clear();  // Clear device histories
    nodeAntennaGains.clear();
    nodeNoiseFigures.clear();
    nodeHardwareVariance.clear();
    urbanObstacles.clear();
    wifiInterferers = NodeContainer();
    
    // **Clear ADR tracking structures**
    nodeCurrentTxPowers.clear();
    nodeCurrentDataRates.clear();
    nodeIdToDeviceIndex.clear();
    
    currentADRMethod = adrMethod;
    
    // Set fixed seed for reproducible results across all ADR methods
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(1);
    
    if (enableEnvironmentalModeling) {
        SetupUrbanEnvironment(radius);
        SetupWiFiInterferers(radius, nWifiInterferers);
        InitializeHardwareVariability(nDevices);
    }
    
    NodeContainer endDevices;
    endDevices.Create(nDevices);
    
    NodeContainer gateways;
    gateways.Create(1);
    
    NodeContainer allNodes = NodeContainer(endDevices, gateways);
    
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    allocator->Add(Vector(0, 0, 15));
    nodePositions[gateways.Get(0)->GetId()] = Vector(0, 0, 15);
    
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-radius));
    xRand->SetAttribute("Max", DoubleValue(radius));
    xRand->SetStream(4001);  // Fixed stream for device X positions
    
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-radius));
    yRand->SetAttribute("Max", DoubleValue(radius));
    yRand->SetStream(4002);  // Fixed stream for device Y positions
    
    Ptr<UniformRandomVariable> powerRand = CreateObject<UniformRandomVariable>();
    powerRand->SetAttribute("Min", DoubleValue(8.0));
    powerRand->SetAttribute("Max", DoubleValue(16.0));
    powerRand->SetStream(4003);  // Fixed stream for TX powers
    
    for (uint32_t i = 0; i < nDevices; i++) {
        uint32_t nodeId = endDevices.Get(i)->GetId();
        Vector pos(xRand->GetValue(), yRand->GetValue(), 1.5);
        allocator->Add(pos);
        nodePositions[nodeId] = pos;
        nodeTxPowers[nodeId] = powerRand->GetValue();
    }
    
    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);
    
    // Create channel with the same propagation model as other examples
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetAttribute("Exponent", DoubleValue(3.76));
    loss->SetAttribute("ReferenceLoss", DoubleValue(35.41));
    loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    
    if (enableEnvironmentalModeling) {
        // Add shadowing for more realistic modeling
        Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
        shadowing->SetAttribute("Variable", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=64]"));
        loss->SetNext(shadowing);
    }
    
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    globalChannel = CreateObject<LoraChannel>(loss, delay);
    
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(globalChannel);
    
    LorawanMacHelper macHelper = LorawanMacHelper();
    LoraHelper helper = LoraHelper();
    
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    // **CRITICAL: Use ALOHA region to disable duty cycle enforcement for fair comparison**
    macHelper.SetRegion(LorawanMacHelper::ALOHA);
    
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, endDevices);
    
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        endDevicesNetDevices.push_back(endDevicesNetDevs.Get(i)->GetObject<LoraNetDevice>());
        // Initialize tracking variables with default values
        uint32_t nodeId = endDevices.Get(i)->GetId();
        currentSF[nodeId] = 7;  // Default SF
        currentTP[nodeId] = nodeTxPowers[nodeId];  // Use initial power
        
        // **CRITICAL: Map node ID to device index for ADR parameter extraction**
        nodeIdToDeviceIndex[nodeId] = i;
        std::cout << "📋 Mapped Node ID " << nodeId << " → Device Index " << i << std::endl;
    }
    
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    NetDeviceContainer gatewayNetDevs = helper.Install(phyHelper, macHelper, gateways);
    
    // **SET UP PROPER ns-3 NETWORK SERVER WITH ADR COMPONENT**
    if (adrMethod == ADRMethod::ON) {
        // Create network server node
        NodeContainer networkServer;
        networkServer.Create(1);
        
        // Create point-to-point links between gateways and network server
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
        p2p.SetChannelAttribute("Delay", StringValue("2ms"));
        
        // Store network server app registration details for later
        P2PGwRegistration_t gwRegistration;
        for (auto gw = gateways.Begin(); gw != gateways.End(); ++gw) {
            auto container = p2p.Install(networkServer.Get(0), *gw);
            auto serverP2PNetDev = DynamicCast<PointToPointNetDevice>(container.Get(0));
            gwRegistration.emplace_back(serverP2PNetDev, *gw);
        }
        
        // Install the NetworkServer application with AdrComponent
        NetworkServerHelper networkServerHelper;
        networkServerHelper.EnableAdr(true);
        networkServerHelper.SetAdr("ns3::AdrComponent");
        networkServerHelper.SetGatewaysP2P(gwRegistration);
        networkServerHelper.SetEndDevices(endDevices);
        networkServerHelper.Install(networkServer.Get(0));
        
        // **CRITICAL: Enable ADR uplink bit on all end devices**
        for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
            Ptr<LoraNetDevice> loraNetDevice = endDevicesNetDevs.Get(i)->GetObject<LoraNetDevice>();
            Ptr<EndDeviceLorawanMac> endDeviceMac = DynamicCast<EndDeviceLorawanMac>(loraNetDevice->GetMac());
            if (endDeviceMac) {
                endDeviceMac->SetUplinkAdrBit(true);
                uint32_t nodeId = endDevices.Get(i)->GetId();
                std::cout << "ADR uplink bit enabled for device " << nodeId << std::endl;
                
                // Get initial ADR parameters
                double initialTxPower = endDeviceMac->GetTransmissionPowerDbm();
                uint8_t initialDataRate = endDeviceMac->GetDataRate();
                std::cout << "Initial ADR parameters for device " << nodeId 
                          << " - TX Power: " << initialTxPower << " dBm, Data Rate: " 
                          << (int)initialDataRate << std::endl;
            }
        }
        
        // **ESSENTIAL: Connect trace sources to capture ADR parameter changes**
        Config::Connect("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::EndDeviceLorawanMac/TxPower",
                       MakeCallback(&OnTxPowerChange));
        Config::Connect("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::EndDeviceLorawanMac/DataRate",
                       MakeCallback(&OnDataRateChange));
        
        std::cout << "✅ ADR trace sources connected for parameter tracking" << std::endl;
        
        // Install the Forwarder application on the gateways
        ForwarderHelper forwarderHelper;
        ApplicationContainer forwarders = forwarderHelper.Install(gateways);
        
        std::cout << "✅ Proper ns-3 Network Server with AdrComponent installed" << std::endl;
        std::cout << "✅ Classical ADR enabled via ns-3 AdrComponent" << std::endl;
    } else {
        for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
            Ptr<LoraNetDevice> loraNetDevice = endDevicesNetDevs.Get(i)->GetObject<LoraNetDevice>();
            if (loraNetDevice) {
                std::cout << "✅ Configured device " << i << " for " 
                          << (adrMethod == ADRMethod::DDQN ? "DDQN ADR" : "No ADR") << std::endl;
            }
        }
        
        if (adrMethod == ADRMethod::DDQN) {
            std::cout << "✅ DDQN-PER ADR enabled" << std::endl;
        } else {
            std::cout << "ADR disabled" << std::endl;
        }
    }
    
    if (adrMethod == ADRMethod::DDQN) {
        std::cout << "\n🎯 Initializing OPTIMIZED DDQN agents with interference awareness..." << std::endl;
        for (uint32_t i = 0; i < nDevices; ++i) {
            uint32_t nodeId = endDevices.Get(i)->GetId();
            
            // Calculate distance from gateway for warm start
            Vector pos = nodePositions[nodeId];
            double distance = std::sqrt(pos.x * pos.x + pos.y * pos.y);
            
            // Initialize with distance-based heuristic (warm start)
            uint8_t initialSF = getInitialSFForDistance(distance);
            double initialTP = getInitialTPForDistance(distance);
            
            currentSF[nodeId] = initialSF;
            currentTP[nodeId] = initialTP;
            nodeTxPowers[nodeId] = initialTP;
            
            // Initialize device history
            deviceHistories[nodeId] = DeviceHistory();
            deviceHistories[nodeId].currentChannel = i % 8;  // Distribute across channels
            
            // Apply to device immediately
            Ptr<EndDeviceLorawanMac> mac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[i]->GetMac());
            if (mac) {
                mac->SetDataRate(12 - initialSF);  // DR = 12 - SF
                mac->SetTransmissionPowerDbm(initialTP);
            }
            
            // Create OPTIMIZED agent with interference awareness
            optimizedAgents[nodeId] = std::make_unique<OptimizedDDQNAgent>(
                0.0005,  // learningRate (lower for stability)
                0.3,     // epsilon - start at 30% exploration (with collision detection)
                0.999,   // epsilonDecay - very slow decay
                0.05,    // minEpsilon - lower minimum
                0.95,    // gamma - discount factor
                50       // targetUpdateFreq
            );
            
            // Also create legacy agent for comparison/fallback
            ddqnAgents[nodeId] = std::make_unique<DDQNPERADRAgent>(
                0.01,    // learningRate
                0.5,     // epsilon
                0.998,   // epsilonDecay
                0.1,     // minEpsilon
                0.9,     // gamma
                50,      // targetUpdateFreq
                1.0,     // pdrWeight
                0.2      // energyWeight
            );
            
            std::cout << "  Device " << nodeId << ": distance=" << std::fixed << std::setprecision(0) 
                      << distance << "m, initialSF=" << (int)initialSF 
                      << ", initialTP=" << initialTP << "dBm"
                      << ", channel=" << (int)deviceHistories[nodeId].currentChannel << std::endl;
        }
        std::cout << "  ✅ Using Dueling DQN with interference detection" << std::endl;
        std::cout << "  ✅ Extended action space (48 actions including channel changes)" << std::endl;
        std::cout << "  ✅ Collision detection heuristics enabled" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "\n🔧 Configuring " << nDevices << " devices with period " << appPeriodSeconds << "s" << std::endl;
    
    for (uint32_t i = 0; i < nDevices; i++) {
        PeriodicSenderHelper appHelper = PeriodicSenderHelper();
        appHelper.SetPeriod(Seconds(appPeriodSeconds));
        appHelper.SetPacketSize(20);  // Ensure packet size is set
        
        ApplicationContainer app = appHelper.Install(endDevices.Get(i));
        
        Ptr<UniformRandomVariable> randomStart = CreateObject<UniformRandomVariable>();
        randomStart->SetAttribute("Min", DoubleValue(0.0));
        randomStart->SetAttribute("Max", DoubleValue(appPeriodSeconds));
        randomStart->SetStream(5000 + i);  // Fixed stream for consistent app start times
        double startTime = randomStart->GetValue();
        app.Start(Seconds(startTime));
        
        std::cout << "  Device " << i << " (Node " << endDevices.Get(i)->GetId() 
                  << "): Period=" << appPeriodSeconds << "s, StartTime=" << startTime << "s" << std::endl;
    }
    
    std::cout << "✅ All devices configured\n" << std::endl;
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Phy/$ns3::EndDeviceLoraPhy/StartSending",
                                 MakeCallback(&OnTransmissionStart));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::GatewayLorawanMac/ReceivedPacket",
                                 MakeCallback(&OnPacketReceived));
    
    std::cout << "=== Simulation Run Details ===" << std::endl;
    std::cout << "ADR Mode: " << (adrMethod == ADRMethod::ON ? "CLASSICAL" : (adrMethod == ADRMethod::DDQN ? "DDQN-PER" : "OFF")) << std::endl;
    std::cout << "Output file: " << csvFileName << std::endl;
    
    Simulator::Schedule(Seconds(simulationTime), [csvFileName]() {
        WriteCSVOutput(csvFileName);
    });
    
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    
    // **CRITICAL: Verify packet count after simulation**
    uint32_t totalPacketsSent = 0;
    uint32_t totalPacketsReceived = 0;
    for (const auto& record : completedPackets) {
        totalPacketsSent++;
        if (record.packetReceived) {
            totalPacketsReceived++;
        }
    }
    
    double pdr = totalPacketsSent > 0 ? (100.0 * totalPacketsReceived / totalPacketsSent) : 0.0;
    
    std::cout << "\n=== SIMULATION STATISTICS ===" << std::endl;
    std::cout << "Simulation Duration: " << simulationTime << " seconds" << std::endl;
    std::cout << "Total Packets Sent: " << totalPacketsSent << std::endl;
    std::cout << "Total Packets Received: " << totalPacketsReceived << std::endl;
    std::cout << "Packet Delivery Ratio: " << std::fixed << std::setprecision(2) << pdr << "%" << std::endl;
    std::cout << "Output File: " << csvFileName << std::endl;
    std::cout << "============================\n" << std::endl;
    
    Simulator::Destroy();
    
    std::cout << "Simulation completed!" << std::endl;
}
