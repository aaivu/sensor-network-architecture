#ifndef ADR_AGENTS_H
#define ADR_AGENTS_H

#include "adr-ddqn.h"
#include <vector>
#include <memory>
#include <random>
#include <map>
#include <deque>
#include <cmath>
#include <algorithm>
#include <string>

namespace ns3 {
namespace lorawan {

// ============================================================================
// DEVICE HISTORY - Tracks observable metrics for realistic state representation
// ============================================================================
struct DeviceHistory {
    std::deque<double> rssiHistory;
    std::deque<double> snrHistory;
    std::deque<bool> packetResults;
    std::deque<double> timestamps;
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

// ============================================================================
// COLLISION DETECTION HEURISTIC
// ============================================================================
struct CollisionIndicators {
    double collisionProbability;
    bool likelyCollision;
    bool likelyWeakSignal;
    bool likelyInterference;
    
    static CollisionIndicators analyze(
        double preTxRssi, double rxSnr, bool packetSuccess,
        double rssiVariance, uint32_t consecutiveLosses, double recentPdr) {
        CollisionIndicators result;
        result.collisionProbability = 0.0;
        result.likelyCollision = false;
        result.likelyWeakSignal = false;
        result.likelyInterference = false;
        
        if (!packetSuccess) {
            if (preTxRssi > -105.0) {
                result.collisionProbability += 0.4;
                result.likelyCollision = true;
            }
            if (rssiVariance > 8.0) {
                result.collisionProbability += 0.3;
                result.likelyInterference = true;
            }
            if (rxSnr > 5.0 && consecutiveLosses < 3) {
                result.collisionProbability += 0.2;
                result.likelyCollision = true;
            }
            if (rxSnr < -5.0) {
                result.likelyWeakSignal = true;
                result.collisionProbability *= 0.5;
            }
            if (recentPdr > 0.3 && recentPdr < 0.7 && rssiVariance > 5.0) {
                result.likelyInterference = true;
            }
        }
        result.collisionProbability = std::min(1.0, result.collisionProbability);
        return result;
    }
};

// ============================================================================
// EXTENDED ACTION SPACE with Channel Selection (48 actions)
// ============================================================================
class ExtendedActionSpace {
public:
    static const int NUM_ACTIONS = 48;
    
    struct ActionResult {
        int newSF;
        int newTP;
        int channelDelta;
        bool forceChannelHop;
    };
    
    static ActionResult decodeAction(int action, uint8_t currentSF, double currentTP, uint8_t currentChannel) {
        ActionResult result = {-1, -1, 0, false};
        
        if (action < 6) {
            result.newSF = 7 + action;
        } else if (action < 12) {
            int tpLevels[] = {8, 9, 11, 13, 14, 17};
            result.newTP = tpLevels[action - 6];
        } else if (action < 20) {
            int targetChannel = action - 12;
            result.channelDelta = targetChannel - currentChannel;
            result.forceChannelHop = true;
        } else if (action < 36) {
            int idx = action - 20;
            int sfIdx = idx / 3;
            int chAction = idx % 3;
            result.newSF = 7 + sfIdx;
            if (chAction == 1) result.channelDelta = 1;
            else if (chAction == 2) result.forceChannelHop = true;
        } else {
            int idx = action - 36;
            int sfIdx = idx / 2;
            int tpIdx = idx % 2;
            result.newSF = 7 + sfIdx;
            result.newTP = tpIdx ? 14 : 8;
        }
        return result;
    }
};

// ============================================================================
// DUELING DQN NETWORK
// ============================================================================
class DuelingQNetwork {
private:
    static const int INPUT_SIZE = 12;
    static const int HIDDEN_SIZE = 64;
    static const int VALUE_SIZE = 32;
    static const int ADVANTAGE_SIZE = 32;
    static const int OUTPUT_SIZE = 48;
    
    std::vector<std::vector<double>> sharedWeights1, sharedWeights2;
    std::vector<double> sharedBias1, sharedBias2;
    std::vector<std::vector<double>> valueWeights1, valueWeights2;
    std::vector<double> valueBias1, valueBias2;
    std::vector<std::vector<double>> advWeights1, advWeights2;
    std::vector<double> advBias1, advBias2;
    double learningRate;
    
    double leakyRelu(double x, double alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }
    
public:
    DuelingQNetwork(double lr = 0.0005);
    void initializeWeights();
    std::vector<double> forward(const std::vector<double>& input);
    void copyFrom(const DuelingQNetwork& other);
};

// ============================================================================
// OPTIMIZED DDQN AGENT FOR SHORT RANGE
// ============================================================================
class OptimizedDDQNAgent {
private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;
    std::map<std::string, int> stateVisitCounts;
    double curiosityBonus = 0.1;
    std::mt19937 rng;
    
    std::string discretizeState(const std::vector<double>& state) {
        std::string key;
        for (double v : state) key += std::to_string((int)(v * 10)) + "_";
        return key;
    }
    
public:
    OptimizedDDQNAgent(double lr = 0.0005, double eps = 0.3, double epsDecay = 0.999,
                      double epsMin = 0.05, double g = 0.95, int targetFreq = 50);
    
    std::vector<double> buildRealisticState(
        const DeviceHistory& history, uint8_t currentSF, double currentTP,
        uint8_t currentChannel, double currentTime);
    
    int selectAction(const std::vector<double>& state, uint32_t nodeId,
                     const DeviceHistory& history, double currentTime);
    
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done);
    void train(size_t batchSize = 32);
    double getEpsilon() const { return epsilon; }
    void increaseExploration();
};

// ============================================================================
// PPO POLICY NETWORK
// ============================================================================
class PolicyNetwork {
private:
    static const int INPUT_SIZE = 12;
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 2;
    
    std::vector<std::vector<double>> weights1, weights2, weightsMean, weightsLogStd;
    std::vector<double> bias1, bias2, biasMean, biasLogStd;
    double learningRate;
    std::mt19937 rng;
    
    double tanhFunc(double x) { return std::tanh(x); }
    
public:
    PolicyNetwork(double lr = 0.0003);
    void initializeWeights();
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input);
};

// ============================================================================
// PPO VALUE NETWORK (Critic)
// ============================================================================
class ValueNetwork {
private:
    static const int INPUT_SIZE = 12;
    static const int HIDDEN_SIZE = 64;
    
    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    double learningRate;
    std::mt19937 rng;
    
    double tanhFunc(double x) { return std::tanh(x); }
    
public:
    ValueNetwork(double lr = 0.001);
    void initializeWeights();
    double forward(const std::vector<double>& input);
};

// ============================================================================
// PPO AGENT
// ============================================================================
class PPOAgent {
private:
    PolicyNetwork actor;
    ValueNetwork critic;
    double clipEpsilon = 0.2;
    double gamma = 0.95;
    double gaeLambda = 0.95;
    double entropyCoeff = 0.01;
    std::mt19937 rng;
    
    struct Transition {
        std::vector<double> state;
        double sfAction, tpAction;
        double logProb;
        double reward;
        double value;
        bool done;
    };
    std::vector<Transition> trajectoryBuffer;
    
    double gaussianLogProb(double x, double mean, double logStd) {
        double std = std::exp(logStd);
        return -0.5 * std::pow((x - mean) / std, 2) - logStd - 0.5 * std::log(2 * M_PI);
    }
    
public:
    PPOAgent();
    
    std::vector<double> buildState(
        const DeviceHistory& history, uint8_t currentSF, double currentTP,
        uint8_t currentChannel, double currentTime);
    
    std::tuple<uint8_t, double, double> selectAction(const std::vector<double>& state);
    
    void storeTransition(const std::vector<double>& state, uint8_t sf, double tp,
                         double logProb, double reward, bool done);
    void update();
};

// ============================================================================
// MARL AGENT using Independent Q-Learning with coordination signals
// ============================================================================
class MARLAgent {
private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;
    std::mt19937 rng;
    std::map<uint32_t, int> otherAgentActions;
    std::map<uint32_t, uint8_t> otherAgentChannels;
    
public:
    MARLAgent(double lr = 0.0005, double eps = 0.3, double epsDecay = 0.999,
              double epsMin = 0.05, double g = 0.95, int targetFreq = 50);
    
    std::vector<double> buildCoordinatedState(
        const DeviceHistory& history, uint8_t currentSF, double currentTP,
        uint8_t currentChannel, double currentTime, uint32_t nodeId,
        const std::map<uint32_t, DeviceHistory>& allHistories,
        const std::map<uint32_t, uint8_t>& allSFs,
        const std::map<uint32_t, uint8_t>& allChannels);
    
    int selectAction(const std::vector<double>& state, uint8_t currentChannel,
                     const std::map<uint32_t, uint8_t>& allChannels, uint32_t nodeId);
    
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done);
    void train(size_t batchSize = 32);
    double getEpsilon() const { return epsilon; }
    void increaseExploration();
};

// ============================================================================
// INTERFERENCE-AWARE REWARD FUNCTION
// ============================================================================
double calculateInterferenceAwareReward(
    uint32_t nodeId, bool packetSuccess, double preTxRssi, double rxSnr,
    double rssiVariance, uint32_t consecutiveLosses, uint8_t currentSF,
    double currentTP, uint8_t currentChannel, uint8_t previousChannel,
    double recentPdr, double pdrTrend, double timeSinceLastSuccess);

} // namespace lorawan
} // namespace ns3

#endif // ADR_AGENTS_H
