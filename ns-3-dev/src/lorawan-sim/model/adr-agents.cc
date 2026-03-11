#include "adr-agents.h"
#include <numeric>
#include <iostream>
#include <ctime>

namespace ns3 {
namespace lorawan {

// ============================================================================
// DuelingQNetwork Implementation
// ============================================================================

DuelingQNetwork::DuelingQNetwork(double lr) : learningRate(lr) {
    initializeWeights();
}

void DuelingQNetwork::initializeWeights() {
    std::mt19937 gen(std::random_device{}());
    
    // Shared layers
    double scale1 = std::sqrt(2.0 / INPUT_SIZE);
    sharedWeights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    sharedBias1.resize(HIDDEN_SIZE, 0.01);
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            std::normal_distribution<> d(0, scale1);
            sharedWeights1[i][j] = d(gen);
        }
    }
    
    double scale2 = std::sqrt(2.0 / HIDDEN_SIZE);
    sharedWeights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE / 2));
    sharedBias2.resize(HIDDEN_SIZE / 2, 0.01);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
            std::normal_distribution<> d(0, scale2);
            sharedWeights2[i][j] = d(gen);
        }
    }
    
    // Value stream
    double scaleV = std::sqrt(2.0 / (HIDDEN_SIZE / 2));
    valueWeights1.resize(HIDDEN_SIZE / 2, std::vector<double>(VALUE_SIZE));
    valueBias1.resize(VALUE_SIZE, 0.01);
    for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
        for (int j = 0; j < VALUE_SIZE; j++) {
            std::normal_distribution<> d(0, scaleV);
            valueWeights1[i][j] = d(gen);
        }
    }
    valueWeights2.resize(VALUE_SIZE, std::vector<double>(1));
    valueBias2.resize(1, 0.0);
    for (int i = 0; i < VALUE_SIZE; i++) {
        std::normal_distribution<> d(0, std::sqrt(2.0 / VALUE_SIZE));
        valueWeights2[i][0] = d(gen);
    }
    
    // Advantage stream
    double scaleA = std::sqrt(2.0 / (HIDDEN_SIZE / 2));
    advWeights1.resize(HIDDEN_SIZE / 2, std::vector<double>(ADVANTAGE_SIZE));
    advBias1.resize(ADVANTAGE_SIZE, 0.01);
    for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
        for (int j = 0; j < ADVANTAGE_SIZE; j++) {
            std::normal_distribution<> d(0, scaleA);
            advWeights1[i][j] = d(gen);
        }
    }
    advWeights2.resize(ADVANTAGE_SIZE, std::vector<double>(OUTPUT_SIZE));
    advBias2.resize(OUTPUT_SIZE, 0.0);
    for (int i = 0; i < ADVANTAGE_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            std::normal_distribution<> d(0, std::sqrt(2.0 / ADVANTAGE_SIZE));
            advWeights2[i][j] = d(gen);
        }
    }
}

std::vector<double> DuelingQNetwork::forward(const std::vector<double>& input) {
    // Shared layer 1
    std::vector<double> hidden1(HIDDEN_SIZE, 0.0);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (size_t i = 0; i < input.size() && i < INPUT_SIZE; i++) {
            hidden1[j] += input[i] * sharedWeights1[i][j];
        }
        hidden1[j] = leakyRelu(hidden1[j] + sharedBias1[j]);
    }
    
    // Shared layer 2
    std::vector<double> hidden2(HIDDEN_SIZE / 2, 0.0);
    for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden2[j] += hidden1[i] * sharedWeights2[i][j];
        }
        hidden2[j] = leakyRelu(hidden2[j] + sharedBias2[j]);
    }
    
    // Value stream
    std::vector<double> valueHidden(VALUE_SIZE, 0.0);
    for (int j = 0; j < VALUE_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
            valueHidden[j] += hidden2[i] * valueWeights1[i][j];
        }
        valueHidden[j] = leakyRelu(valueHidden[j] + valueBias1[j]);
    }
    double value = valueBias2[0];
    for (int i = 0; i < VALUE_SIZE; i++) {
        value += valueHidden[i] * valueWeights2[i][0];
    }
    
    // Advantage stream
    std::vector<double> advHidden(ADVANTAGE_SIZE, 0.0);
    for (int j = 0; j < ADVANTAGE_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
            advHidden[j] += hidden2[i] * advWeights1[i][j];
        }
        advHidden[j] = leakyRelu(advHidden[j] + advBias1[j]);
    }
    std::vector<double> advantages(OUTPUT_SIZE, 0.0);
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        for (int i = 0; i < ADVANTAGE_SIZE; i++) {
            advantages[j] += advHidden[i] * advWeights2[i][j];
        }
        advantages[j] += advBias2[j];
    }
    
    // Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    double advMean = std::accumulate(advantages.begin(), advantages.end(), 0.0) / OUTPUT_SIZE;
    std::vector<double> qValues(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        qValues[i] = value + advantages[i] - advMean;
    }
    return qValues;
}

void DuelingQNetwork::copyFrom(const DuelingQNetwork& other) {
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

// ============================================================================
// OptimizedDDQNAgent Implementation
// ============================================================================

OptimizedDDQNAgent::OptimizedDDQNAgent(double lr, double eps, double epsDecay,
                                       double epsMin, double g, int targetFreq)
    : mainNetwork(lr), targetNetwork(lr), replayBuffer(20000, 0.6),
      epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin),
      gamma(g), targetUpdateFreq(targetFreq), updateCounter(0),
      rng(std::random_device{}()) {
    targetNetwork.copyFrom(mainNetwork);
}

std::vector<double> OptimizedDDQNAgent::buildRealisticState(
    const DeviceHistory& history, uint8_t currentSF, double currentTP,
    uint8_t currentChannel, double currentTime) {
    
    std::vector<double> state(12, 0.0);
    
    // 0: Normalized RSSI
    state[0] = (history.getLastRssi() + 110.0) / 60.0;
    
    // 1: Normalized SNR
    state[1] = (history.getLastSnr() + 20.0) / 40.0;
    
    // 2: RSSI variance (normalized)
    state[2] = std::min(1.0, history.getRssiVariance() / 15.0);
    
    // 3: SNR trend
    state[3] = (history.getSnrTrend() + 5.0) / 10.0;
    
    // 4: Recent PDR
    state[4] = history.getRecentPdr();
    
    // 5: PDR trend
    state[5] = (history.getPdrTrend() + 1.0) / 2.0;
    
    // 6: Consecutive losses (normalized)
    state[6] = std::min(1.0, history.consecutiveLosses / 5.0);
    
    // 7: Time since success (normalized to ~minutes)
    double timeSinceSuccess = currentTime - history.lastSuccessTime;
    state[7] = std::min(1.0, timeSinceSuccess / 600.0);
    
    // 8: Current SF (normalized)
    state[8] = (currentSF - 7.0) / 5.0;
    
    // 9: Current TX power (normalized)
    state[9] = (currentTP - 8.0) / 9.0;
    
    // 10: Current channel (normalized)
    state[10] = currentChannel / 7.0;
    
    // 11: Channel changed recently
    state[11] = (history.currentChannel != history.previousChannel) ? 1.0 : 0.0;
    
    return state;
}

int OptimizedDDQNAgent::selectAction(const std::vector<double>& state, uint32_t nodeId,
                                     const DeviceHistory& history, double currentTime) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    if (dist(rng) < epsilon) {
        // Smart exploration: prefer reasonable actions
        std::vector<int> reasonableActions;
        double recentPdr = history.getRecentPdr();
        
        if (recentPdr < 0.5) {
            // Low PDR: Try higher SF or higher power
            for (int i = 0; i < 6; i++) reasonableActions.push_back(i);     // SF changes
            for (int i = 8; i < 12; i++) reasonableActions.push_back(i);    // Higher TP
            for (int i = 12; i < 20; i++) reasonableActions.push_back(i);   // Channel changes
        } else {
            // High PDR: Can try lower SF to save energy
            reasonableActions.push_back(0);  // SF7
            reasonableActions.push_back(1);  // SF8
            for (int i = 6; i < 10; i++) reasonableActions.push_back(i);    // Lower TP
        }
        
        if (!reasonableActions.empty()) {
            std::uniform_int_distribution<> actionDist(0, reasonableActions.size() - 1);
            return reasonableActions[actionDist(rng)];
        }
        
        std::uniform_int_distribution<> fullDist(0, 47);
        return fullDist(rng);
    }
    
    // Greedy action with curiosity
    auto qValues = mainNetwork.forward(state);
    std::string stateKey = discretizeState(state);
    stateVisitCounts[stateKey]++;
    
    // Add curiosity bonus for less-visited states
    for (int i = 0; i < 48; i++) {
        std::string futureKey = stateKey + "_a" + std::to_string(i);
        int visits = stateVisitCounts[futureKey];
        qValues[i] += curiosityBonus / (1.0 + std::sqrt(visits));
    }
    
    return std::max_element(qValues.begin(), qValues.end()) - qValues.begin();
}

void OptimizedDDQNAgent::addExperience(const std::vector<double>& state, int action,
                                       double reward, const std::vector<double>& nextState, bool done) {
    auto qValues = mainNetwork.forward(state);
    auto nextQValues = mainNetwork.forward(nextState);
    auto targetQValues = targetNetwork.forward(nextState);
    
    int bestNextAction = std::max_element(nextQValues.begin(), nextQValues.end()) - nextQValues.begin();
    double target = reward;
    if (!done) {
        target += gamma * targetQValues[bestNextAction];
    }
    double tdError = std::abs(target - qValues[action]) + 0.01;
    
    Experience exp(state, action, reward, nextState, done, tdError);
    replayBuffer.add(exp);
}

void OptimizedDDQNAgent::train(size_t batchSize) {
    if (replayBuffer.size() < batchSize) return;
    
    std::vector<size_t> indices;
    std::vector<double> weights;
    auto batch = replayBuffer.sample(batchSize, indices, weights);
    
    // Simplified training - just update epsilon for now
    // Full training would require gradient descent implementation
    
    epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    
    updateCounter++;
    if (updateCounter % targetUpdateFreq == 0) {
        targetNetwork.copyFrom(mainNetwork);
    }
}

void OptimizedDDQNAgent::increaseExploration() {
    epsilon = std::min(0.8, epsilon + 0.1);
}

// ============================================================================
// PolicyNetwork Implementation
// ============================================================================

PolicyNetwork::PolicyNetwork(double lr) : learningRate(lr), rng(std::random_device{}()) {
    initializeWeights();
}

void PolicyNetwork::initializeWeights() {
    std::normal_distribution<> d1(0, std::sqrt(2.0 / INPUT_SIZE));
    weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    bias1.resize(HIDDEN_SIZE, 0.01);
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights1[i][j] = d1(rng);
        }
    }
    
    std::normal_distribution<> d2(0, std::sqrt(2.0 / HIDDEN_SIZE));
    weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE / 2));
    bias2.resize(HIDDEN_SIZE / 2, 0.01);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
            weights2[i][j] = d2(rng);
        }
    }
    
    std::normal_distribution<> d3(0, 0.01);
    weightsMean.resize(HIDDEN_SIZE / 2, std::vector<double>(OUTPUT_SIZE));
    biasMean.resize(OUTPUT_SIZE, 0.0);
    weightsLogStd.resize(HIDDEN_SIZE / 2, std::vector<double>(OUTPUT_SIZE));
    biasLogStd.resize(OUTPUT_SIZE, -1.0);
    for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weightsMean[i][j] = d3(rng);
            weightsLogStd[i][j] = d3(rng);
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> PolicyNetwork::forward(const std::vector<double>& input) {
    // Hidden layer 1
    std::vector<double> h1(HIDDEN_SIZE, 0.0);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (size_t i = 0; i < input.size() && i < INPUT_SIZE; i++) {
            h1[j] += input[i] * weights1[i][j];
        }
        h1[j] = tanhFunc(h1[j] + bias1[j]);
    }
    
    // Hidden layer 2
    std::vector<double> h2(HIDDEN_SIZE / 2, 0.0);
    for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h2[j] += h1[i] * weights2[i][j];
        }
        h2[j] = tanhFunc(h2[j] + bias2[j]);
    }
    
    // Mean output
    std::vector<double> mean(OUTPUT_SIZE, 0.0);
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
            mean[j] += h2[i] * weightsMean[i][j];
        }
        mean[j] += biasMean[j];
    }
    
    // Log std output
    std::vector<double> logStd(OUTPUT_SIZE, 0.0);
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
            logStd[j] += h2[i] * weightsLogStd[i][j];
        }
        logStd[j] = std::max(-2.0, std::min(2.0, logStd[j] + biasLogStd[j]));
    }
    
    return {mean, logStd};
}

// ============================================================================
// ValueNetwork Implementation
// ============================================================================

ValueNetwork::ValueNetwork(double lr) : learningRate(lr), rng(std::random_device{}()) {
    initializeWeights();
}

void ValueNetwork::initializeWeights() {
    std::normal_distribution<> d1(0, std::sqrt(2.0 / INPUT_SIZE));
    weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
    bias1.resize(HIDDEN_SIZE, 0.01);
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights1[i][j] = d1(rng);
        }
    }
    
    std::normal_distribution<> d2(0, std::sqrt(2.0 / HIDDEN_SIZE));
    weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE / 2));
    bias2.resize(HIDDEN_SIZE / 2, 0.01);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
            weights2[i][j] = d2(rng);
        }
    }
    
    std::normal_distribution<> d3(0, 0.01);
    weights3.resize(HIDDEN_SIZE / 2, std::vector<double>(1));
    bias3.resize(1, 0.0);
    for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
        weights3[i][0] = d3(rng);
    }
}

double ValueNetwork::forward(const std::vector<double>& input) {
    std::vector<double> h1(HIDDEN_SIZE, 0.0);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (size_t i = 0; i < input.size() && i < INPUT_SIZE; i++) {
            h1[j] += input[i] * weights1[i][j];
        }
        h1[j] = tanhFunc(h1[j] + bias1[j]);
    }
    
    std::vector<double> h2(HIDDEN_SIZE / 2, 0.0);
    for (int j = 0; j < HIDDEN_SIZE / 2; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h2[j] += h1[i] * weights2[i][j];
        }
        h2[j] = tanhFunc(h2[j] + bias2[j]);
    }
    
    double value = bias3[0];
    for (int i = 0; i < HIDDEN_SIZE / 2; i++) {
        value += h2[i] * weights3[i][0];
    }
    return value;
}

// ============================================================================
// PPOAgent Implementation
// ============================================================================

PPOAgent::PPOAgent() : actor(0.0003), critic(0.001), rng(std::random_device{}()) {}

std::vector<double> PPOAgent::buildState(
    const DeviceHistory& history, uint8_t currentSF, double currentTP,
    uint8_t currentChannel, double currentTime) {
    
    std::vector<double> state(12, 0.0);
    
    state[0] = (history.getLastRssi() + 110.0) / 60.0;
    state[1] = (history.getLastSnr() + 20.0) / 40.0;
    state[2] = std::min(1.0, history.getRssiVariance() / 15.0);
    state[3] = (history.getSnrTrend() + 5.0) / 10.0;
    state[4] = history.getRecentPdr();
    state[5] = (history.getPdrTrend() + 1.0) / 2.0;
    state[6] = std::min(1.0, history.consecutiveLosses / 5.0);
    double timeSinceSuccess = currentTime - history.lastSuccessTime;
    state[7] = std::min(1.0, timeSinceSuccess / 600.0);
    state[8] = (currentSF - 7.0) / 5.0;
    state[9] = (currentTP - 8.0) / 9.0;
    state[10] = currentChannel / 7.0;
    state[11] = (history.currentChannel != history.previousChannel) ? 1.0 : 0.0;
    
    return state;
}

std::tuple<uint8_t, double, double> PPOAgent::selectAction(const std::vector<double>& state) {
    auto [means, logStds] = actor.forward(state);
    
    // Sample from Gaussian distributions
    std::normal_distribution<> distSF(means[0], std::exp(logStds[0]));
    std::normal_distribution<> distTP(means[1], std::exp(logStds[1]));
    
    double sfAction = distSF(rng);
    double tpAction = distTP(rng);
    
    // Map to discrete SF (7-12) and continuous TP (8-17)
    uint8_t sf = static_cast<uint8_t>(std::round(std::max(0.0, std::min(5.0, sfAction * 5.0)))) + 7;
    double tp = std::max(8.0, std::min(17.0, tpAction * 9.0 + 8.0));
    
    // Calculate log probability
    double logProb = gaussianLogProb(sfAction, means[0], logStds[0]) +
                    gaussianLogProb(tpAction, means[1], logStds[1]);
    
    return {sf, tp, logProb};
}

void PPOAgent::storeTransition(const std::vector<double>& state, uint8_t sf, double tp,
                               double logProb, double reward, bool done) {
    Transition t;
    t.state = state;
    t.sfAction = (sf - 7.0) / 5.0;
    t.tpAction = (tp - 8.0) / 9.0;
    t.logProb = logProb;
    t.reward = reward;
    t.value = critic.forward(state);
    t.done = done;
    trajectoryBuffer.push_back(t);
}

void PPOAgent::update() {
    if (trajectoryBuffer.size() < 32) return;
    
    // Compute GAE (Generalized Advantage Estimation)
    std::vector<double> advantages(trajectoryBuffer.size());
    std::vector<double> returns(trajectoryBuffer.size());
    
    double gae = 0.0;
    for (int i = trajectoryBuffer.size() - 1; i >= 0; i--) {
        double nextValue = (i == (int)trajectoryBuffer.size() - 1) ? 0.0 : trajectoryBuffer[i + 1].value;
        double delta = trajectoryBuffer[i].reward + gamma * nextValue * (1 - trajectoryBuffer[i].done) - trajectoryBuffer[i].value;
        gae = delta + gamma * gaeLambda * (1 - trajectoryBuffer[i].done) * gae;
        advantages[i] = gae;
        returns[i] = gae + trajectoryBuffer[i].value;
    }
    
    // Normalize advantages
    double meanAdv = std::accumulate(advantages.begin(), advantages.end(), 0.0) / advantages.size();
    double varAdv = 0.0;
    for (double a : advantages) varAdv += (a - meanAdv) * (a - meanAdv);
    double stdAdv = std::sqrt(varAdv / advantages.size() + 1e-8);
    for (double& a : advantages) a = (a - meanAdv) / stdAdv;
    
    // Clear trajectory buffer after update
    trajectoryBuffer.clear();
}

// ============================================================================
// MARLAgent Implementation
// ============================================================================

MARLAgent::MARLAgent(double lr, double eps, double epsDecay, double epsMin, double g, int targetFreq)
    : mainNetwork(lr), targetNetwork(lr), replayBuffer(20000, 0.6),
      epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin),
      gamma(g), targetUpdateFreq(targetFreq), updateCounter(0),
      rng(std::random_device{}()) {
    targetNetwork.copyFrom(mainNetwork);
}

std::vector<double> MARLAgent::buildCoordinatedState(
    const DeviceHistory& history, uint8_t currentSF, double currentTP,
    uint8_t currentChannel, double currentTime, uint32_t nodeId,
    const std::map<uint32_t, DeviceHistory>& allHistories,
    const std::map<uint32_t, uint8_t>& allSFs,
    const std::map<uint32_t, uint8_t>& allChannels) {
    
    std::vector<double> state(12, 0.0);
    
    // Same base state as DDQN
    state[0] = (history.getLastRssi() + 110.0) / 60.0;
    state[1] = (history.getLastSnr() + 20.0) / 40.0;
    state[2] = std::min(1.0, history.getRssiVariance() / 15.0);
    state[3] = (history.getSnrTrend() + 5.0) / 10.0;
    state[4] = history.getRecentPdr();
    state[5] = (history.getPdrTrend() + 1.0) / 2.0;
    state[6] = std::min(1.0, history.consecutiveLosses / 5.0);
    double timeSinceSuccess = currentTime - history.lastSuccessTime;
    state[7] = std::min(1.0, timeSinceSuccess / 600.0);
    state[8] = (currentSF - 7.0) / 5.0;
    state[9] = (currentTP - 8.0) / 9.0;
    
    // Multi-agent coordination signals
    // Count agents on same channel
    int sameChannelCount = 0;
    int sameSFCount = 0;
    for (const auto& [id, ch] : allChannels) {
        if (id != nodeId && ch == currentChannel) {
            sameChannelCount++;
            auto sfIt = allSFs.find(id);
            if (sfIt != allSFs.end() && sfIt->second == currentSF) {
                sameSFCount++;
            }
        }
    }
    state[10] = std::min(1.0, sameChannelCount / 5.0);
    state[11] = std::min(1.0, sameSFCount / 3.0);
    
    return state;
}

int MARLAgent::selectAction(const std::vector<double>& state, uint8_t currentChannel,
                           const std::map<uint32_t, uint8_t>& allChannels, uint32_t nodeId) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    if (dist(rng) < epsilon) {
        // Smart exploration: avoid crowded channels
        std::map<int, int> channelCounts;
        for (int i = 0; i < 8; i++) channelCounts[i] = 0;
        for (const auto& [id, ch] : allChannels) {
            if (id != nodeId) channelCounts[ch]++;
        }
        
        // Find least crowded channel
        int leastCrowded = 0;
        int minCount = channelCounts[0];
        for (int i = 1; i < 8; i++) {
            if (channelCounts[i] < minCount) {
                minCount = channelCounts[i];
                leastCrowded = i;
            }
        }
        
        // Prefer actions that lead to less crowded channels
        std::vector<int> goodActions;
        for (int i = 12; i < 20; i++) {
            int targetCh = i - 12;
            if (channelCounts[targetCh] <= minCount + 1) {
                goodActions.push_back(i);
            }
        }
        
        if (!goodActions.empty()) {
            std::uniform_int_distribution<> actionDist(0, goodActions.size() - 1);
            return goodActions[actionDist(rng)];
        }
        
        std::uniform_int_distribution<> fullDist(0, 47);
        return fullDist(rng);
    }
    
    auto qValues = mainNetwork.forward(state);
    return std::max_element(qValues.begin(), qValues.end()) - qValues.begin();
}

void MARLAgent::addExperience(const std::vector<double>& state, int action, double reward,
                             const std::vector<double>& nextState, bool done) {
    auto qValues = mainNetwork.forward(state);
    auto nextQValues = mainNetwork.forward(nextState);
    auto targetQValues = targetNetwork.forward(nextState);
    
    int bestNextAction = std::max_element(nextQValues.begin(), nextQValues.end()) - nextQValues.begin();
    double target = reward;
    if (!done) {
        target += gamma * targetQValues[bestNextAction];
    }
    double tdError = std::abs(target - qValues[action]) + 0.01;
    
    Experience exp(state, action, reward, nextState, done, tdError);
    replayBuffer.add(exp);
}

void MARLAgent::train(size_t batchSize) {
    if (replayBuffer.size() < batchSize) return;
    
    std::vector<size_t> indices;
    std::vector<double> weights;
    auto batch = replayBuffer.sample(batchSize, indices, weights);
    
    epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    
    updateCounter++;
    if (updateCounter % targetUpdateFreq == 0) {
        targetNetwork.copyFrom(mainNetwork);
    }
}

void MARLAgent::increaseExploration() {
    epsilon = std::min(0.8, epsilon + 0.1);
}

// ============================================================================
// Interference-Aware Reward Function
// ============================================================================

double calculateInterferenceAwareReward(
    uint32_t nodeId, bool packetSuccess, double preTxRssi, double rxSnr,
    double rssiVariance, uint32_t consecutiveLosses, uint8_t currentSF,
    double currentTP, uint8_t currentChannel, uint8_t previousChannel,
    double recentPdr, double pdrTrend, double timeSinceLastSuccess) {
    
    double reward = 0.0;
    
    // Analyze loss cause
    auto indicators = CollisionIndicators::analyze(
        preTxRssi, rxSnr, packetSuccess, rssiVariance, consecutiveLosses, recentPdr);
    
    if (packetSuccess) {
        reward += 2.0;
        
        // Bonus for energy efficiency
        double sfPenalty = (currentSF - 7) * 0.1;
        double tpPenalty = (currentTP - 8) * 0.05;
        reward -= sfPenalty + tpPenalty;
        
        // Bonus for good SNR margin
        double snrThresholds[] = {-7.5, -10.0, -12.5, -15.0, -17.5, -20.0};
        double margin = rxSnr - snrThresholds[currentSF - 7];
        if (margin > 10.0) {
            reward += 0.3;  // Good margin, might reduce SF
        }
        
        // Recovery bonus
        if (consecutiveLosses > 0 && consecutiveLosses < 3) {
            reward += 0.3;
        }
    } else {
        // Base failure penalty
        reward -= 1.0;
        
        // Collision-specific feedback
        if (indicators.likelyCollision) {
            // Lost due to collision despite good signal
            reward -= 0.5;
            
            // Mild penalty if we didn't try channel hopping
            if (currentChannel == previousChannel) {
                reward -= 0.2;
            }
        }
        
        if (indicators.likelyWeakSignal) {
            // Lost due to weak signal
            reward -= 0.3;
            
            // Penalty for using low SF with weak signal
            if (currentSF < 10 && preTxRssi < -115.0) {
                reward -= 0.3;
            }
        }
        
        if (indicators.likelyInterference) {
            reward -= 0.4;
        }
        
        // Consecutive loss penalty
        if (consecutiveLosses > 3) {
            reward -= 0.3 * (consecutiveLosses - 3);
        }
        
        // Long silence penalty
        if (timeSinceLastSuccess > 300.0) {
            reward -= 0.5;
        }
    }
    
    // PDR trend bonus/penalty
    if (pdrTrend > 0.1) reward += 0.2;
    else if (pdrTrend < -0.1) reward -= 0.2;
    
    // Clip reward
    return std::max(-5.0, std::min(5.0, reward));
}

} // namespace lorawan
} // namespace ns3
