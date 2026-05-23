// -*- mode: c++; -*-
#pragma once
// Dueling DDQN: ExtendedActionSpace, DuelingQNetwork, OptimizedDDQNAgent

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
            // Actions 6-11: Pure TP change (8, 9, 11, 13, 14, 17 dBm)
            // NOTE: Using minimum 8 dBm to avoid channel power violations
            int tpLevels[] = {8, 9, 11, 13, 14, 17};
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
                     const DeviceHistory& history, double currentTime,
                     double congestionLevel = 0.5) {  // NEW: congestion parameter
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        // ========================================
        // LOW CONGESTION (< 0.2): Prefer SF7, minimal exploration
        // ========================================
        if (congestionLevel < 0.2) {
            // Very low congestion - mostly exploit, minimal exploration
            double adjustedEpsilon = epsilon * 0.2;  // Very low exploration
            
            if (dist(rng) < adjustedEpsilon) {
                // Only explore SF7-SF8 in low congestion
                std::uniform_int_distribution<int> lowSfDist(0, 1);
                return lowSfDist(rng);
            }
            
            // Exploitation with strong SF7 preference
            std::vector<double> qValues = mainNetwork.forward(state);
            qValues[0] += 15.0;  // Strong preference for SF7
            qValues[1] += 8.0;   // SF8 is second choice
            qValues[2] += 3.0;   // SF9 is third
            
            return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
        }
        
        // ========================================
        // MEDIUM CONGESTION (0.2 - 0.5): Reduced exploration, prefer low SF
        // ========================================
        if (congestionLevel < 0.5) {
            // Reduced exploration in medium congestion
            double adjustedEpsilon = epsilon * 0.4;
            
            if (dist(rng) < adjustedEpsilon) {
                // Explore SF7-SF9 only (actions 0-2)
                std::uniform_int_distribution<int> medSfDist(0, 2);
                return medSfDist(rng);
            }
            
            std::vector<double> qValues = mainNetwork.forward(state);
            
            // Bias towards lower SFs
            qValues[0] += 8.0;   // SF7
            qValues[1] += 5.0;   // SF8
            qValues[2] += 2.0;   // SF9
            
            return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
        }
        
        // ========================================
        // HIGH CONGESTION: Full interference-aware behavior (original code)
        // ========================================
        
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
        
        // ========================================
        // NEW: High SNR + High SF + Low PDR = Try Lower SF
        // ========================================
        double lastSnr = history.getLastSnr();
        uint8_t currentSFFromState = (uint8_t)(state[8] * 5.0 + 7.0);  // Decode from state
        
        if (lastSnr > 20.0 && currentSFFromState > 8 && 
            history.getRecentPdr() > 0.2 && history.getRecentPdr() < 0.6) {
            // Good SNR, using high SF, mediocre PDR - try reducing SF
            if (dist(rng) < 0.4) {  // 40% chance to explore lower SF
                // Actions 0-2 correspond to SF 7, 8, 9 (keep SF low)
                std::uniform_int_distribution<int> lowSfDist(0, 2);
                int action = lowSfDist(rng);
                std::cout << "🔬 Exploring lower SF for high-SNR device (Node " << nodeId 
                          << ", SNR=" << lastSnr << "dB, CurrentSF=" << (int)currentSFFromState << ")" << std::endl;
                return action;
            }
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
 * ESTIMATE NETWORK CONGESTION from observable metrics
 * Returns value 0.0 (no congestion) to 1.0 (high congestion)
 * REVISED: Better medium-traffic detection for 20 devices @ 60s period
 */
