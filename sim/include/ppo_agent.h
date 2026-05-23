// -*- mode: c++; -*-
#pragma once
// PPO Agent: PolicyNetwork, ValueNetwork, PPOAgent


// ============================================================================
// PPO (PROXIMAL POLICY OPTIMIZATION) AGENT
// ============================================================================

/**
 * Simple Policy Network for PPO
 * Outputs mean and log_std for continuous SF/TP actions
 */
class PolicyNetwork {
private:
    static const int INPUT_SIZE = 12;
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 2;  // SF (continuous 7-12) and TP (continuous 2-14)
    
    std::vector<std::vector<double>> weights1, weights2, weightsMean, weightsLogStd;
    std::vector<double> bias1, bias2, biasMean, biasLogStd;
    double learningRate;
    std::mt19937 rng;
    
    double tanh(double x) { return std::tanh(x); }
    double relu(double x) { return std::max(0.0, x); }
    
public:
    PolicyNetwork(double lr = 0.0003) : learningRate(lr), rng(std::random_device{}()) {
        initializeWeights();
    }
    
    void initializeWeights() {
        std::normal_distribution<double> dis(0.0, 0.1);
        
        weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        weightsMean.resize(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE));
        weightsLogStd.resize(HIDDEN_SIZE, std::vector<double>(OUTPUT_SIZE));
        
        bias1.resize(HIDDEN_SIZE, 0.0);
        bias2.resize(HIDDEN_SIZE, 0.0);
        biasMean.resize(OUTPUT_SIZE, 0.0);
        biasLogStd.resize(OUTPUT_SIZE, -0.5);  // Initialize to small std
        
        for (auto& row : weights1) for (auto& w : row) w = dis(rng);
        for (auto& row : weights2) for (auto& w : row) w = dis(rng);
        for (auto& row : weightsMean) for (auto& w : row) w = dis(rng) * 0.01;
        for (auto& row : weightsLogStd) for (auto& w : row) w = dis(rng) * 0.01;
    }
    
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input) {
        // Hidden layer 1
        std::vector<double> h1(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h1[j] = bias1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                h1[j] += input[i] * weights1[i][j];
            }
            h1[j] = tanh(h1[j]);
        }
        
        // Hidden layer 2
        std::vector<double> h2(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h2[j] = bias2[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                h2[j] += h1[i] * weights2[i][j];
            }
            h2[j] = tanh(h2[j]);
        }
        
        // Output: mean and log_std
        std::vector<double> mean(OUTPUT_SIZE, 0.0);
        std::vector<double> logStd(OUTPUT_SIZE, 0.0);
        
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            mean[j] = biasMean[j];
            logStd[j] = biasLogStd[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                mean[j] += h2[i] * weightsMean[i][j];
                logStd[j] += h2[i] * weightsLogStd[i][j];
            }
            // Clamp log_std for stability
            logStd[j] = std::max(-2.0, std::min(2.0, logStd[j]));
        }
        
        return {mean, logStd};
    }
    
    void updateWeights(const std::vector<double>& gradient, double stepSize) {
        // Simplified weight update (in practice would use Adam optimizer)
        // This is a placeholder for the actual PPO update
    }
};

/**
 * Value Network for PPO (Critic)
 */
class ValueNetwork {
private:
    static const int INPUT_SIZE = 12;
    static const int HIDDEN_SIZE = 64;
    
    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    double learningRate;
    std::mt19937 rng;
    
    double tanh(double x) { return std::tanh(x); }
    
public:
    ValueNetwork(double lr = 0.001) : learningRate(lr), rng(std::random_device{}()) {
        initializeWeights();
    }
    
    void initializeWeights() {
        std::normal_distribution<double> dis(0.0, 0.1);
        
        weights1.resize(INPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        weights3.resize(HIDDEN_SIZE, std::vector<double>(1));
        
        bias1.resize(HIDDEN_SIZE, 0.0);
        bias2.resize(HIDDEN_SIZE, 0.0);
        bias3.resize(1, 0.0);
        
        for (auto& row : weights1) for (auto& w : row) w = dis(rng);
        for (auto& row : weights2) for (auto& w : row) w = dis(rng);
        for (auto& row : weights3) for (auto& w : row) w = dis(rng);
    }
    
    double forward(const std::vector<double>& input) {
        std::vector<double> h1(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h1[j] = bias1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                h1[j] += input[i] * weights1[i][j];
            }
            h1[j] = tanh(h1[j]);
        }
        
        std::vector<double> h2(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h2[j] = bias2[j];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                h2[j] += h1[i] * weights2[i][j];
            }
            h2[j] = tanh(h2[j]);
        }
        
        double value = bias3[0];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            value += h2[i] * weights3[i][0];
        }
        return value;
    }
};

/**
 * PPO Agent for LoRaWAN ADR
 * Uses continuous action space for SF and TP
 */
class PPOAgent {
private:
    PolicyNetwork actor;
    ValueNetwork critic;
    
    double clipEpsilon = 0.2;
    double gamma = 0.95;
    double gaeLambda = 0.95;
    double entropyCoeff = 0.01;
    
    std::mt19937 rng;
    
    // Trajectory buffer for PPO updates
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
    PPOAgent() : actor(0.0003), critic(0.001), rng(std::random_device{}()) {}
    
    /**
     * Build state vector (same as OptimizedDDQNAgent)
     */
    std::vector<double> buildState(
        const DeviceHistory& history,
        uint8_t currentSF,
        double currentTP,
        uint8_t currentChannel,
        double currentTime)
    {
        double timeSinceLastSuccess = currentTime - history.lastSuccessTime;
        
        return {
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
            std::fmod(currentTime / 3600.0, 24.0) / 24.0
        };
    }
    
    /**
     * Select action using policy network
     * Returns (SF, TP, logProb)
     */
    std::tuple<uint8_t, double, double> selectAction(const std::vector<double>& state) {
        auto [mean, logStd] = actor.forward(state);
        
        // Sample from Gaussian
        std::normal_distribution<double> dist(0.0, 1.0);
        
        double sfNoise = dist(rng);
        double tpNoise = dist(rng);
        
        // SF: map to [7, 12]
        double sfContinuous = mean[0] + sfNoise * std::exp(logStd[0]);
        sfContinuous = 9.5 + 2.5 * std::tanh(sfContinuous);  // Center at 9.5, range [7, 12]
        uint8_t sf = (uint8_t)std::round(std::max(7.0, std::min(12.0, sfContinuous)));
        
        // TP: map to [2, 14]
        double tpContinuous = mean[1] + tpNoise * std::exp(logStd[1]);
        tpContinuous = 8.0 + 6.0 * std::tanh(tpContinuous);  // Center at 8, range [2, 14]
        double tp = std::max(2.0, std::min(14.0, tpContinuous));
        
        // Calculate log probability
        double logProb = gaussianLogProb(sfContinuous, mean[0], logStd[0]) +
                        gaussianLogProb(tpContinuous, mean[1], logStd[1]);
        
        return {sf, tp, logProb};
    }
    
    void storeTransition(const std::vector<double>& state, uint8_t sf, double tp,
                         double logProb, double reward, bool done) {
        Transition t;
        t.state = state;
        t.sfAction = sf;
        t.tpAction = tp;
        t.logProb = logProb;
        t.reward = reward;
        t.value = critic.forward(state);
        t.done = done;
        trajectoryBuffer.push_back(t);
    }
    
    void update() {
        if (trajectoryBuffer.size() < 32) return;
        
        // Compute returns and advantages using GAE
        std::vector<double> returns(trajectoryBuffer.size());
        std::vector<double> advantages(trajectoryBuffer.size());
        
        double lastValue = 0.0;
        double lastGae = 0.0;
        
        for (int t = trajectoryBuffer.size() - 1; t >= 0; t--) {
            double nextValue = (t == (int)trajectoryBuffer.size() - 1) ? 0.0 : trajectoryBuffer[t + 1].value;
            double delta = trajectoryBuffer[t].reward + gamma * nextValue - trajectoryBuffer[t].value;
            lastGae = delta + gamma * gaeLambda * lastGae;
            advantages[t] = lastGae;
            returns[t] = advantages[t] + trajectoryBuffer[t].value;
        }
        
        // Normalize advantages
        double advMean = 0.0, advStd = 0.0;
        for (double a : advantages) advMean += a;
        advMean /= advantages.size();
        for (double a : advantages) advStd += (a - advMean) * (a - advMean);
        advStd = std::sqrt(advStd / advantages.size() + 1e-8);
        for (double& a : advantages) a = (a - advMean) / advStd;
        
        // PPO update would go here (simplified - just clear buffer)
        trajectoryBuffer.clear();
    }
};

// Global PPO agents map
std::map<uint32_t, std::unique_ptr<PPOAgent>> ppoAgents;

// ============================================================================
// MULTI-AGENT RL (MARL) with Independent Q-Learning + Coordination
// ============================================================================

/**
 * MARL Agent using Independent Q-Learning with coordination signals
 * Each device learns independently but receives information about other devices
 */
