// -*- mode: c++; -*-
#pragma once
// Simple DDQN-PER: Experience, SimpleQNetwork, PrioritizedReplayBuffer, DDQNPERADRAgent


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
