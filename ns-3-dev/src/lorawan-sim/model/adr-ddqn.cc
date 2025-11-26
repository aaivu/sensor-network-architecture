#include "adr-ddqn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ns3 {
namespace lorawan {

// ============================================================================
// Experience Implementation
// ============================================================================

Experience::Experience(const std::vector<double>& s, int a, double r, 
                      const std::vector<double>& ns, bool d, double p)
    : state(s), action(a), reward(r), nextState(ns), done(d), priority(p), tdError(0.0) {}

// ============================================================================
// SimpleQNetwork Implementation
// ============================================================================

SimpleQNetwork::SimpleQNetwork(double lr) : learningRate(lr) {
    initializeWeights();
}

double SimpleQNetwork::relu(double x) {
    return std::max(0.0, x);
}

double SimpleQNetwork::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

void SimpleQNetwork::initializeWeights() {
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

std::vector<double> SimpleQNetwork::forward(const std::vector<double>& input) {
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

void SimpleQNetwork::updateWeights(const std::vector<double>& targetOutput) {
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

void SimpleQNetwork::copyFrom(const SimpleQNetwork& other) {
    weights1 = other.weights1;
    weights2 = other.weights2;
    weights3 = other.weights3;
    bias1 = other.bias1;
    bias2 = other.bias2;
    bias3 = other.bias3;
}

// ============================================================================
// PrioritizedReplayBuffer Implementation
// ============================================================================

PrioritizedReplayBuffer::PrioritizedReplayBuffer(size_t maxSize, double alpha, double beta, double epsilon)
    : maxSize(maxSize), currentSize(0), position(0), alpha(alpha), beta(beta), epsilon(epsilon) {
    buffer.reserve(maxSize);
    priorities.reserve(maxSize);
}

void PrioritizedReplayBuffer::add(const Experience& experience) {
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

std::vector<Experience> PrioritizedReplayBuffer::sample(size_t batchSize, std::vector<size_t>& indices, std::vector<double>& weights) {
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

void PrioritizedReplayBuffer::updatePriorities(const std::vector<size_t>& indices, const std::vector<double>& tdErrors) {
    for (size_t i = 0; i < indices.size(); i++) {
        priorities[indices[i]] = std::abs(tdErrors[i]) + epsilon;
    }
}

size_t PrioritizedReplayBuffer::size() const {
    return currentSize;
}

// ============================================================================
// DDQNPERADRAgent Implementation
// ============================================================================

DDQNPERADRAgent::DDQNPERADRAgent(double lr, double eps, double epsDecay, 
                                 double epsMin, double gam, int targetFreq,
                                 double pdrW, double energyW)
    : mainNetwork(lr), targetNetwork(lr), replayBuffer(10000),
      epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin), gamma(gam),
      targetUpdateFreq(targetFreq), updateCounter(0),
      pdrWeight(pdrW), energyWeight(energyW), rng(std::random_device{}()) {
    
    sfActions = {7, 8, 9, 10, 11, 12};
    tpActions = {2, 8, 14};
    targetNetwork.copyFrom(mainNetwork);
}

double DDQNPERADRAgent::getRequiredSNR(int sf) {
    std::map<int, double> requiredSNR = {
        {7, -7.5}, {8, -10}, {9, -12.5}, {10, -15}, {11, -17.5}, {12, -20}
    };
    return requiredSNR[sf];
}

int DDQNPERADRAgent::getOptimalSFForDistance(double distance) {
    if (distance < 300) return 7;
    else if (distance < 600) return 8;
    else if (distance < 900) return 9;
    else if (distance < 1200) return 10;
    else if (distance < 1500) return 11;
    else return 12;
}

void DDQNPERADRAgent::increaseExploration() {
    epsilon = std::min(0.8, epsilon + 0.3);
    std::cout << "🔄 Epsilon boosted to " << epsilon << " for better exploration" << std::endl;
}

int DDQNPERADRAgent::selectAction(const std::vector<double>& state) {
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    if (dis(rng) < epsilon) {
        std::uniform_int_distribution<int> actionDis(0, 11);
        return actionDis(rng);
    } else {
        std::vector<double> qValues = mainNetwork.forward(state);
        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }
}

void DDQNPERADRAgent::addExperience(const std::vector<double>& state, int action, double reward,
                                   const std::vector<double>& nextState, bool done) {
    Experience exp(state, action, reward, nextState, done);
    replayBuffer.add(exp);
}

void DDQNPERADRAgent::train(size_t batchSize) {
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

std::pair<int, int> DDQNPERADRAgent::getParameters(int action) {
    if (action < 6) {
        return {sfActions[action], -1};
    } else if (action < 12) {
        int sfIdx = (action - 6) / 2;
        int tpIdx = (action - 6) % 2;
        return {sfActions[sfIdx + 3], tpActions[tpIdx]};
    } else {
        return {-1, tpActions[action - 12]};
    }
}

double DDQNPERADRAgent::calculateReward(double pdr, double energyConsumption, double prevPdr, double prevEnergy) {
    double pdrImprovement = pdr - prevPdr;
    double energyReduction = prevEnergy - energyConsumption;
    
    double reward = pdrWeight * pdrImprovement + energyWeight * energyReduction;
    
    if (pdr > 0.8 && energyConsumption < 0.2) {
        reward += 1.0;
    }
    
    return reward;
}

double DDQNPERADRAgent::calculateAdvancedReward(uint32_t nodeId, double pdr, double energyConsumption, 
                                               double snr, bool packetSuccess, int currentSF, double currentTP,
                                               double distance, uint32_t packetsSent) {
    double reward = 0.0;
    
    if (packetSuccess) {
        reward += 50.0;
    } else {
        reward -= 50.0;
    }
    
    double requiredSnr = getRequiredSNR(currentSF);
    double snrMargin = snr - requiredSnr;
    
    if (snrMargin < -10.0) {
        if (currentSF < 12) {
            reward -= 200.0;
            std::cout << "🔴 CRITICAL: SNR=" << snr << "dB requires SF12, using SF=" << currentSF << std::endl;
        } else {
            reward += 100.0;
        }
    } else if (snrMargin < -5.0) {
        if (currentSF < 10) {
            reward -= 150.0;
        } else {
            reward += 75.0;
        }
    } else if (snrMargin < 0.0) {
        if (currentSF < 8) {
            reward -= 100.0;
        } else {
            reward += 50.0;
        }
    } else if (snrMargin > 15.0) {
        if (currentSF > 7) {
            reward -= 30.0;
        } else {
            reward += 25.0;
        }
    }
    
    if (packetsSent >= 20) {
        if (pdr < 0.1) {
            if (currentSF < 12) {
                reward -= 300.0;
                std::cout << "⚠️ EMERGENCY: PDR=" << pdr*100 << "% - MUST USE SF12!" << std::endl;
            }
        } else if (pdr < 0.3) {
            if (currentSF < 10) {
                reward -= 200.0;
            }
        } else if (pdr < 0.6) {
            if (currentSF < 8) {
                reward -= 100.0;
            }
        }
    }
    
    if (currentTP > 14.0 && snrMargin > 5.0) {
        reward -= 50.0;
    } else if (currentTP > 11.0 && snrMargin > 10.0) {
        reward -= 25.0;
    }
    
    static std::map<uint32_t, int> lastSF;
    if (lastSF.find(nodeId) != lastSF.end()) {
        int previousSF = lastSF[nodeId];
        
        if (!packetSuccess && currentSF > previousSF) {
            reward += 75.0;
            std::cout << "✅ LEARNING: Increased SF " << previousSF << "→" << currentSF << " after failure" << std::endl;
        }
        
        if (pdr < 0.5 && currentSF < previousSF) {
            reward -= 100.0;
        }
    }
    lastSF[nodeId] = currentSF;
    
    if (energyConsumption > 0.5) {
        if (pdr < 0.7) {
            reward -= 75.0;
        }
    }
    
    return reward;
}

double DDQNPERADRAgent::getEpsilon() const {
    return epsilon;
}

} // namespace lorawan
} // namespace ns3
