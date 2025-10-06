/*
 * ddqn-per-adr-example.cc
 * 
 * Double Deep Q-Network with Prioritized Experience Replay (DDQN-PER) for LoRaWAN ADR
 * This implements the reinforcement learning approach described in the research paper
 * for optimizing Spreading Factor (SF) and Transmission Power (TP) to maximize PDR
 * while minimizing energy consumption.
 * 
 * Key Features:
 * - DDQN architecture with main and target networks
 * - Prioritized Experience Replay buffer
 * - Multi-objective reward function (PDR vs Energy Consumption)
 * - Dynamic SF and TP optimization
 * - Integration with enhanced pre-TX RSSI environment
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

NS_LOG_COMPONENT_DEFINE("DDQNPERADRExample");

// ============================================================================
// DDQN-PER IMPLEMENTATION
// ============================================================================

/**
 * Experience structure for Prioritized Experience Replay
 */
struct Experience {
    std::vector<double> state;      // [SNR, SF, TP]
    int action;                     // Action index
    double reward;                  // Reward received
    std::vector<double> nextState;  // Next state [SNR, SF, TP]
    bool done;                      // Episode termination
    double priority;                // Priority for sampling
    double tdError;                 // TD-error for priority updates
    
    Experience(const std::vector<double>& s, int a, double r, 
              const std::vector<double>& ns, bool d, double p = 1.0)
        : state(s), action(a), reward(r), nextState(ns), done(d), priority(p), tdError(0.0) {}
};

/**
 * Simple Neural Network implementation for Q-value approximation
 */
class SimpleQNetwork {
private:
    static const int INPUT_SIZE = 3;    // [SNR, SF, TP]
    static const int HIDDEN1_SIZE = 128;
    static const int HIDDEN2_SIZE = 64;
    static const int OUTPUT_SIZE = 12;   // 6 SF actions + 6 TP actions
    
    // Network weights
    std::vector<std::vector<double>> weights1;  // Input to Hidden1
    std::vector<std::vector<double>> weights2;  // Hidden1 to Hidden2
    std::vector<std::vector<double>> weights3;  // Hidden2 to Output
    std::vector<double> bias1, bias2, bias3;
    
    // For backpropagation
    std::vector<double> lastInput;
    std::vector<double> lastHidden1;
    std::vector<double> lastHidden2;
    std::vector<double> lastOutput;
    
    double learningRate;
    
    // Activation functions
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
        
        // Initialize weights
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
        
        // Hidden layer 1
        lastHidden1.resize(HIDDEN1_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            lastHidden1[i] = bias1[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                lastHidden1[i] += input[j] * weights1[j][i];
            }
            lastHidden1[i] = relu(lastHidden1[i]);
        }
        
        // Hidden layer 2
        lastHidden2.resize(HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            lastHidden2[i] = bias2[i];
            for (int j = 0; j < HIDDEN1_SIZE; j++) {
                lastHidden2[i] += lastHidden1[j] * weights2[j][i];
            }
            lastHidden2[i] = relu(lastHidden2[i]);
        }
        
        // Output layer
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
        // Simple gradient descent for demonstration
        // In practice, you'd use more sophisticated optimizers
        
        std::vector<double> outputError(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            outputError[i] = targetOutput[i] - lastOutput[i];
        }
        
        // Update output layer weights
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                weights3[i][j] += learningRate * outputError[j] * lastHidden2[i];
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            bias3[i] += learningRate * outputError[i];
        }
        
        // Backpropagate to hidden layers (simplified)
        std::vector<double> hidden2Error(HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            hidden2Error[i] = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                hidden2Error[i] += outputError[j] * weights3[i][j];
            }
            hidden2Error[i] *= reluDerivative(lastHidden2[i]);
        }
        
        // Update hidden layer 2 weights
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                weights2[i][j] += learningRate * hidden2Error[j] * lastHidden1[i];
            }
        }
        
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            bias2[i] += learningRate * hidden2Error[i];
        }
        
        // Backpropagate to hidden layer 1
        std::vector<double> hidden1Error(HIDDEN1_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            hidden1Error[i] = 0.0;
            for (int j = 0; j < HIDDEN2_SIZE; j++) {
                hidden1Error[i] += hidden2Error[j] * weights2[i][j];
            }
            hidden1Error[i] *= reluDerivative(lastHidden1[i]);
        }
        
        // Update hidden layer 1 weights
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
    size_t maxSize;
    size_t currentSize;
    size_t position;
    double alpha;  // Priority exponent
    double beta;   // Importance sampling exponent
    double epsilon; // Small constant to ensure non-zero priorities
    
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
        
        // Calculate probability distribution
        std::vector<double> probs(currentSize);
        double totalPriority = 0.0;
        
        for (size_t i = 0; i < currentSize; i++) {
            probs[i] = std::pow(priorities[i] + epsilon, alpha);
            totalPriority += probs[i];
        }
        
        for (size_t i = 0; i < currentSize; i++) {
            probs[i] /= totalPriority;
        }
        
        // Sample experiences
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
            
            // Calculate importance sampling weight
            double weight = std::pow(currentSize * probs[idx], -beta);
            weights.push_back(weight);
            maxWeight = std::max(maxWeight, weight);
        }
        
        // Normalize weights
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
    bool empty() const { return currentSize == 0; }
};

/**
 * DDQN-PER Agent for LoRaWAN ADR optimization
 */
class DDQNPERADRAgent {
private:
    SimpleQNetwork mainNetwork;
    SimpleQNetwork targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    
    double epsilon;                // Exploration rate
    double epsilonDecay;          // Epsilon decay rate
    double epsilonMin;            // Minimum epsilon
    double gamma;                 // Discount factor
    // double learningRate;          // Learning rate (commented out as unused)
    int targetUpdateFreq;         // Target network update frequency
    int updateCounter;            // Counter for target updates
    
    // Action space definition
    std::vector<int> sfActions = {7, 8, 9, 10, 11, 12};        // SF actions
    std::vector<int> tpActions = {2, 5, 8, 11, 14, 17};        // TP actions (dBm)
    
    // Reward function weights
    double pdrWeight;             // Weight for PDR in reward
    double energyWeight;          // Weight for energy consumption in reward
    
    std::mt19937 rng;
    
public:
    DDQNPERADRAgent(double lr = 0.001, double eps = 1.0, double epsDecay = 0.995, 
                   double epsMin = 0.01, double gam = 0.95, int targetFreq = 100,
                   double pdrW = 1.0, double energyW = 0.5)
        : mainNetwork(lr), targetNetwork(lr), replayBuffer(10000),
          epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin), gamma(gam),
          targetUpdateFreq(targetFreq), updateCounter(0),
          pdrWeight(pdrW), energyWeight(energyW), rng(std::random_device{}()) {
        
        // Initialize target network with same weights as main network
        targetNetwork.copyFrom(mainNetwork);
    }
    
    int selectAction(const std::vector<double>& state) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        if (dis(rng) < epsilon) {
            // Explore: random action
            std::uniform_int_distribution<int> actionDis(0, 11); // 12 total actions
            return actionDis(rng);
        } else {
            // Exploit: best action from Q-network
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
            
            // Current Q-values
            std::vector<double> currentQ = mainNetwork.forward(exp.state);
            
            // Target Q-values using Double DQN
            std::vector<double> nextQMain = mainNetwork.forward(exp.nextState);
            std::vector<double> nextQTarget = targetNetwork.forward(exp.nextState);
            
            // Find best action from main network
            int bestAction = std::distance(nextQMain.begin(), 
                                         std::max_element(nextQMain.begin(), nextQMain.end()));
            
            // Calculate target Q-value using target network
            double targetQ = exp.reward;
            if (!exp.done) {
                targetQ += gamma * nextQTarget[bestAction];
            }
            
            // Calculate TD-error
            double tdError = targetQ - currentQ[exp.action];
            tdErrors.push_back(tdError);
            
            // Update Q-value
            std::vector<double> targetOutput = currentQ;
            targetOutput[exp.action] = targetQ;
            
            // Apply importance sampling weight
            for (double& val : targetOutput) {
                val *= weights[i];
            }
            
            mainNetwork.updateWeights(targetOutput);
        }
        
        // Update priorities in replay buffer
        replayBuffer.updatePriorities(indices, tdErrors);
        
        // Update target network
        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0) {
            targetNetwork.copyFrom(mainNetwork);
        }
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }
    
    std::pair<int, int> getParameters(int action) {
        // Convert action index to SF and TP parameters
        if (action < 6) {
            // SF adjustment actions (0-5)
            return {sfActions[action], -1}; // -1 means no TP change
        } else {
            // TP adjustment actions (6-11)
            return {-1, tpActions[action - 6]}; // -1 means no SF change
        }
    }
    
    double calculateReward(double pdr, double energyConsumption, double prevPdr, double prevEnergy) {
        // Multi-objective reward function
        double pdrImprovement = pdr - prevPdr;
        double energyReduction = prevEnergy - energyConsumption; // Positive if energy reduced
        
        double reward = pdrWeight * pdrImprovement + energyWeight * energyReduction;
        
        // Bonus for achieving high PDR with low energy
        if (pdr > 0.8 && energyConsumption < 0.2) {
            reward += 1.0; // Bonus reward
        }
        
        return reward;
    }
    
    double getEpsilon() const { return epsilon; }
    void setEpsilon(double eps) { epsilon = eps; }
};

// ============================================================================
// SIMULATION ENVIRONMENT INTEGRATION
// ============================================================================

// Global tracking structures (reusing from advanced example)
std::map<uint32_t, bool> activeTransmitters;
std::map<uint32_t, double> nodeTxPowers;
std::map<uint32_t, Vector> nodePositions;
std::map<uint32_t, Time> transmissionEndTimes;
std::map<uint32_t, int> currentSF;           // Current SF for each device
std::map<uint32_t, double> currentTP;        // Current TP for each device
std::map<uint32_t, double> lastSNR;          // Last measured SNR for each device

// DDQN-PER agents (one per device)
std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>> adrAgents;

// Performance tracking
struct DeviceMetrics {
    uint32_t packetsSent = 0;
    uint32_t packetsReceived = 0;
    double totalEnergyConsumed = 0.0;
    double lastPDR = 0.0;
    double lastEnergyPerPacket = 0.0;
    std::vector<double> snrHistory;
    std::vector<double> pdrHistory;
    std::vector<double> energyHistory;
};

std::map<uint32_t, DeviceMetrics> deviceMetrics;

// Global simulation parameters
Ptr<LoraChannel> globalChannel;
std::vector<Ptr<LoraNetDevice>> endDevicesNetDevices;
double simulationStartTime = 0.0;

/**
 * Enhanced energy calculation based on LoRaWAN parameters
 */
double CalculateEnergyConsumption(int sf, double txPowerDbm, uint32_t payloadBytes) {
    // LoRaWAN energy model based on SX1276 specifications
    
    // Base current consumption (mA) - remove unused variable warning
    // double baseCurrent = 10.8; // Sleep mode current
    double txCurrent = 120.0;  // TX mode base current
    
    // TX power dependent current (additional mA per dBm above 2dBm)
    double powerCurrent = (txPowerDbm - 2.0) * 3.2; // Approximate linear relationship
    
    // Calculate transmission time
    LoraTxParameters txParams;
    txParams.sf = sf;
    txParams.headerDisabled = false;
    txParams.codingRate = 1; // CR 4/5
    txParams.bandwidthHz = 125000;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (sf > 10);
    
    Ptr<Packet> dummyPacket = Create<Packet>(payloadBytes);
    Time txTime = LoraPhy::GetOnAirTime(dummyPacket, txParams);
    
    // Total current during transmission
    double totalCurrent = txCurrent + powerCurrent;
    
    // Energy consumption in mJ (3.3V supply)
    double energyMj = (totalCurrent * 3.3 * txTime.GetSeconds()) / 1000.0;
    
    return energyMj;
}

/**
 * Update device state and trigger DDQN-PER learning
 */
void UpdateDeviceADR(uint32_t nodeId, double snr, bool packetSuccess) {
    if (adrAgents.find(nodeId) == adrAgents.end()) {
        return; // Agent not initialized
    }
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    
    // Update metrics
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    
    // Calculate current PDR and energy consumption
    double currentPDR = (double)metrics.packetsReceived / metrics.packetsSent;
    double energyThisPacket = CalculateEnergyConsumption(currentSF[nodeId], currentTP[nodeId], 20);
    metrics.totalEnergyConsumed += energyThisPacket;
    double avgEnergyPerPacket = metrics.totalEnergyConsumed / metrics.packetsSent;
    
    // Store SNR
    lastSNR[nodeId] = snr;
    metrics.snrHistory.push_back(snr);
    
    // Create state vector [SNR, SF, TP] (normalized)
    std::vector<double> currentState = {
        (snr + 20.0) / 40.0,  // SNR normalized to [0,1] (assuming -20 to +20 dB range)
        (currentSF[nodeId] - 7.0) / 5.0,  // SF normalized to [0,1] (SF 7-12)
        (currentTP[nodeId] - 2.0) / 15.0  // TP normalized to [0,1] (2-17 dBm)
    };
    
    // Calculate reward
    double reward = adrAgents[nodeId]->calculateReward(currentPDR, avgEnergyPerPacket, 
                                                      metrics.lastPDR, metrics.lastEnergyPerPacket);
    
    // Add experience to replay buffer (if we have previous state)
    static std::map<uint32_t, std::vector<double>> previousStates;
    static std::map<uint32_t, int> previousActions;
    
    if (previousStates.find(nodeId) != previousStates.end()) {
        adrAgents[nodeId]->addExperience(previousStates[nodeId], previousActions[nodeId], 
                                        reward, currentState, false);
        
        // Train the agent
        adrAgents[nodeId]->train();
    }
    
    // Select next action
    int action = adrAgents[nodeId]->selectAction(currentState);
    auto [newSF, newTP] = adrAgents[nodeId]->getParameters(action);
    
    // Apply parameter changes (update our tracking variables)
    if (newSF != -1) {
        currentSF[nodeId] = newSF;
        // Note: In a full implementation, this would interface with the MAC layer
        // For now, we track parameters and use them in energy/transmission calculations
        NS_LOG_DEBUG("Updated SF for node " << nodeId << " to " << newSF);
    }
    
    if (newTP != -1) {
        currentTP[nodeId] = newTP;
        nodeTxPowers[nodeId] = newTP;
        // Note: In a full implementation, this would interface with the MAC layer
        // For now, we track parameters and use them in energy/transmission calculations  
        NS_LOG_DEBUG("Updated TP for node " << nodeId << " to " << newTP << " dBm");
    }
    
    // Store current state and action for next iteration
    previousStates[nodeId] = currentState;
    previousActions[nodeId] = action;
    
    // Update last metrics
    metrics.lastPDR = currentPDR;
    metrics.lastEnergyPerPacket = avgEnergyPerPacket;
    metrics.pdrHistory.push_back(currentPDR);
    metrics.energyHistory.push_back(avgEnergyPerPacket);
    
    NS_LOG_INFO("DDQN-PER ADR Update - Node " << nodeId 
               << ", SNR: " << snr << " dB"
               << ", Action: " << action
               << ", New SF: " << currentSF[nodeId]
               << ", New TP: " << currentTP[nodeId] << " dBm"
               << ", PDR: " << currentPDR
               << ", Energy: " << avgEnergyPerPacket << " mJ"
               << ", Reward: " << reward
               << ", Epsilon: " << adrAgents[nodeId]->getEpsilon());
}

/**
 * Enhanced RSSI sampling (reusing from advanced example)
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    // Simplified version - reuse the logic from advanced example
    Vector txPos = nodePositions[txNodeId];
    
    Ptr<UniformRandomVariable> noiseVar = CreateObject<UniformRandomVariable>();
    noiseVar->SetAttribute("Min", DoubleValue(-115.0));
    noiseVar->SetAttribute("Max", DoubleValue(-105.0));
    double ambientNoise = noiseVar->GetValue();
    
    double totalPowerLinear = DbmToW(ambientNoise);
    
    // Add interference from active transmitters
    for (const auto& pair : activeTransmitters) {
        uint32_t otherNodeId = pair.first;
        bool isTransmitting = pair.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) continue;
        if (currentTime >= transmissionEndTimes[otherNodeId]) continue;
        
        Vector interfererPos = nodePositions[otherNodeId];
        double interfererTxPower = nodeTxPowers[otherNodeId];
        
        Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel>();
        txMobility->SetPosition(txPos);
        
        Ptr<ConstantPositionMobilityModel> interfererMobility = CreateObject<ConstantPositionMobilityModel>();
        interfererMobility->SetPosition(interfererPos);
        
        double rxPowerDbm = globalChannel->GetRxPower(interfererTxPower, interfererMobility, txMobility);
        totalPowerLinear += DbmToW(rxPowerDbm);
    }
    
    double preTxRssiDbm = WToDbm(totalPowerLinear);
    
    // Add measurement noise
    Ptr<NormalRandomVariable> rssiNoise = CreateObject<NormalRandomVariable>();
    rssiNoise->SetAttribute("Mean", DoubleValue(0.0));
    rssiNoise->SetAttribute("Variance", DoubleValue(1.0));
    
    double noisyRssiDbm = preTxRssiDbm + rssiNoise->GetValue();
    
    RssiSample sample;
    sample.preTxRssiDbm = noisyRssiDbm;
    sample.ambientNoiseDbm = ambientNoise;
    return sample;
}

// Packet tracking for performance evaluation
struct PacketRecord {
    double timestamp;
    uint32_t nodeId;
    uint32_t fcnt;
    double txPowerDbm;
    uint8_t sf;
    Vector txPos;
    double preTxRssiDbm;
    double ambientNoiseDbm;
    bool packetReceived;
    double gatewayRssiDbm;
    double gatewaySnrDb;
    double energyConsumed;
    double currentPDR;
    double currentEnergyPerPacket;
    int adrAction;
    double adrReward;
};

std::vector<PacketRecord> completedPackets;
std::map<uint32_t, uint32_t> deviceFrameCounters;

/**
 * Callback triggered when a node starts transmitting
 */
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    // Sample pre-TX RSSI
    RssiSample rssiSample = SamplePreTxRssi(nodeId, currentTime);
    
    // Mark node as transmitting
    activeTransmitters[nodeId] = true;
    
    // Calculate transmission duration
    LoraTxParameters txParams;
    txParams.sf = currentSF[nodeId];
    txParams.headerDisabled = false;
    txParams.codingRate = 1;
    txParams.bandwidthHz = 125000;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (currentSF[nodeId] > 10);
    
    Time txDuration = LoraPhy::GetOnAirTime(packet->Copy(), txParams);
    transmissionEndTimes[nodeId] = currentTime + txDuration;
    
    // Calculate energy consumption for this packet
    double energyThisPacket = CalculateEnergyConsumption(currentSF[nodeId], currentTP[nodeId], packet->GetSize());
    
    // Create packet record
    PacketRecord record;
    record.timestamp = currentTime.GetSeconds();
    record.nodeId = nodeId;
    record.fcnt = ++deviceFrameCounters[nodeId];
    record.txPowerDbm = currentTP[nodeId];
    record.sf = currentSF[nodeId];
    record.txPos = nodePositions[nodeId];
    record.preTxRssiDbm = rssiSample.preTxRssiDbm;
    record.ambientNoiseDbm = rssiSample.ambientNoiseDbm;
    record.packetReceived = false;
    record.gatewayRssiDbm = -999.0;
    record.gatewaySnrDb = -999.0;
    record.energyConsumed = energyThisPacket;
    record.currentPDR = deviceMetrics[nodeId].packetsReceived / (double)(deviceMetrics[nodeId].packetsSent + 1);
    record.currentEnergyPerPacket = (deviceMetrics[nodeId].totalEnergyConsumed + energyThisPacket) / (deviceMetrics[nodeId].packetsSent + 1);
    record.adrAction = -1; // Will be updated after ADR decision
    record.adrReward = 0.0; // Will be updated after reward calculation
    
    completedPackets.push_back(record);
    
    NS_LOG_INFO("TX START - Node " << nodeId 
               << ", SF: " << static_cast<int>(currentSF[nodeId])
               << ", TP: " << currentTP[nodeId] << " dBm"
               << ", Energy: " << energyThisPacket << " mJ"
               << ", Pre-TX RSSI: " << rssiSample.preTxRssiDbm << " dBm");
    
    // Schedule transmission end
    Simulator::Schedule(txDuration, [nodeId]() {
        activeTransmitters[nodeId] = false;
    });
}

/**
 * Callback triggered when a packet is received at gateway
 */
void OnPacketReceived(Ptr<const Packet> packet) {
    Time rxTime = Simulator::Now();
    
    // Find corresponding packet record and update
    for (auto& record : completedPackets) {
        if (!record.packetReceived && 
            rxTime.GetSeconds() >= record.timestamp &&
            std::abs(record.timestamp - rxTime.GetSeconds()) < 3.0) {
            
            record.packetReceived = true;
            
            // Calculate gateway RSSI and SNR
            Vector gatewayPos(0, 0, 15);
            double distance = CalculateDistance(record.txPos, gatewayPos);
            double pathLossDb = 20 * std::log10(distance) + 20 * std::log10(868e6) - 147.55;
            record.gatewayRssiDbm = record.txPowerDbm - pathLossDb;
            record.gatewaySnrDb = record.gatewayRssiDbm - record.ambientNoiseDbm;
            
            // Trigger DDQN-PER ADR update
            UpdateDeviceADR(record.nodeId, record.gatewaySnrDb, true);
            
            NS_LOG_INFO("PACKET RECEIVED - Node " << record.nodeId 
                       << ", Gateway RSSI: " << record.gatewayRssiDbm << " dBm"
                       << ", SNR: " << record.gatewaySnrDb << " dB");
            break;
        }
    }
}

/**
 * Callback for packet loss (no reception)
 */
void OnPacketLost(Ptr<const Packet> packet, uint32_t systemId) {
    // Find the packet record and trigger ADR update with failure
    Time currentTime = Simulator::Now();
    
    for (auto& record : completedPackets) {
        if (record.nodeId == systemId && !record.packetReceived &&
            std::abs(record.timestamp - currentTime.GetSeconds()) < 0.1) {
            
            // Trigger DDQN-PER ADR update with packet loss
            UpdateDeviceADR(record.nodeId, lastSNR[record.nodeId], false);
            
            NS_LOG_INFO("PACKET LOST - Node " << record.nodeId 
                       << ", Triggering ADR adaptation");
            break;
        }
    }
}

/**
 * Write comprehensive CSV output for analysis
 */
void WriteCSVOutput(const std::string& filename) {
    std::string folderName = "lorawan_datasets";
    system(("mkdir -p " + folderName).c_str());
    
    std::string baseFilename = filename;
    size_t lastDot = baseFilename.find_last_of('.');
    if (lastDot != std::string::npos) {
        baseFilename = baseFilename.substr(0, lastDot);
    }
    
    // Group packets by device
    std::map<uint32_t, std::vector<PacketRecord>> devicePackets;
    for (const auto& record : completedPackets) {
        devicePackets[record.nodeId].push_back(record);
    }
    
    // Write separate CSV file for each device
    Vector gatewayPos(0, 0, 15);
    for (const auto& devicePair : devicePackets) {
        uint32_t deviceId = devicePair.first;
        const std::vector<PacketRecord>& packets = devicePair.second;
        
        std::string deviceFilename = folderName + "/" + baseFilename + "_device_" + std::to_string(deviceId) + "_dataset.csv";
        std::ofstream deviceCsvFile(deviceFilename);
        
        // Enhanced header for DDQN-PER analysis
        deviceCsvFile << "timestamp,devEUI,fcnt,tx_power_dBm,sf,bw,cr,tx_x,tx_y,tx_z,"
                     << "pre_tx_rssi_dBm,ambient_noise_dBm,expected_rx_rssi_dBm,snr_db,"
                     << "gateway_id,packet_received,energy_consumed_mJ,current_pdr,"
                     << "current_energy_per_packet_mJ,adr_action,adr_reward,distance_m\n";
        
        for (const auto& record : packets) {
            double distance = CalculateDistance(record.txPos, gatewayPos);
            
            deviceCsvFile << std::fixed << std::setprecision(6)
                         << record.timestamp << ","
                         << "DEV" << record.nodeId << ","
                         << record.fcnt << ","
                         << record.txPowerDbm << ","
                         << static_cast<int>(record.sf) << ","
                         << 125000 << ","
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
                         << record.currentPDR << ","
                         << record.currentEnergyPerPacket << ","
                         << record.adrAction << ","
                         << record.adrReward << ","
                         << distance << "\n";
        }
        
        deviceCsvFile.close();
        std::cout << "DDQN-PER Device " << deviceId << " dataset saved to: " << deviceFilename 
                  << " (" << packets.size() << " packets)" << std::endl;
    }
    
    // Calculate and display final statistics
    int totalPackets = completedPackets.size();
    int receivedPackets = 0;
    double totalEnergy = 0.0;
    
    for (const auto& record : completedPackets) {
        if (record.packetReceived) {
            receivedPackets++;
        }
        totalEnergy += record.energyConsumed;
    }
    
    double overallPDR = (double)receivedPackets / totalPackets * 100.0;
    double avgEnergyPerPacket = totalEnergy / totalPackets;
    
    std::cout << "\n=== DDQN-PER LoRaWAN ADR RESULTS ===" << std::endl;
    std::cout << "Total packets transmitted: " << totalPackets << std::endl;
    std::cout << "Successfully received: " << receivedPackets << " (" << overallPDR << "%)" << std::endl;
    std::cout << "Average energy per packet: " << avgEnergyPerPacket << " mJ" << std::endl;
    std::cout << "Total energy consumed: " << totalEnergy << " mJ" << std::endl;
    
    // Per-device statistics
    std::cout << "\n=== Per-Device Performance ===" << std::endl;
    for (const auto& pair : deviceMetrics) {
        uint32_t deviceId = pair.first;
        const DeviceMetrics& metrics = pair.second;
        
        double devicePDR = (double)metrics.packetsReceived / metrics.packetsSent * 100.0;
        double deviceAvgEnergy = metrics.totalEnergyConsumed / metrics.packetsSent;
        
        std::cout << "Device " << deviceId 
                  << ": PDR=" << devicePDR << "%"
                  << ", Avg Energy=" << deviceAvgEnergy << " mJ"
                  << ", Final SF=" << currentSF[deviceId]
                  << ", Final TP=" << currentTP[deviceId] << " dBm"
                  << ", Packets=" << metrics.packetsSent 
                  << "/" << metrics.packetsReceived << std::endl;
    }
    
    std::cout << "Per-device DDQN-PER datasets saved to: lorawan_datasets/ folder" << std::endl;
}

/**
 * Run DDQN-PER enhanced LoRaWAN simulation
 */
void RunDDQNPERSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                         double radius, const std::string& csvFileName) {
    
    // Initialize global state
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    endDevicesNetDevices.clear();
    adrAgents.clear();
    deviceMetrics.clear();
    currentSF.clear();
    currentTP.clear();
    lastSNR.clear();
    
    simulationStartTime = Simulator::Now().GetSeconds();
    
    // Create nodes
    NodeContainer endDevices;
    endDevices.Create(nDevices);
    
    NodeContainer gateways;
    gateways.Create(1);
    
    NodeContainer allNodes = NodeContainer(endDevices, gateways);
    
    // Set up mobility
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    // Gateway at center
    allocator->Add(Vector(0, 0, 15));
    nodePositions[gateways.Get(0)->GetId()] = Vector(0, 0, 15);
    
    // Random device positions
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-radius));
    xRand->SetAttribute("Max", DoubleValue(radius));
    
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-radius));
    yRand->SetAttribute("Max", DoubleValue(radius));
    
    for (uint32_t i = 0; i < nDevices; i++) {
        uint32_t nodeId = endDevices.Get(i)->GetId();
        Vector pos(xRand->GetValue(), yRand->GetValue(), 1.5);
        allocator->Add(pos);
        nodePositions[nodeId] = pos;
        
        // Initialize device parameters
        currentSF[nodeId] = 7;        // Start with SF7
        currentTP[nodeId] = 14.0;     // Start with 14 dBm
        nodeTxPowers[nodeId] = 14.0;
        lastSNR[nodeId] = 0.0;
        
        // Initialize DDQN-PER agent for each device
        adrAgents[nodeId] = std::make_unique<DDQNPERADRAgent>();
        
        // Initialize device metrics
        deviceMetrics[nodeId] = DeviceMetrics();
    }
    
    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);
    
    // Create LoRa channel with log-distance propagation
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetPathLossExponent(3.76);
    loss->SetReference(1, 7.7);
    
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    globalChannel = CreateObject<LoraChannel>(loss, delay);
    
    // Set up LoRaWAN stack
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(globalChannel);
    
    LorawanMacHelper macHelper = LorawanMacHelper();
    LoraHelper helper = LoraHelper();
    
    // Install on end devices
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, endDevices);
    
    // Store end device net devices and set initial parameters
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        Ptr<LoraNetDevice> loraNetDevice = DynamicCast<LoraNetDevice>(endDevicesNetDevs.Get(i));
        endDevicesNetDevices.push_back(loraNetDevice);
        
        uint32_t nodeId = endDevices.Get(i)->GetId();
        
        // Note: In a full implementation, we would set initial SF and TP on the MAC/PHY
        // For this simulation, we track parameters in our global maps and use them
        // in energy calculations and transmission modeling
        NS_LOG_INFO("Device " << nodeId << " initialized with SF=" << currentSF[nodeId] 
                    << ", TP=" << currentTP[nodeId] << " dBm");
    }
    
    // Install on gateway
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    NetDeviceContainer gwNetDevices = helper.Install(phyHelper, macHelper, gateways);
    
    // Set up applications
    ApplicationContainer appContainer;
    
    for (uint32_t i = 0; i < nDevices; i++) {
        // Create randomized transmission intervals
        Ptr<UniformRandomVariable> intervalRand = CreateObject<UniformRandomVariable>();
        intervalRand->SetAttribute("Min", DoubleValue(appPeriodSeconds * 0.8));
        intervalRand->SetAttribute("Max", DoubleValue(appPeriodSeconds * 1.2));
        
        PeriodicSenderHelper appHelper = PeriodicSenderHelper();
        appHelper.SetPacketSize(20); // 20 byte payload
        appHelper.SetPeriod(Seconds(intervalRand->GetValue()));
        
        ApplicationContainer app = appHelper.Install(endDevices.Get(i));
        appContainer.Add(app);
    }
    
    // Stagger application start times
    for (uint32_t i = 0; i < nDevices; i++) {
        Ptr<UniformRandomVariable> startRand = CreateObject<UniformRandomVariable>();
        startRand->SetAttribute("Min", DoubleValue(0.0));
        startRand->SetAttribute("Max", DoubleValue(appPeriodSeconds * 0.5));
        
        appContainer.Get(i)->SetStartTime(Seconds(startRand->GetValue()));
        appContainer.Get(i)->SetStopTime(Seconds(simulationTime));
    }
    
    // Connect callbacks
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Phy/$ns3::EndDeviceLoraPhy/StartSending",
                                 MakeCallback(&OnTransmissionStart));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::GatewayLorawanMac/ReceivedPacket",
                                 MakeCallback(&OnPacketReceived));
    
    std::cout << "\n=== DDQN-PER Enhanced LoRaWAN ADR Simulation ===" << std::endl;
    std::cout << "Devices: " << nDevices << std::endl;
    std::cout << "Simulation time: " << simulationTime << " seconds" << std::endl;
    std::cout << "Area radius: " << radius << " meters" << std::endl;
    std::cout << "Using Double Deep Q-Network with Prioritized Experience Replay for ADR" << std::endl;
    std::cout << "Multi-objective optimization: PDR maximization + Energy minimization" << std::endl;
    std::cout << "Output file: " << csvFileName << std::endl;
    
    // Schedule CSV output
    Simulator::Schedule(Seconds(simulationTime), [csvFileName]() {
        WriteCSVOutput(csvFileName);
    });
    
    // Run simulation
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "DDQN-PER ADR simulation completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    // Simulation parameters
    uint32_t nDevices = 10;
    double simulationTime = 600.0; // 10 minutes for initial testing
    double appPeriodSeconds = 60.0; // 1 minute intervals
    double radius = 1000.0; // meters
    std::string csvFileName = "ddqn_per_adr_dataset.csv";
    
    // Parse command line
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.Parse(argc, argv);
    
    // Enable logging
    LogComponentEnable("DDQNPERADRExample", LOG_LEVEL_INFO);
    
    std::cout << "\n=== DDQN-PER LoRaWAN ADR Implementation ===" << std::endl;
    std::cout << "Research-based Reinforcement Learning ADR using:" << std::endl;
    std::cout << "- Double Deep Q-Network (DDQN) for stable Q-learning" << std::endl;
    std::cout << "- Prioritized Experience Replay (PER) for efficient learning" << std::endl;
    std::cout << "- Multi-objective reward: PDR maximization + Energy minimization" << std::endl;
    std::cout << "- State space: [SNR, SF, TP]" << std::endl;
    std::cout << "- Action space: SF adjustment (7-12) + TP adjustment (2-17 dBm)" << std::endl;
    
    // Run the simulation
    RunDDQNPERSimulation(nDevices, simulationTime, appPeriodSeconds, radius, csvFileName);
    
    return 0;
}