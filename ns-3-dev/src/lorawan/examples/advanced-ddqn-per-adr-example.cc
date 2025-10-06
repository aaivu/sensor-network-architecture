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
    static const int INPUT_SIZE = 3;
    static const int HIDDEN1_SIZE = 128;
    static const int HIDDEN2_SIZE = 64;
    static const int OUTPUT_SIZE = 12;
    
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
    std::vector<int> tpActions = {2, 5, 8, 11, 14, 17};
    double pdrWeight, energyWeight;
    std::mt19937 rng;
    
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
        if (action < 6) {
            return {sfActions[action], -1};
        } else {
            return {-1, tpActions[action - 6]};
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
    
    double getEpsilon() const { return epsilon; }
};

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

// Forward declarations
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
void UpdateClassicalADR(uint32_t nodeId, double snr, bool packetSuccess);
void WriteCSVOutput(const std::string& filename);

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
 * Calculate energy consumption for a packet
 */
double CalculateEnergyConsumption(int sf, double txPowerDbm, uint32_t payloadBytes) {
    double txCurrent = 120.0;
    double powerCurrent = (txPowerDbm - 2.0) * 3.2;
    
    LoraTxParameters txParams;
    txParams.sf = sf;
    txParams.headerDisabled = false;
    txParams.codingRate = 1;
    txParams.bandwidthHz = 125000;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (sf > 10);
    
    Ptr<Packet> dummyPacket = Create<Packet>(payloadBytes);
    Time txTime = LoraPhy::GetOnAirTime(dummyPacket, txParams);
    
    double totalCurrent = txCurrent + powerCurrent;
    double energyMj = (totalCurrent * 3.3 * txTime.GetSeconds()) / 1000.0;
    
    return energyMj;
}

/**
 * Callback triggered when a node starts transmitting
 */
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    RssiSample rssiSample = SamplePreTxRssi(nodeId, currentTime);
    activeTransmitters[nodeId] = true;
    
    // **GET ACTUAL PARAMETERS FROM DEVICE**
    uint8_t actualSF = 7;
    double actualTxPower = nodeTxPowers[nodeId];
    
    if (nodeId < endDevicesNetDevices.size()) {
        // Read SF from PHY layer
        Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
            endDevicesNetDevices[nodeId]->GetPhy());
        if (edPhy) {
            actualSF = edPhy->GetSpreadingFactor();
            currentSF[nodeId] = actualSF;
        }
        
        // Read TX power from MAC layer
        Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
            endDevicesNetDevices[nodeId]->GetMac());
        if (edMac) {
            actualTxPower = edMac->GetTransmissionPowerDbm();
            currentTP[nodeId] = actualTxPower;
            nodeTxPowers[nodeId] = actualTxPower;
        }
        
        std::cout << "📡 Device " << nodeId << " transmitting with SF=" 
                  << (int)actualSF << ", TP=" << actualTxPower << " dBm" << std::endl;
    }
    
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
    transmissionEndTimes[nodeId] = currentTime + txDuration;
    
    // Create packet record with ACTUAL parameters
    PacketRecord record;
    record.timestamp = currentTime.GetSeconds();
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
    
    // Schedule transmission end
    Simulator::Schedule(txDuration, [nodeId]() {
        activeTransmitters[nodeId] = false;
    });
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
            
            if (currentADRMethod == ADRMethod::DDQN) {
                UpdateDDQNADR(it->nodeId, it->gatewaySnrDb, true);
            } else if (currentADRMethod == ADRMethod::ON) {
                UpdateClassicalADR(it->nodeId, it->gatewaySnrDb, true);
            }
            
            NS_LOG_INFO("RX SUCCESS - Node " << it->nodeId << " at " << rxTime.GetSeconds() 
                       << "s, SNR: " << it->gatewaySnrDb << " dB");
            return;
        }
    }
}

/**
 * Update function for DDQN-PER ADR
 */
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess) {
    if (ddqnAgents.find(nodeId) == ddqnAgents.end()) return;
    
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    
    double currentPDR = (double)metrics.packetsReceived / metrics.packetsSent;
    uint8_t currentSFVal = currentSF[nodeId];
    double currentTPVal = currentTP[nodeId];
    
    double energyThisPacket = CalculateEnergyConsumption(currentSFVal, currentTPVal, 20);
    metrics.totalEnergyConsumed += energyThisPacket;
    double avgEnergyPerPacket = metrics.totalEnergyConsumed / metrics.packetsSent;
    
    // **IMPROVED STATE NORMALIZATION**
    std::vector<double> currentState = {
        std::max(0.0, std::min(1.0, (snr + 20.0) / 40.0)),
        (currentSFVal - 7.0) / 5.0,
        (currentTPVal - 2.0) / 15.0
    };
    
    // **IMPROVED REWARD FUNCTION**
    double reward = 0.0;
    if (metrics.packetsSent >= 2) {
        double pdrImprovement = currentPDR - metrics.lastPDR;
        double energyReduction = metrics.lastEnergyPerPacket - avgEnergyPerPacket;
        
        // Reward components
        reward = 10.0 * pdrImprovement +     // PDR improvement (high weight)
                 1.0 * energyReduction +     // Energy efficiency
                 (packetSuccess ? 2.0 : -5.0); // Immediate success/failure
        
        // **DEBUG OUTPUT**
        std::cout << "DDQN Node " << nodeId 
                  << ": PDR=" << currentPDR*100 << "%, Energy=" << avgEnergyPerPacket 
                  << " mJ, Reward=" << reward 
                  << ", SF=" << (int)currentSFVal << ", TP=" << currentTPVal << std::endl;
    }
    
    static std::map<uint32_t, std::vector<double>> previousStates;
    static std::map<uint32_t, int> previousActions;
    
    if (previousStates.find(nodeId) != previousStates.end() && metrics.packetsSent >= 2) {
        ddqnAgents[nodeId]->addExperience(previousStates[nodeId], previousActions[nodeId], 
                                        reward, currentState, false);
        ddqnAgents[nodeId]->train();
    }
    
    int action = ddqnAgents[nodeId]->selectAction(currentState);
    auto [newSF, newTP] = ddqnAgents[nodeId]->getParameters(action);
    
    // **FIXED: APPLY PARAMETERS THROUGH MAC LAYER**
        if (newSF != -1) {
        currentSF[nodeId] = newSF;
        
        // Schedule parameter update for next transmission
        Simulator::Schedule(Seconds(0.1), [nodeId, newSF]() {
            if (nodeId < endDevicesNetDevices.size()) {
                Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
                    endDevicesNetDevices[nodeId]->GetPhy());
                if (edPhy) {
                    edPhy->SetSpreadingFactor(newSF);
                    
                    // **ALSO UPDATE MAC LAYER DATA RATE**
                    Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                        endDevicesNetDevices[nodeId]->GetMac());
                    if (edMac) {
                        // Map SF to data rate index (SF12->DR0, SF11->DR1, ..., SF7->DR5)
                        uint8_t dataRate = 12 - newSF;
                        edMac->SetDataRate(dataRate);
                    }
                    
                    std::cout << "✅ Applied SF=" << (int)newSF 
                              << " + DR=" << (int)(12-newSF) 
                              << " to device " << nodeId << std::endl;
                }
            }
        });
    }
    
    if (newTP != -1) {
        currentTP[nodeId] = newTP;
        nodeTxPowers[nodeId] = newTP;  // Update global tracking
        
        // **APPLY TX POWER THROUGH MAC LAYER (CORRECT METHOD)**
        if (nodeId < endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                // Use the correct MAC layer method
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
 * Update function for Classical ADR
 */
void UpdateClassicalADR(uint32_t nodeId, double snr, bool packetSuccess) {
    DeviceMetrics& metrics = deviceMetrics[nodeId];
    
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    
    double currentPDR = (double)metrics.packetsReceived / metrics.packetsSent;
    
    // Classical ADR algorithm (simplified but realistic)
    // Only perform ADR if we have enough data points
    if (metrics.packetsSent >= 20) {
        // Classical ADR thresholds (based on LoRaWAN 1.0.4 specification)
        const double SNR_MARGIN = 2.5; // dB
        const double REQUIRED_SNR[6] = {-20, -17.5, -15, -12.5, -10, -7.5}; // SF7-12
        
        int currentSFIdx = currentSF[nodeId] - 7;
        double requiredSnr = REQUIRED_SNR[currentSFIdx];
        double snrMargin = snr - requiredSnr;
        
        // SF optimization
        if (snrMargin > SNR_MARGIN + 3.0 && currentSF[nodeId] > 7) {
            // Good link quality - decrease SF for higher data rate
            currentSF[nodeId]--;
        } else if (snrMargin < SNR_MARGIN && currentSF[nodeId] < 12) {
            // Poor link quality - increase SF for better sensitivity
            currentSF[nodeId]++;
        }
        
        // TX Power optimization
        if (snrMargin > SNR_MARGIN + 5.0 && currentTP[nodeId] > 2.0) {
            // Excellent link quality - reduce power for energy saving
            currentTP[nodeId] = std::max(2.0, currentTP[nodeId] - 3.0);
        } else if (snrMargin < SNR_MARGIN - 2.0 && currentTP[nodeId] < 17.0) {
            // Very poor link quality - increase power
            currentTP[nodeId] = std::min(17.0, currentTP[nodeId] + 3.0);
        }
    }
    
    NS_LOG_INFO("Classical ADR Update - Node " << nodeId 
               << ", SNR: " << snr << " dB"
               << ", SF: " << (int)currentSF[nodeId]
               << ", TP: " << currentTP[nodeId] << " dBm"
               << ", PDR: " << currentPDR);
}

/**
 * Run a single simulation
 */
void RunSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                  double radius, const std::string& csvFileName, ADRMethod adrMethod, uint32_t nWifiInterferers);

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
        std::cout << "\nRunning simulation for NO ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "no_adr_" + csvFileName, ADRMethod::OFF, nWifiInterferers);
        
        std::cout << "\nRunning simulation for CLASSICAL ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "adr_" + csvFileName, ADRMethod::ON, nWifiInterferers);
        
        std::cout << "\nRunning simulation for DDQN-PER ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "ddqn_adr_" + csvFileName, ADRMethod::DDQN, nWifiInterferers);
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
    
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    endDevicesNetDevices.clear();
    ddqnAgents.clear();
    deviceMetrics.clear();
    nodeAntennaGains.clear();
    nodeNoiseFigures.clear();
    nodeHardwareVariance.clear();
    urbanObstacles.clear();
    wifiInterferers = NodeContainer();
    
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
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, endDevices);
    
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        endDevicesNetDevices.push_back(endDevicesNetDevs.Get(i)->GetObject<LoraNetDevice>());
        // Initialize tracking variables with default values
        uint32_t nodeId = endDevices.Get(i)->GetId();
        currentSF[nodeId] = 7;  // Default SF
        currentTP[nodeId] = nodeTxPowers[nodeId];  // Use initial power
    }
    
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    helper.Install(phyHelper, macHelper, gateways);
    
    if (adrMethod == ADRMethod::ON) {
        // Enable built-in ADR (if available in this ns-3 version)
        // For now, we'll handle classical ADR manually in callbacks
        std::cout << "Classical ADR enabled (manual implementation)" << std::endl;
    } else {
        // Disable ADR
        std::cout << "ADR disabled" << std::endl;
    }
    
    if (adrMethod == ADRMethod::DDQN) {
        for (uint32_t i = 0; i < nDevices; ++i) {
            uint32_t nodeId = endDevices.Get(i)->GetId();
            ddqnAgents[nodeId] = std::make_unique<DDQNPERADRAgent>();
        }
    }
    
    for (uint32_t i = 0; i < nDevices; i++) {
        PeriodicSenderHelper appHelper = PeriodicSenderHelper();
        appHelper.SetPeriod(Seconds(appPeriodSeconds));
        
        ApplicationContainer app = appHelper.Install(endDevices.Get(i));
        
        Ptr<UniformRandomVariable> randomStart = CreateObject<UniformRandomVariable>();
        randomStart->SetAttribute("Min", DoubleValue(0.0));
        randomStart->SetAttribute("Max", DoubleValue(appPeriodSeconds));
        randomStart->SetStream(5000 + i);  // Fixed stream for consistent app start times
        app.Start(Seconds(randomStart->GetValue()));
    }
    
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
    Simulator::Destroy();
    
    std::cout << "Simulation completed!" << std::endl;
}
