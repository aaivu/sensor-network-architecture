/*
 * rl-adr-simple-example.cc
 * 
 * Simplified Reinforcement Learning ADR implementation for LoRaWAN
 * Combines Pre-TX RSSI sampling with Q-Learning based parameter optimization
 * 
 * Based on DDQN-PER methodology but simplified for ns-3 compatibility
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

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("RLAdrSimpleExample");

// ===========================
// SIMPLIFIED Q-LEARNING AGENT
// ===========================

/**
 * Simplified Q-Learning agent for LoRaWAN parameter optimization
 */
class SimpleQLearningAgent {
public:
    SimpleQLearningAgent(double learningRate = 0.1, double discountFactor = 0.9, double epsilon = 0.1);
    
    uint32_t SelectAction(const std::vector<double>& state);
    void UpdateQValue(const std::vector<double>& state, uint32_t action, double reward, const std::vector<double>& nextState);
    void LogPerformance(double episode, double avgReward);
    
private:
    double learningRate;
    double discountFactor; 
    double epsilon;
    
    // Simple Q-table (state hash -> action -> Q-value)
    std::map<uint32_t, std::vector<double>> qTable;
    
    uint32_t StateToHash(const std::vector<double>& state);
    double GetQValue(const std::vector<double>& state, uint32_t action);
    void SetQValue(const std::vector<double>& state, uint32_t action, double value);
    
    Ptr<UniformRandomVariable> randomGenerator;
};

SimpleQLearningAgent::SimpleQLearningAgent(double lr, double df, double eps) 
    : learningRate(lr), discountFactor(df), epsilon(eps) {
    randomGenerator = CreateObject<UniformRandomVariable>();
    randomGenerator->SetAttribute("Min", DoubleValue(0.0));
    randomGenerator->SetAttribute("Max", DoubleValue(1.0));
}

uint32_t SimpleQLearningAgent::StateToHash(const std::vector<double>& state) {
    // Simple hash function for state discretization
    uint32_t hash = 0;
    for (size_t i = 0; i < state.size(); i++) {
        // Discretize each state dimension to 10 levels
        uint32_t discreteValue = static_cast<uint32_t>(state[i] * 10) % 10;
        hash += discreteValue * static_cast<uint32_t>(std::pow(10, i));
    }
    return hash;
}

double SimpleQLearningAgent::GetQValue(const std::vector<double>& state, uint32_t action) {
    uint32_t hash = StateToHash(state);
    if (qTable.find(hash) == qTable.end()) {
        qTable[hash] = std::vector<double>(5, 0.0); // 5 actions, initialized to 0
    }
    return qTable[hash][action];
}

void SimpleQLearningAgent::SetQValue(const std::vector<double>& state, uint32_t action, double value) {
    uint32_t hash = StateToHash(state);
    if (qTable.find(hash) == qTable.end()) {
        qTable[hash] = std::vector<double>(5, 0.0);
    }
    qTable[hash][action] = value;
}

uint32_t SimpleQLearningAgent::SelectAction(const std::vector<double>& state) {
    if (randomGenerator->GetValue() < epsilon) {
        // Random action (exploration)
        return static_cast<uint32_t>(randomGenerator->GetValue() * 5); // 5 actions
    } else {
        // Greedy action (exploitation)
        double maxQ = GetQValue(state, 0);
        uint32_t bestAction = 0;
        
        for (uint32_t action = 1; action < 5; action++) {
            double q = GetQValue(state, action);
            if (q > maxQ) {
                maxQ = q;
                bestAction = action;
            }
        }
        return bestAction;
    }
}

void SimpleQLearningAgent::UpdateQValue(const std::vector<double>& state, uint32_t action, 
                                       double reward, const std::vector<double>& nextState) {
    // Find max Q-value for next state
    double maxNextQ = GetQValue(nextState, 0);
    for (uint32_t nextAction = 1; nextAction < 5; nextAction++) {
        double q = GetQValue(nextState, nextAction);
        if (q > maxNextQ) {
            maxNextQ = q;
        }
    }
    
    // Q-learning update rule
    double currentQ = GetQValue(state, action);
    double newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
    SetQValue(state, action, newQ);
}

void SimpleQLearningAgent::LogPerformance(double episode, double avgReward) {
    NS_LOG_INFO("RL Episode " << episode << " - Avg Reward: " << avgReward 
               << " - Q-table size: " << qTable.size());
}

// ===========================
// LoRaWAN INTEGRATION
// ===========================

// Global tracking structures
std::map<uint32_t, bool> activeTransmitters;
std::map<uint32_t, double> nodeTxPowers;
std::map<uint32_t, Vector> nodePositions;
std::map<uint32_t, Time> transmissionEndTimes;
std::map<uint32_t, uint8_t> nodeSpreadingFactors;

// RL-specific structures
std::shared_ptr<SimpleQLearningAgent> rlAgent;
std::map<uint32_t, std::vector<double>> deviceStates;
std::map<uint32_t, uint32_t> deviceActions;
std::map<uint32_t, double> deviceRewards;

// Performance tracking
struct PerformanceMetrics {
    uint32_t totalPackets;
    uint32_t receivedPackets;
    double totalEnergyConsumed;
    
    PerformanceMetrics() : totalPackets(0), receivedPackets(0), totalEnergyConsumed(0.0) {}
    
    double GetPDR() const {
        return totalPackets > 0 ? static_cast<double>(receivedPackets) / totalPackets : 0.0;
    }
    
    double GetEnergyPerPacket() const {
        return receivedPackets > 0 ? totalEnergyConsumed / receivedPackets : 0.0;
    }
};

PerformanceMetrics performanceMetrics;
Ptr<LoraChannel> globalChannel;

// Action space
enum ActionType {
    INCREASE_SF = 0,
    DECREASE_SF = 1,
    INCREASE_TP = 2,
    DECREASE_TP = 3,
    NO_CHANGE = 4
};

const uint32_t NUM_ACTIONS = 5;
const uint32_t STATE_SIZE = 4; // [SNR, SF, TP, PreTxRSSI]

// Reward function weights
const double PDR_WEIGHT = 1.0;
const double ENERGY_WEIGHT = 0.3;

/**
 * RSSI sampling structure
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

/**
 * Pre-TX RSSI sampling (simplified)
 */
RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    Vector txPos = nodePositions[txNodeId];
    
    // Calculate ambient noise with variations
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
        
        // Calculate received power from interferer
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

/**
 * Calculate SNR for state representation
 */
double CalculateSNR(uint32_t nodeId) {
    Vector nodePos = nodePositions[nodeId];
    Vector gatewayPos(0, 0, 15);
    
    double distance = CalculateDistance(nodePos, gatewayPos);
    double pathLoss = 20 * std::log10(distance) + 20 * std::log10(868e6) - 147.55;
    
    double txPower = nodeTxPowers[nodeId];
    double rxPower = txPower - pathLoss;
    double snr = rxPower - (-110.0); // Noise floor
    
    return snr;
}

/**
 * Get current state for RL agent
 */
std::vector<double> GetDeviceState(uint32_t nodeId) {
    double snr = CalculateSNR(nodeId);
    double sf = static_cast<double>(nodeSpreadingFactors[nodeId]);
    double tp = nodeTxPowers[nodeId];
    
    RssiSample rssiSample = SamplePreTxRssi(nodeId, Simulator::Now());
    double preTxRssi = rssiSample.preTxRssiDbm;
    
    // Normalize state values to [0,1] range
    std::vector<double> state = {
        (snr + 20.0) / 40.0,        // SNR normalized
        (sf - 7.0) / 5.0,           // SF normalized  
        (tp - 2.0) / 12.0,          // TP normalized
        (preTxRssi + 120.0) / 20.0  // RSSI normalized
    };
    
    return state;
}

/**
 * Apply RL action to device parameters
 */
void ApplyAction(uint32_t nodeId, uint32_t action) {
    uint8_t currentSF = nodeSpreadingFactors[nodeId];
    double currentTP = nodeTxPowers[nodeId];
    
    switch (action) {
        case INCREASE_SF:
            if (currentSF < 12) {
                nodeSpreadingFactors[nodeId] = currentSF + 1;
            }
            break;
        case DECREASE_SF:
            if (currentSF > 7) {
                nodeSpreadingFactors[nodeId] = currentSF - 1;
            }
            break;
        case INCREASE_TP:
            if (currentTP < 14.0) {
                nodeTxPowers[nodeId] = std::min(14.0, currentTP + 2.0);
            }
            break;
        case DECREASE_TP:
            if (currentTP > 2.0) {
                nodeTxPowers[nodeId] = std::max(2.0, currentTP - 2.0);
            }
            break;
        case NO_CHANGE:
        default:
            break;
    }
}

/**
 * Calculate reward based on PDR and energy consumption
 */
double CalculateReward(uint32_t nodeId, bool packetReceived) {
    double pdrReward = packetReceived ? 1.0 : -1.0;
    
    uint8_t sf = nodeSpreadingFactors[nodeId];
    double tp = nodeTxPowers[nodeId];
    
    // Energy penalty increases with SF and TP
    double energyPenalty = -(sf - 7.0) * 0.1 - (tp - 2.0) * 0.05;
    
    double reward = PDR_WEIGHT * pdrReward + ENERGY_WEIGHT * energyPenalty;
    
    return reward;
}

/**
 * Energy consumption calculation
 */
double CalculateTxEnergy(uint8_t sf, double txPowerDbm, double durationSeconds) {
    double baseCurrent = 20.0; // mA baseline
    double txCurrent = baseCurrent + (txPowerDbm - 2.0) * 2.0;
    double voltage = 3.3; // V
    
    double sfFactor = std::pow(2, sf - 7);
    
    return (txCurrent * voltage * durationSeconds * sfFactor) / 1000.0; // mJ
}

// Packet tracking
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
    double rlReward;
    uint32_t rlAction;
};

std::vector<PacketRecord> completedPackets;
std::map<uint32_t, uint32_t> deviceFrameCounters;

/**
 * Enhanced transmission start callback with RL
 */
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    // Get current state for RL
    std::vector<double> currentState = GetDeviceState(nodeId);
    deviceStates[nodeId] = currentState;
    
    // RL agent selects action
    if (rlAgent) {
        uint32_t action = rlAgent->SelectAction(currentState);
        deviceActions[nodeId] = action;
        ApplyAction(nodeId, action);
    }
    
    // Sample pre-TX RSSI
    RssiSample rssiSample = SamplePreTxRssi(nodeId, currentTime);
    
    // Mark as transmitting
    activeTransmitters[nodeId] = true;
    
    // Get transmission parameters
    uint8_t actualSF = nodeSpreadingFactors[nodeId];
    double actualTxPower = nodeTxPowers[nodeId];
    uint32_t actualBW = 125000;
    
    // Calculate transmission duration
    LoraTxParameters txParams;
    txParams.sf = actualSF;
    txParams.headerDisabled = false;
    txParams.codingRate = 1;
    txParams.bandwidthHz = actualBW;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (actualSF > 10);
    
    Time txDuration = LoraPhy::GetOnAirTime(packet->Copy(), txParams);
    transmissionEndTimes[nodeId] = currentTime + txDuration;
    
    // Create packet record
    PacketRecord record;
    record.timestamp = currentTime.GetSeconds();
    record.nodeId = nodeId;
    record.devEui = "DEV" + std::to_string(nodeId);
    record.fcnt = ++deviceFrameCounters[nodeId];
    record.txPowerDbm = actualTxPower;
    record.sf = actualSF;
    record.bw = actualBW;
    record.txPos = nodePositions[nodeId];
    record.preTxRssiDbm = rssiSample.preTxRssiDbm;
    record.ambientNoiseDbm = rssiSample.ambientNoiseDbm;
    record.packetReceived = false;
    record.gatewayRssiDbm = -999.0;
    record.gatewaySnrDb = -999.0;
    record.payloadBytes = packet->GetSize();
    record.rlReward = 0.0;
    record.rlAction = deviceActions.count(nodeId) ? deviceActions[nodeId] : NO_CHANGE;
    
    completedPackets.push_back(record);
    
    // Update performance metrics
    performanceMetrics.totalPackets++;
    
    double txEnergy = CalculateTxEnergy(actualSF, actualTxPower, txDuration.GetSeconds());
    performanceMetrics.totalEnergyConsumed += txEnergy;
    
    NS_LOG_INFO("RL-ADR TX START - Node " << nodeId << " at " << currentTime.GetSeconds() 
               << "s, SF: " << static_cast<int>(actualSF) << ", TP: " << actualTxPower 
               << " dBm, RSSI: " << rssiSample.preTxRssiDbm << " dBm");
    
    // Schedule transmission end
    Simulator::Schedule(txDuration, [nodeId]() {
        activeTransmitters[nodeId] = false;
    });
}

/**
 * Enhanced packet reception callback with RL learning
 */
void OnPacketReceived(Ptr<const Packet> packet) {
    Time rxTime = Simulator::Now();
    
    // Find corresponding packet record
    for (auto& record : completedPackets) {
        if (!record.packetReceived && 
            (rxTime.GetSeconds() - record.timestamp) >= 0.0 && 
            (rxTime.GetSeconds() - record.timestamp) <= 5.0) {
            
            record.packetReceived = true;
            performanceMetrics.receivedPackets++;
            
            // Calculate RL reward
            double reward = CalculateReward(record.nodeId, true);
            record.rlReward = reward;
            deviceRewards[record.nodeId] = reward;
            
            // Store experience and train RL agent
            if (rlAgent && deviceStates.count(record.nodeId)) {
                std::vector<double> nextState = GetDeviceState(record.nodeId);
                
                rlAgent->UpdateQValue(
                    deviceStates[record.nodeId],  // Previous state
                    deviceActions[record.nodeId], // Action taken
                    reward,                       // Reward received
                    nextState                     // Current state
                );
                
                deviceStates[record.nodeId] = nextState;
            }
            
            NS_LOG_INFO("RL-ADR RX SUCCESS - Node " << record.nodeId 
                       << " at " << rxTime.GetSeconds() << "s, Reward: " << reward);
            break;
        }
    }
}

/**
 * Write comprehensive results
 */
void WriteRLResults(const std::string& filename) {
    system("mkdir -p lorawan_rl_results");
    
    std::ofstream resultsFile("lorawan_rl_results/" + filename);
    resultsFile << "timestamp,nodeId,devEui,fcnt,txPowerDbm,sf,bw,txPosX,txPosY,txPosZ,"
                << "preTxRssiDbm,ambientNoiseDbm,packetReceived,gatewayRssiDbm,gatewaySnrDb,"
                << "payloadBytes,rlReward,rlAction,distanceToGW\n";
    
    Vector gatewayPos(0, 0, 15);
    for (const auto& record : completedPackets) {
        double distance = CalculateDistance(record.txPos, gatewayPos);
        
        resultsFile << std::fixed << std::setprecision(6)
                   << record.timestamp << "," << record.nodeId << "," << record.devEui << ","
                   << record.fcnt << "," << record.txPowerDbm << "," << static_cast<int>(record.sf) << ","
                   << record.bw << "," << record.txPos.x << "," << record.txPos.y << "," << record.txPos.z << ","
                   << record.preTxRssiDbm << "," << record.ambientNoiseDbm << "," << record.packetReceived << ","
                   << record.gatewayRssiDbm << "," << record.gatewaySnrDb << "," << record.payloadBytes << ","
                   << record.rlReward << "," << record.rlAction << "," << distance << "\n";
    }
    resultsFile.close();
    
    // Write summary
    std::ofstream summaryFile("lorawan_rl_results/" + filename.substr(0, filename.find_last_of('.')) + "_summary.csv");
    summaryFile << "metric,value\n";
    summaryFile << "total_packets," << performanceMetrics.totalPackets << "\n";
    summaryFile << "received_packets," << performanceMetrics.receivedPackets << "\n";
    summaryFile << "pdr_percent," << (performanceMetrics.GetPDR() * 100.0) << "\n";
    summaryFile << "total_energy_mj," << performanceMetrics.totalEnergyConsumed << "\n";
    summaryFile << "energy_per_packet_mj," << performanceMetrics.GetEnergyPerPacket() << "\n";
    summaryFile.close();
    
    std::cout << "\n=== RL-ADR SIMULATION RESULTS ===" << std::endl;
    std::cout << "Total packets transmitted: " << performanceMetrics.totalPackets << std::endl;
    std::cout << "Successfully received: " << performanceMetrics.receivedPackets 
              << " (" << (performanceMetrics.GetPDR() * 100.0) << "%)" << std::endl;
    std::cout << "Total energy consumed: " << performanceMetrics.totalEnergyConsumed << " mJ" << std::endl;
    std::cout << "Energy per received packet: " << performanceMetrics.GetEnergyPerPacket() << " mJ" << std::endl;
    std::cout << "Results saved to: lorawan_rl_results/ folder" << std::endl;
}

/**
 * Main simulation function
 */
int main(int argc, char* argv[]) {
    // Simulation parameters
    uint32_t nDevices = 15;
    double simulationTime = 600.0;
    double appPeriodSeconds = 30.0;
    double radius = 1200.0;
    std::string resultFileName = "rl_adr_results.csv";
    
    // Parse command line
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("resultsFile", "Output results file name", resultFileName);
    cmd.Parse(argc, argv);
    
    // Enable logging
    LogComponentEnable("RLAdrSimpleExample", LOG_LEVEL_INFO);
    
    std::cout << "\n=== Reinforcement Learning ADR with Pre-TX RSSI ===" << std::endl;
    std::cout << "Devices: " << nDevices << std::endl;
    std::cout << "Simulation time: " << simulationTime << " seconds" << std::endl;
    std::cout << "Area radius: " << radius << " meters" << std::endl;
    std::cout << "Using Q-Learning for intelligent ADR" << std::endl;
    
    // Initialize RL agent
    rlAgent = std::make_shared<SimpleQLearningAgent>(0.1, 0.9, 0.1);
    
    // Reset global state
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    deviceStates.clear();
    deviceActions.clear();
    deviceRewards.clear();
    performanceMetrics = PerformanceMetrics();
    
    // Create nodes
    NodeContainer endDevices;
    endDevices.Create(nDevices);
    
    NodeContainer gateways;
    gateways.Create(1);
    
    // Set up mobility
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    // Gateway at center
    allocator->Add(Vector(0, 0, 15));
    nodePositions[gateways.Get(0)->GetId()] = Vector(0, 0, 15);
    
    // Randomize device positions and parameters
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-radius));
    xRand->SetAttribute("Max", DoubleValue(radius));
    
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-radius));
    yRand->SetAttribute("Max", DoubleValue(radius));
    
    Ptr<UniformRandomVariable> powerRand = CreateObject<UniformRandomVariable>();
    powerRand->SetAttribute("Min", DoubleValue(2.0));
    powerRand->SetAttribute("Max", DoubleValue(14.0));
    
    std::vector<uint8_t> spreadingFactors = {7, 8, 9, 10, 11, 12};
    Ptr<UniformRandomVariable> sfRand = CreateObject<UniformRandomVariable>();
    sfRand->SetAttribute("Min", DoubleValue(0));
    sfRand->SetAttribute("Max", DoubleValue(spreadingFactors.size() - 1));
    
    for (uint32_t i = 0; i < nDevices; i++) {
        double x = xRand->GetValue();
        double y = yRand->GetValue();
        Vector pos(x, y, 1.5);
        
        allocator->Add(pos);
        
        uint32_t nodeId = endDevices.Get(i)->GetId();
        nodePositions[nodeId] = pos;
        nodeTxPowers[nodeId] = powerRand->GetValue();
        nodeSpreadingFactors[nodeId] = spreadingFactors[static_cast<uint32_t>(sfRand->GetValue())];
    }
    
    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(NodeContainer(endDevices, gateways));
    
    // Create channel
    Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetPathLossExponent(3.76);
    lossModel->SetReference(1, 7.7);
    
    Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
    shadowing->SetAttribute("Variable", StringValue("ns3::LogNormalRandomVariable[Mu=0|Sigma=8.0]"));
    lossModel->SetNext(shadowing);
    
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    globalChannel = CreateObject<LoraChannel>(lossModel, delay);
    
    // Set up LoRaWAN stack
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(globalChannel);
    
    LorawanMacHelper macHelper = LorawanMacHelper();
    LoraHelper helper = LoraHelper();
    
    // Install on end devices
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, endDevices);
    
    // Set initial parameters
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        Ptr<LoraNetDevice> loraNetDevice = DynamicCast<LoraNetDevice>(endDevicesNetDevs.Get(i));
        Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(loraNetDevice->GetPhy());
        
        if (edPhy) {
            uint32_t nodeId = endDevices.Get(i)->GetId();
            edPhy->SetSpreadingFactor(nodeSpreadingFactors[nodeId]);
        }
    }
    
    // Install on gateway
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    NetDeviceContainer gwNetDevices = helper.Install(phyHelper, macHelper, gateways);
    
    // Set up callbacks
    Config::Connect("/NodeList/*/DeviceList/0/$ns3::LoraNetDevice/Phy/$ns3::EndDeviceLoraPhy/StartSending",
                    MakeCallback(&OnTransmissionStart));
    
    Config::Connect("/NodeList/*/DeviceList/0/$ns3::LoraNetDevice/Phy/$ns3::GatewayLoraPhy/ReceivedPacket",
                    MakeCallback(&OnPacketReceived));
    
    // Install applications
    PeriodicSenderHelper appHelper = PeriodicSenderHelper();
    appHelper.SetPeriod(Seconds(appPeriodSeconds));
    appHelper.SetPacketSize(20);
    ApplicationContainer appContainer = appHelper.Install(endDevices);
    
    appContainer.Start(Seconds(1));
    appContainer.Stop(Seconds(simulationTime - 1));
    
    std::cout << "Starting RL-ADR simulation..." << std::endl;
    
    // Run simulation
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    
    // Write results
    WriteRLResults(resultFileName);
    
    Simulator::Destroy();
    
    return 0;
}