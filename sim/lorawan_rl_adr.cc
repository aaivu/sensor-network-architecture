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
#include "ns3/rectangle.h"

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
#include <sstream>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("AdvancedDDQNPERADRExample");

// ============================================================================
// Modular includes -- each header contains one logical component.
// Include order matches original code order to preserve type dependencies.
// ============================================================================
#include "include/simple_ddqn.h"
#include "include/device_history.h"
#include "include/dueling_ddqn.h"
#include "include/reward.h"
#include "include/ppo_agent.h"
#include "include/marl_agent.h"
#include "include/marl_ppo_agent.h"
#include "include/mappo_gat_agent.h"
#include "include/globals.h"
#include "include/environment.h"
#include "include/sim_core.h"
#include "include/update_ppo.h"
#include "include/update_marl.h"
#include "include/update_marl_ppo.h"
#include "include/update_mappo_gat.h"

/**
 * Run a single simulation
 */
void RunSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                  double radius, const std::string& csvFileName, ADRMethod adrMethod, uint32_t nWifiInterferers, uint32_t nLteInterferers);

/**
 * Get optimal initial SF based on distance AND congestion level
 * In low-congestion scenarios (high app period), use SF7 regardless of distance
 * because link budget typically has margin and SF7 reduces airtime
 */
uint8_t getInitialSFForDistance(double distance) {
    // Check global app period - if high (low congestion), always use SF7
    // This matches No ADR behavior which uses SF7 for all devices
    if (globalAppPeriodSeconds >= 100.0) {
        // Low congestion - use SF7 to minimize airtime and collisions
        // Even far devices have good SNR margin in this scenario
        return 7;
    }
    
    // High congestion scenario - use distance-based SF for reliability
    if (distance < 300) return 7;
    if (distance < 500) return 8;
    if (distance < 800) return 9;
    if (distance < 1200) return 10;
    if (distance < 1600) return 11;
    return 12;
}

/**
 * Get optimal initial TX power based on distance AND congestion level
 * In low-congestion scenarios, use maximum power to ensure reliable delivery
 */
double getInitialTPForDistance(double distance) {
    // In low congestion, always use 14dBm to maximize link budget
    // This helps ensure packets are received even from far devices
    if (globalAppPeriodSeconds >= 100.0) {
        return 14.0;  // Maximum power for reliability
    }
    
    // High congestion - use distance-based power to save energy
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
    uint32_t nLteInterferers = 8;
    
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.AddValue("adr", "ADR mode: 'off', 'on', 'ddqn', 'ppo', 'marl', 'mappo_gat', or 'all'", adrModeStr);
    cmd.AddValue("environmental", "Enable environmental effects modeling", environmentalModeling);
    cmd.AddValue("wifiInterferers", "Number of WiFi interfering nodes", nWifiInterferers);
    cmd.AddValue("lteInterferers", "Number of LTE mobile interfering nodes (moving)", nLteInterferers);
    cmd.AddValue("coordDivisor",   "Norm. divisor for n_same coordination signal (default 3)", g_coordDivisor);
    cmd.AddValue("rewardSchedule", "MARL reward weights: 0=adaptive, 1=uniform, 2=outcome_only", g_rewardSchedule);
    cmd.AddValue("rngSeed",        "RNG seed for reproducibility (default 12345)", g_rngSeed);
    cmd.Parse(argc, argv);
    
    enableEnvironmentalModeling = environmentalModeling;
    
    // Set global app period for congestion estimation
    globalAppPeriodSeconds = appPeriodSeconds;
    
    LogComponentEnable("AdvancedDDQNPERADRExample", LOG_LEVEL_INFO);
    
    // **CLEANUP: Remove old dataset files before any new simulations**
    std::cout << "🧹 Cleaning up old lorawan_datasets folder..." << std::endl;
    system("rm -rf lorawan_datasets");
    std::cout << "✅ Old datasets removed" << std::endl;
    
    std::cout << "\n=== ENHANCED ns-3 LoRaWAN Simulation ===" << std::endl;
    std::cout << "Environmental modeling: " << (enableEnvironmentalModeling ? "ENABLED" : "DISABLED") << std::endl;
    if (enableEnvironmentalModeling) {
        std::cout << "  - Urban obstacle modeling" << std::endl;
        std::cout << "  - WiFi interference modeling (" << nWifiInterferers << " nodes)" << std::endl;
        std::cout << "  - LTE mobile interference (" << nLteInterferers << " devices with human mobility)" << std::endl;
        std::cout << "  - Hardware variability modeling" << std::endl;
    }
    
    if (adrModeStr == "all") {
        std::cout << "\n========================================" << std::endl;
        std::cout << "RUNNING ALL ADR METHODS COMPARISON" << std::endl;
        std::cout << "Simulation Time: " << simulationTime << " seconds" << std::endl;
        std::cout << "Number of Devices: " << nDevices << std::endl;
        std::cout << "App Period: " << appPeriodSeconds << " seconds" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "\n[1/7] Running simulation for NO ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "no_adr_" + csvFileName, ADRMethod::OFF, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[2/7] Running simulation for CLASSICAL ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "adr_" + csvFileName, ADRMethod::ON, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[3/7] Running simulation for DDQN-PER ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "ddqn_adr_" + csvFileName, ADRMethod::DDQN, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[4/7] Running simulation for PPO ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "ppo_adr_" + csvFileName, ADRMethod::PPO, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[5/7] Running simulation for MARL ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "marl_adr_" + csvFileName, ADRMethod::MARL, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[6/7] Running simulation for MAPPO-GAT ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "mappo_gat_adr_" + csvFileName, ADRMethod::MAPPO_GAT, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n[7/7] Running simulation for MARL-PPO ADR..." << std::endl;
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, "marl_ppo_adr_" + csvFileName, ADRMethod::MARL_PPO, nWifiInterferers, nLteInterferers);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL ADR METHODS COMPARISON COMPLETED" << std::endl;
        std::cout << "All seven methods used the same:" << std::endl;
        std::cout << "  - Simulation time: " << simulationTime << "s" << std::endl;
        std::cout << "  - Device count: " << nDevices << std::endl;
        std::cout << "  - RNG seed: 12345 (fixed)" << std::endl;
        std::cout << "  - Packet interval: " << appPeriodSeconds << "s" << std::endl;
        std::cout << "Output files:" << std::endl;
        std::cout << "  - no_adr_"       << csvFileName << std::endl;
        std::cout << "  - adr_"          << csvFileName << std::endl;
        std::cout << "  - ddqn_adr_"     << csvFileName << std::endl;
        std::cout << "  - ppo_adr_"      << csvFileName << std::endl;
        std::cout << "  - marl_adr_"     << csvFileName << std::endl;
        std::cout << "  - mappo_gat_adr_" << csvFileName << std::endl;
        std::cout << "  - marl_ppo_adr_"  << csvFileName << std::endl;
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
        } else if (adrModeStr == "ppo") {
            adrMethod = ADRMethod::PPO;
            filePrefix = "ppo_adr_";
        } else if (adrModeStr == "marl") {
            adrMethod = ADRMethod::MARL;
            filePrefix = "marl_adr_";
        } else if (adrModeStr == "mappo_gat") {
            adrMethod = ADRMethod::MAPPO_GAT;
            filePrefix = "mappo_gat_adr_";
        } else if (adrModeStr == "marl_ppo") {
            adrMethod = ADRMethod::MARL_PPO;
            filePrefix = "marl_ppo_adr_";
        } else {
            adrMethod = ADRMethod::OFF;
            filePrefix = "no_adr_";
        }
        RunSimulation(nDevices, simulationTime, appPeriodSeconds, radius, filePrefix + csvFileName, adrMethod, nWifiInterferers, nLteInterferers);
    }
    
    return 0;
}

void RunSimulation(uint32_t nDevices, double simulationTime, double appPeriodSeconds, 
                  double radius, const std::string& csvFileName, ADRMethod adrMethod, uint32_t nWifiInterferers, uint32_t nLteInterferers) {
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "Starting simulation with parameters:" << std::endl;
    std::cout << "  Duration: " << simulationTime << " seconds" << std::endl;
    std::cout << "  App Period: " << appPeriodSeconds << " seconds" << std::endl;
    std::cout << "  Devices: " << nDevices << std::endl;
    std::string adrModeLabel = "OFF";
    if (adrMethod == ADRMethod::ON) adrModeLabel = "CLASSICAL";
    else if (adrMethod == ADRMethod::DDQN) adrModeLabel = "DDQN-PER (Optimized)";
    else if (adrMethod == ADRMethod::PPO) adrModeLabel = "PPO";
    else if (adrMethod == ADRMethod::MARL)      adrModeLabel = "MARL";
    else if (adrMethod == ADRMethod::MAPPO_GAT) adrModeLabel = "MAPPO-GAT";
    else if (adrMethod == ADRMethod::MARL_PPO)  adrModeLabel = "MARL-PPO";
    std::cout << "  ADR Mode: " << adrModeLabel << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    endDevicesNetDevices.clear();
    ddqnAgents.clear();
    optimizedAgents.clear();  // Clear optimized agents
    ppoAgents.clear();        // Clear PPO agents
    marlAgents.clear();       // Clear MARL agents
    mappoGatAgents.clear();   // Clear MAPPO-GAT agents
    marlPpoAgents.clear();    // Clear MARL-PPO agents
    deviceChannels.clear();   // Clear channel tracking for MARL / MAPPO-GAT
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
    
    // Set seed for reproducible results (exposed via --rngSeed for ablation sweeps)
    RngSeedManager::SetSeed(g_rngSeed);
    RngSeedManager::SetRun(1);
    
    if (enableEnvironmentalModeling) {
        SetupUrbanEnvironment(radius);
        SetupWiFiInterferers(radius, nWifiInterferers);
        SetupLTEMobileInterferers(radius, nLteInterferers);
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
                          << (adrMethod == ADRMethod::DDQN ? "DDQN ADR" : 
                              (adrMethod == ADRMethod::PPO ? "PPO ADR" : 
                              (adrMethod == ADRMethod::MARL ? "MARL ADR" : 
                              (adrMethod == ADRMethod::MAPPO_GAT ? "MAPPO-GAT ADR" : "No ADR")))) << std::endl;
            }
        }
        
        if (adrMethod == ADRMethod::DDQN) {
            std::cout << "✅ DDQN-PER ADR enabled" << std::endl;
        } else if (adrMethod == ADRMethod::PPO) {
            std::cout << "✅ PPO ADR enabled" << std::endl;
        } else if (adrMethod == ADRMethod::MARL) {
            std::cout << "✅ MARL ADR enabled" << std::endl;
        } else if (adrMethod == ADRMethod::MAPPO_GAT) {
            std::cout << "✅ MAPPO-GAT ADR enabled" << std::endl;
        } else {
            std::cout << "ADR disabled" << std::endl;
        }
    }
    
    // Initialize RL agents (DDQN, PPO, MARL, or MAPPO-GAT)
    if (adrMethod == ADRMethod::DDQN || adrMethod == ADRMethod::PPO ||
        adrMethod == ADRMethod::MARL || adrMethod == ADRMethod::MAPPO_GAT) {
        std::string agentType = (adrMethod == ADRMethod::DDQN ? "DDQN" :
                                (adrMethod == ADRMethod::PPO  ? "PPO"  :
                                (adrMethod == ADRMethod::MARL ? "MARL" : "MAPPO-GAT")));
        std::cout << "\n🎯 Initializing " << agentType << " agents..." << std::endl;
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
            
            // Initialize channel tracking for MARL coordination
            deviceChannels[nodeId] = i % 8;
            
            // Apply to device immediately
            // CRITICAL: In low-congestion mode, DON'T override hardware variability settings
            Ptr<EndDeviceLorawanMac> mac = DynamicCast<EndDeviceLorawanMac>(
                endDevicesNetDevices[i]->GetMac());
            if (mac && globalAppPeriodSeconds < 100.0) {
                // High congestion - apply RL-optimized initial parameters
                mac->SetDataRate(12 - initialSF);  // DR = 12 - SF
                mac->SetTransmissionPowerDbm(initialTP);
            }
            // In low-congestion mode, device keeps its default hardware variability settings
            
            // Create agents based on ADR method
            if (adrMethod == ADRMethod::DDQN) {
                // Create OPTIMIZED DDQN agent with interference awareness
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
            } else if (adrMethod == ADRMethod::PPO) {
                // Create PPO agent
                ppoAgents[nodeId] = std::make_unique<PPOAgent>();
            } else if (adrMethod == ADRMethod::MARL) {
                // Create MARL agent
                marlAgents[nodeId] = std::make_unique<MARLAgent>(
                    0.0005,  // learningRate
                    0.3,     // epsilon - start at 30% exploration
                    0.999,   // epsilonDecay
                    0.05,    // minEpsilon
                    0.95,    // gamma
                    50       // targetUpdateFreq
                );
            } else if (adrMethod == ADRMethod::MAPPO_GAT) {
                // Create MAPPO-GAT agent (shared GAT encoder + centralized critic)
                mappoGatAgents[nodeId] = std::make_unique<MAPPOGATAgent>(
                    0.0003,  // actorLR
                    0.3,     // epsilon
                    0.999,   // epsilonDecay
                    0.05,    // minEpsilon
                    0.95,    // gamma
                    0.95,    // gaeLambda
                    0.2      // clipEpsilon
                );
            } else if (adrMethod == ADRMethod::MARL_PPO) {
                // Create MARL-PPO agent (PPO with coordination signal)
                marlPpoAgents[nodeId] = std::make_unique<MARLPPOAgent>(
                    0.0003,  // actorLR
                    0.3,     // epsilon
                    0.999,   // epsilonDecay
                    0.05,    // minEpsilon
                    0.95,    // gamma
                    0.95,    // gaeLambda
                    0.2      // clipEpsilon
                );
            }
            
            std::cout << "  Device " << nodeId << ": distance=" << std::fixed << std::setprecision(0) 
                      << distance << "m, initialSF=" << (int)initialSF 
                      << ", initialTP=" << initialTP << "dBm"
                      << ", channel=" << (int)deviceHistories[nodeId].currentChannel << std::endl;
        }
        
        if (adrMethod == ADRMethod::DDQN) {
            std::cout << "  ✅ Using Dueling DQN with interference detection" << std::endl;
            std::cout << "  ✅ Extended action space (48 actions including channel changes)" << std::endl;
            std::cout << "  ✅ Collision detection heuristics enabled" << std::endl;
        } else if (adrMethod == ADRMethod::PPO) {
            std::cout << "  ✅ Using PPO with continuous action space" << std::endl;
            std::cout << "  ✅ Actor-Critic architecture with GAE" << std::endl;
        } else if (adrMethod == ADRMethod::MARL) {
            std::cout << "  ✅ Using MARL with coordination signals" << std::endl;
            std::cout << "  ✅ Collision-aware channel selection" << std::endl;
            std::cout << "  ✅ Extended action space (48 actions)" << std::endl;
        } else if (adrMethod == ADRMethod::MAPPO_GAT) {
            std::cout << "  ✅ Using MAPPO-GAT (Multi-Agent PPO + Graph Attention Network)" << std::endl;
            std::cout << "  ✅ Shared GAT encoder: attends over on-channel neighbours" << std::endl;
            std::cout << "  ✅ Centralized critic (CTDE): global state mean for stable advantages" << std::endl;
            std::cout << "  ✅ Extended action space (48 actions)" << std::endl;
        } else if (adrMethod == ADRMethod::MARL_PPO) {
            std::cout << "  ✅ Using MARL-PPO (Independent PPO with coordination signal)" << std::endl;
            std::cout << "  ✅ 12-dim coordinated state: local 11-dim + n_same_ch/3 coordination" << std::endl;
            std::cout << "  ✅ Coordination-aware channel-hop pre-selection (same as MARL-DDQN)" << std::endl;
            std::cout << "  ✅ Softmax actor + GAE clipped PPO objective" << std::endl;
            std::cout << "  ✅ Extended action space (48 actions)" << std::endl;
        }
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
    std::string adrModeStr = "OFF";
    if (adrMethod == ADRMethod::ON) adrModeStr = "CLASSICAL";
    else if (adrMethod == ADRMethod::DDQN) adrModeStr = "DDQN-PER";
    else if (adrMethod == ADRMethod::PPO) adrModeStr = "PPO";
    else if (adrMethod == ADRMethod::MARL)      adrModeStr = "MARL";
    else if (adrMethod == ADRMethod::MAPPO_GAT) adrModeStr = "MAPPO-GAT";
    std::cout << "ADR Mode: " << adrModeStr << std::endl;
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
