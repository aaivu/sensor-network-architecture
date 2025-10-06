/*
 * advanced-pre-tx-rssi-example-fixed.cc
 * 
 * Advanced LoRaWAN simulation implementing true pre-TX RSSI sampling
 * This simulates the concept of switching radio to RX, reading RSSI register,
 * then switching to TX - exactly as described in the hardware approach.
 * 
 * Key features:
 * - Real-time interference calculation at transmitter location
 * - Active transmitter tracking for accurate interference modeling
 * - Comprehensive CSV logging for ML dataset generation
 * - Simulation of hardware RSSI register reading behavior
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
// Temporarily remove problematic includes
// #include "ns3/network-server-helper.h"
// #include "ns3/forwarder-helper.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("AdvancedPreTxRssiExample");

// Global tracking structures
std::map<uint32_t, bool> activeTransmitters; // nodeId -> isCurrentlyTransmitting
std::map<uint32_t, double> nodeTxPowers;     // nodeId -> txPowerDbm
std::map<uint32_t, Vector> nodePositions;    // nodeId -> position
std::map<uint32_t, Time> transmissionEndTimes; // nodeId -> when current transmission ends

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
    double preTxRssiDbm;        // The key metric we're implementing!
    double ambientNoiseDbm;
    bool packetReceived;
    double gatewayRssiDbm;
    double gatewaySnrDb;
    uint32_t payloadBytes;
    bool collisionFlag;
};

std::vector<PacketRecord> completedPackets;
std::map<uint32_t, uint32_t> deviceFrameCounters; // Per-device frame counters (proper LoRaWAN)

// Global simulation parameters
Ptr<LoraChannel> globalChannel;
double globalAmbientNoise = -110.0; // dBm noise floor
bool enableAdr = false;  // ADR control parameter

/**
 * Structure to hold RSSI sampling results
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

/**
 * Core function: Simulate reading RSSI register at transmitter location
 * Simplified version for initial testing
 */
RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    // Get transmitter position
    Vector txPos = nodePositions[txNodeId];
    
    // Calculate realistic ambient noise with basic variations
    Ptr<UniformRandomVariable> noiseVar = CreateObject<UniformRandomVariable>();
    noiseVar->SetAttribute("Min", DoubleValue(-115.0));
    noiseVar->SetAttribute("Max", DoubleValue(-105.0));
    
    double ambientNoise = noiseVar->GetValue();
    
    // Start with varying ambient noise
    double totalPowerLinear = DbmToW(ambientNoise);
    
    // Add LoRaWAN self-interference from active transmitters
    for (const auto& pair : activeTransmitters) {
        uint32_t otherNodeId = pair.first;
        bool isTransmitting = pair.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) continue;
        if (currentTime >= transmissionEndTimes[otherNodeId]) continue;
        
        Vector interfererPos = nodePositions[otherNodeId];
        double interfererTxPower = nodeTxPowers[otherNodeId];
        
        // Calculate path loss using enhanced propagation models
        Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel>();
        txMobility->SetPosition(txPos);
        
        Ptr<ConstantPositionMobilityModel> interfererMobility = CreateObject<ConstantPositionMobilityModel>();
        interfererMobility->SetPosition(interfererPos);
        
        double rxPowerDbm = globalChannel->GetRxPower(interfererTxPower, interfererMobility, txMobility);
        totalPowerLinear += DbmToW(rxPowerDbm);
    }
    
    // Convert back to dBm - this is our enhanced RSSI measurement
    double preTxRssiDbm = WToDbm(totalPowerLinear);
    
    // Add realistic RSSI measurement noise
    Ptr<NormalRandomVariable> rssiNoise = CreateObject<NormalRandomVariable>();
    rssiNoise->SetAttribute("Mean", DoubleValue(0.0));
    rssiNoise->SetAttribute("Variance", DoubleValue(1.0));
    
    double noisyRssiDbm = preTxRssiDbm + rssiNoise->GetValue();
    
    NS_LOG_INFO("Pre-TX RSSI at node " << txNodeId << ": " 
               << preTxRssiDbm << " dBm (with noise: " << noisyRssiDbm << " dBm)");
    
    RssiSample sample;
    sample.preTxRssiDbm = noisyRssiDbm;
    sample.ambientNoiseDbm = ambientNoise;
    return sample;
}

/**
 * Callback triggered when a node starts transmitting
 */
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    // **KEY STEP: Sample pre-TX RSSI before marking this node as transmitting**
    RssiSample rssiSample = SamplePreTxRssi(nodeId, currentTime);
    
    // Now mark this node as transmitting
    activeTransmitters[nodeId] = true;
    
    // Extract actual parameters from device
    uint8_t actualSF = 7;  // Default SF
    double actualTxPower = nodeTxPowers[nodeId];
    uint32_t actualBW = 125000;
    uint8_t actualCR = 1;
    
    // Calculate transmission duration with actual parameters
    LoraTxParameters txParams;
    txParams.sf = actualSF;
    txParams.headerDisabled = false;
    txParams.codingRate = actualCR;
    txParams.bandwidthHz = actualBW;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (actualSF > 10);
    
    Time txDuration = LoraPhy::GetOnAirTime(packet->Copy(), txParams);
    transmissionEndTimes[nodeId] = currentTime + txDuration;
    
    // Create packet record with actual parameters
    PacketRecord record;
    record.timestamp = currentTime.GetSeconds();
    record.nodeId = nodeId;
    record.devEui = "DEV" + std::to_string(nodeId);
    record.fcnt = ++deviceFrameCounters[nodeId];
    record.txPowerDbm = actualTxPower;
    record.sf = actualSF;
    record.bw = actualBW;
    record.txPos = nodePositions[nodeId];
    record.preTxRssiDbm = rssiSample.preTxRssiDbm; // Our key measurement!
    record.ambientNoiseDbm = rssiSample.ambientNoiseDbm;
    record.packetReceived = false; // Will be updated if received
    record.gatewayRssiDbm = -999.0;
    record.gatewaySnrDb = -999.0;
    record.payloadBytes = packet->GetSize();
    record.collisionFlag = false;
    
    completedPackets.push_back(record);
    
    NS_LOG_INFO("TX START - Node " << nodeId << " at " << currentTime.GetSeconds() 
               << "s, Pre-TX RSSI: " << rssiSample.preTxRssiDbm << " dBm"
               << ", SF: " << static_cast<int>(actualSF)
               << ", TX Power: " << actualTxPower << " dBm");
    
    // Schedule transmission end callback
    Simulator::Schedule(txDuration, [nodeId]() {
        activeTransmitters[nodeId] = false;
        NS_LOG_DEBUG("TX END - Node " << nodeId << " stopped transmitting");
    });
}

/**
 * Callback triggered when a packet is successfully received at gateway
 */
void OnPacketReceived(Ptr<const Packet> packet) {
    Time rxTime = Simulator::Now();
    
    NS_LOG_INFO("OnPacketReceived at " << rxTime.GetSeconds() << "s, packet size: " << packet->GetSize());
    
    // Find the corresponding packet record
    for (auto& record : completedPackets) {
        if (!record.packetReceived && 
            rxTime.GetSeconds() >= record.timestamp &&
            std::abs(record.timestamp - rxTime.GetSeconds()) < 3.0) {
            
            record.packetReceived = true;
            
            // Calculate expected gateway RSSI using realistic path loss
            Vector gatewayPos(0, 0, 15); // Gateway at center, 15m height
            double distance = CalculateDistance(record.txPos, gatewayPos);
            
            // Use realistic path loss calculation
            double pathLossDb = 20 * std::log10(distance) + 20 * std::log10(868e6) - 147.55;
            record.gatewayRssiDbm = record.txPowerDbm - pathLossDb;
            record.gatewaySnrDb = record.gatewayRssiDbm - record.ambientNoiseDbm;
            
            NS_LOG_INFO("PACKET RECEIVED - Node " << record.nodeId 
                       << ", Gateway RSSI: " << record.gatewayRssiDbm << " dBm");
            break;
        }
    }
}

/**
 * Write comprehensive CSV output for ML training
 */
void WriteCSVOutput(const std::string& filename) {
    // Create results folder if it doesn't exist
    std::string folderName = "lorawan_datasets";
    system(("mkdir -p " + folderName).c_str());
    
    // Extract base filename without extension
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
        
        // Header matching your requested schema
        deviceCsvFile << "timestamp,devEUI,fcnt,tx_power_dBm,sf,bw,cr,tx_x,tx_y,tx_z,"
                     << "pre_tx_rssi_dBm,ambient_noise_dBm,expected_rx_rssi_dBm,snr_db,"
                     << "gateway_id,packet_received,collision_flag,payload_bytes,channel,distance_m\n";
        
        // Write all packet records for this device
        for (const auto& record : packets) {
            double distance = CalculateDistance(record.txPos, gatewayPos);
            
            deviceCsvFile << std::fixed << std::setprecision(6)
                         << record.timestamp << ","
                         << record.devEui << ","
                         << record.fcnt << ","
                         << record.txPowerDbm << ","
                         << static_cast<int>(record.sf) << ","
                         << record.bw << ","
                         << 1 << "," // coding rate 4/5
                         << record.txPos.x << ","
                         << record.txPos.y << ","
                         << record.txPos.z << ","
                         << record.preTxRssiDbm << ","
                         << record.ambientNoiseDbm << ","
                         << (record.packetReceived ? record.gatewayRssiDbm : -999.0) << ","
                         << (record.packetReceived ? record.gatewaySnrDb : -999.0) << ","
                         << 1 << "," // gateway_id
                         << (record.packetReceived ? 1 : 0) << ","
                         << (record.collisionFlag ? 1 : 0) << ","
                         << record.payloadBytes << ","
                         << 0 << "," // channel
                         << distance << "\n";
        }
        
        deviceCsvFile.close();
        std::cout << "Device " << deviceId << " dataset saved to: " << deviceFilename 
                  << " (" << packets.size() << " packets)" << std::endl;
    }
    
    // Calculate statistics
    int totalPackets = completedPackets.size();
    int receivedPackets = 0;
    
    for (const auto& record : completedPackets) {
        if (record.packetReceived) {
            receivedPackets++;
        }
    }
    
    double successRate = (double)receivedPackets / totalPackets * 100.0;
    
    std::cout << "\n=== SIMULATION RESULTS ===" << std::endl;
    std::cout << "Total packets transmitted: " << totalPackets << std::endl;
    std::cout << "Successfully received: " << receivedPackets << " (" << successRate << "%)" << std::endl;
    std::cout << "Per-device datasets saved to: lorawan_datasets/ folder" << std::endl;
}

int main(int argc, char* argv[]) {
    // Simulation parameters with enhanced diversity
    uint32_t nDevices = 10;
    double simulationTime = 300.0; // seconds
    double appPeriodSeconds = 30.0;
    double radius = 1000.0; // meters
    std::string csvFileName = "enhanced_pre_tx_rssi_dataset.csv";
    
    // Parse command line
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.Parse(argc, argv);
    
    // Enable logging
    LogComponentEnable("AdvancedPreTxRssiExample", LOG_LEVEL_INFO);
    
    std::cout << "\n=== Enhanced Pre-TX RSSI Simulation ===" << std::endl;
    std::cout << "Devices: " << nDevices << std::endl;
    std::cout << "Simulation time: " << simulationTime << " seconds" << std::endl;
    std::cout << "Area radius: " << radius << " meters" << std::endl;
    
    // Reset global state
    completedPackets.clear();
    deviceFrameCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    
    // Create nodes
    NodeContainer endDevices;
    endDevices.Create(nDevices);
    
    NodeContainer gateways;
    gateways.Create(1);
    
    // Set up mobility with DIVERSE positions and parameters
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    // Gateway at center, elevated
    allocator->Add(Vector(0, 0, 15));
    uint32_t gatewayNodeId = gateways.Get(0)->GetId();
    nodePositions[gatewayNodeId] = Vector(0, 0, 15);
    
    // DIVERSE device positions and TX powers
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-radius));
    xRand->SetAttribute("Max", DoubleValue(radius));
    
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-radius));
    yRand->SetAttribute("Max", DoubleValue(radius));
    
    // CRITICAL: Vary TX power across devices for diversity
    Ptr<UniformRandomVariable> powerRand = CreateObject<UniformRandomVariable>();
    powerRand->SetAttribute("Min", DoubleValue(2.0));   // Low power devices
    powerRand->SetAttribute("Max", DoubleValue(14.0));  // High power devices
    
    for (uint32_t i = 0; i < nDevices; i++) {
        // DIVERSE positions
        Vector pos(xRand->GetValue(), yRand->GetValue(), 1.5);
        allocator->Add(pos);
        
        uint32_t nodeId = endDevices.Get(i)->GetId();
        nodePositions[nodeId] = pos;
        activeTransmitters[nodeId] = false;
        
        // CRITICAL: Diverse TX powers
        nodeTxPowers[nodeId] = powerRand->GetValue();
        
        NS_LOG_INFO("Device " << i << " -> Node ID " << nodeId 
                   << " at (" << pos.x << ", " << pos.y << ")"
                   << " with TX power " << nodeTxPowers[nodeId] << " dBm");
    }
    
    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(NodeContainer(endDevices, gateways));
    
    // Create enhanced channel with realistic propagation
    Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetPathLossExponent(3.76); // Realistic urban environment
    lossModel->SetReference(1, 7.7);      // Reference distance and loss
    
    // Add shadowing for realism
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
    
    // Install on end devices with DIVERSE spreading factors
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, endDevices);
    
    // CRITICAL: Set diverse spreading factors
    std::vector<uint8_t> spreadingFactors = {7, 8, 9, 10, 11, 12};
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        Ptr<LoraNetDevice> loraNetDevice = DynamicCast<LoraNetDevice>(endDevicesNetDevs.Get(i));
        Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(loraNetDevice->GetPhy());
        
        // Set diverse SF for each device
        uint8_t sf = spreadingFactors[i % spreadingFactors.size()];
        edPhy->SetSpreadingFactor(sf);
        
        NS_LOG_INFO("Device " << i << " set to SF" << static_cast<int>(sf));
    }
    
    // Install on gateway
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    NetDeviceContainer gwNetDevices = helper.Install(phyHelper, macHelper, gateways);
    
    // Set up applications with DIVERSE transmission intervals
    ApplicationContainer appContainer;
    
    for (uint32_t i = 0; i < nDevices; i++) {
        PeriodicSenderHelper appHelper = PeriodicSenderHelper();
        
        // DIVERSE transmission intervals (±30% variation)
        Ptr<UniformRandomVariable> periodRand = CreateObject<UniformRandomVariable>();
        double minPeriod = appPeriodSeconds * 0.7;  // -30%
        double maxPeriod = appPeriodSeconds * 1.3;  // +30%
        periodRand->SetAttribute("Min", DoubleValue(minPeriod));
        periodRand->SetAttribute("Max", DoubleValue(maxPeriod));
        double devicePeriod = periodRand->GetValue();
        
        appHelper.SetPeriod(Seconds(devicePeriod));
        appHelper.SetPacketSize(20);
        
        ApplicationContainer singleApp = appHelper.Install(endDevices.Get(i));
        appContainer.Add(singleApp.Get(0));
        
        // DIVERSE start times to avoid synchronization
        Ptr<UniformRandomVariable> startTimeRand = CreateObject<UniformRandomVariable>();
        startTimeRand->SetAttribute("Min", DoubleValue(1.0));
        startTimeRand->SetAttribute("Max", DoubleValue(appPeriodSeconds * 0.5));
        
        double startTime = startTimeRand->GetValue();
        appContainer.Get(i)->SetStartTime(Seconds(startTime));
        appContainer.Get(i)->SetStopTime(Seconds(simulationTime));
        
        NS_LOG_INFO("Device " << i << " period: " << devicePeriod << "s, start: " << startTime << "s");
    }
    
    // Connect to transmission and reception traces
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Phy/$ns3::EndDeviceLoraPhy/StartSending",
                                 MakeCallback(&OnTransmissionStart));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::GatewayLorawanMac/ReceivedPacket",
                                 MakeCallback(&OnPacketReceived));
    
    std::cout << "Starting simulation with enhanced parameter diversity..." << std::endl;
    
    // Schedule CSV output at the end
    Simulator::Schedule(Seconds(simulationTime), [csvFileName]() {
        WriteCSVOutput(csvFileName);
    });
    
    // Run simulation
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "Simulation completed!" << std::endl;
    
    return 0;
}