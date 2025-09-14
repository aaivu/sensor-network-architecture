/*
 * pre-tx-rssi-example.cc
 * 
 * This example demonstrates LoRaWAN simulation with pre-transmission RSSI sampling
 * and comprehensive CSV logging for machine learning dataset generation.
 * 
 * The simulation:
 * 1. Creates end devices and gateways at specified positions
 * 2. Before each transmission, samples the ambient RSSI at the transmitter location
 * 3. Logs comprehensive packet information including pre-TX RSSI and reception outcomes
 * 4. Outputs data in CSV format suitable for ML training
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/lorawan-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include <fstream>
#include <iomanip>
#include <map>
#include <cmath>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE ("PreTxRssiExample");

// Global variables for logging and tracking
std::ofstream csvFile;
std::map<uint32_t, bool> activeTransmitters; // nodeId -> isTransmitting
std::map<uint32_t, double> nodeTxPowers; // nodeId -> txPower in dBm
std::map<std::pair<uint32_t, uint32_t>, PacketData> pendingPackets; // (nodeId, fcnt) -> packet data

struct PacketData {
    double timestamp;
    uint32_t nodeId;
    std::string devEui;
    uint32_t fcnt;
    double txPowerDbm;
    uint8_t sf;
    uint32_t bw;
    uint8_t cr;
    Vector3D txPos;
    double preTxRssiDbm;
    double ambientNoiseDbm;
    bool packetReceived;
    double gatewayRssiDbm;
    double gatewaySnrDb;
    uint32_t gatewayId;
    bool collisionFlag;
    uint32_t payloadBytes;
    uint32_t channel;
};

// Utility functions for power conversion
double DbmToW(double dbm) {
    return std::pow(10.0, (dbm - 30.0) / 10.0);
}

double WToDbm(double w) {
    if (w <= 0) return -200.0; // Very low power for zero/negative values
    return 10.0 * std::log10(w) + 30.0;
}

// Function to compute pre-TX RSSI at transmitter location
double ComputePreTxRssi(Ptr<Node> txNode, Ptr<LoraChannel> channel, double ambientNoiseDbm) {
    Ptr<MobilityModel> txMobility = txNode->GetObject<MobilityModel>();
    uint32_t txNodeId = txNode->GetId();
    
    // Start with ambient noise power
    double totalPowerW = DbmToW(ambientNoiseDbm);
    
    // Get the propagation loss model from the channel
    Ptr<PropagationLossModel> lossModel = channel->GetPropagationLossModel();
    
    // Iterate through all active transmitters except the current one
    for (auto &activeTx : activeTransmitters) {
        uint32_t otherNodeId = activeTx.first;
        bool isTransmitting = activeTx.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) {
            continue;
        }
        
        // Get the other node and its mobility
        Ptr<Node> otherNode = NodeList::GetNode(otherNodeId);
        Ptr<MobilityModel> otherMobility = otherNode->GetObject<MobilityModel>();
        
        // Get the TX power of the other transmitter
        double otherTxPowerDbm = nodeTxPowers[otherNodeId];
        
        // Calculate received power at our TX location from the other transmitter
        double rxPowerDbm = lossModel->CalcRxPower(otherTxPowerDbm, otherMobility, txMobility);
        double rxPowerW = DbmToW(rxPowerDbm);
        
        totalPowerW += rxPowerW;
    }
    
    return WToDbm(totalPowerW);
}

// Callback for when a packet is sent from an end device
void OnPacketSent(Ptr<Packet const> packet, uint32_t nodeId) {
    NS_LOG_INFO("Packet sent from node " << nodeId << " at " << Simulator::Now().GetSeconds());
    
    // Mark this node as transmitting
    activeTransmitters[nodeId] = true;
    
    // Get node and its properties
    Ptr<Node> node = NodeList::GetNode(nodeId);
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    Vector3D position = mobility->GetPosition();
    
    // Get LoRaWAN device properties
    Ptr<LoraNetDevice> loraNetDevice = node->GetDevice(0)->GetObject<LoraNetDevice>();
    Ptr<ClassAEndDeviceLorawanMac> mac = loraNetDevice->GetMac()->GetObject<ClassAEndDeviceLorawanMac>();
    
    // Extract packet information
    LorawanMacHeader macHdr;
    packet->PeekHeader(macHdr);
    
    PacketData packetData;
    packetData.timestamp = Simulator::Now().GetSeconds();
    packetData.nodeId = nodeId;
    packetData.devEui = "DEV" + std::to_string(nodeId);
    packetData.fcnt = macHdr.GetFCnt();
    packetData.txPowerDbm = nodeTxPowers[nodeId];
    packetData.sf = mac->GetDataRate(); // Simplified - actual SF extraction would be more complex
    packetData.bw = 125000; // 125 kHz default for LoRaWAN
    packetData.cr = 1; // 4/5 coding rate
    packetData.txPos = position;
    packetData.ambientNoiseDbm = -110.0; // Typical noise floor
    packetData.packetReceived = false; // Will be updated in receive callback
    packetData.gatewayRssiDbm = -200.0;
    packetData.gatewaySnrDb = -100.0;
    packetData.gatewayId = 0;
    packetData.collisionFlag = false;
    packetData.payloadBytes = packet->GetSize();
    packetData.channel = 0; // Simplified
    
    // Compute pre-TX RSSI
    Ptr<LoraChannel> channel = loraNetDevice->GetPhy()->GetChannel()->GetObject<LoraChannel>();
    packetData.preTxRssiDbm = ComputePreTxRssi(node, channel, packetData.ambientNoiseDbm);
    
    // Store packet data for later update when received/lost
    std::pair<uint32_t, uint32_t> key(nodeId, packetData.fcnt);
    pendingPackets[key] = packetData;
}

// Callback for when transmission ends
void OnPacketTransmissionEnd(Ptr<Packet const> packet, uint32_t nodeId) {
    NS_LOG_INFO("Packet transmission ended for node " << nodeId);
    
    // Mark this node as not transmitting
    activeTransmitters[nodeId] = false;
}

// Callback for when a packet is received at a gateway
void OnPacketReceived(Ptr<Packet const> packet, uint32_t gwId) {
    NS_LOG_INFO("Packet received at gateway " << gwId << " at " << Simulator::Now().GetSeconds());
    
    // Extract packet information to find corresponding sent packet
    LorawanMacHeader macHdr;
    packet->PeekHeader(macHdr);
    
    // Try to find the corresponding sent packet
    // This is simplified - in reality you'd need to match by device address and frame counter
    for (auto &pending : pendingPackets) {
        if (pending.second.fcnt == macHdr.GetFCnt() && !pending.second.packetReceived) {
            pending.second.packetReceived = true;
            pending.second.gatewayId = gwId;
            
            // Get gateway node and calculate RSSI/SNR
            Ptr<Node> gwNode = NodeList::GetNode(gwId);
            Ptr<LoraNetDevice> gwLoraNetDevice = gwNode->GetDevice(0)->GetObject<LoraNetDevice>();
            
            // Simplified RSSI/SNR calculation - in reality this would come from the PHY layer
            Ptr<Node> endDevice = NodeList::GetNode(pending.second.nodeId);
            Ptr<MobilityModel> endDeviceMobility = endDevice->GetObject<MobilityModel>();
            Ptr<MobilityModel> gwMobility = gwNode->GetObject<MobilityModel>();
            
            Ptr<LoraChannel> channel = gwLoraNetDevice->GetPhy()->GetChannel()->GetObject<LoraChannel>();
            Ptr<PropagationLossModel> lossModel = channel->GetPropagationLossModel();
            
            double rxPowerDbm = lossModel->CalcRxPower(pending.second.txPowerDbm, endDeviceMobility, gwMobility);
            pending.second.gatewayRssiDbm = rxPowerDbm;
            pending.second.gatewaySnrDb = rxPowerDbm - pending.second.ambientNoiseDbm; // Simplified SNR
            
            break;
        }
    }
}

// Function to write CSV header
void WriteCsvHeader() {
    csvFile << "timestamp_s,devEui,nodeId,fcnt,tx_power_dBm,sf,bw,cr,tx_x,tx_y,tx_z,"
            << "pre_tx_rssi_dBm,ambient_noise_dBm,expected_gateway_rssi_dBm,gateway_snr_dB,"
            << "gateway_id,packet_received,collision_flag,payload_bytes,channel\n";
}

// Function to write packet data to CSV
void WritePacketToCsv(const PacketData &data) {
    csvFile << std::fixed << std::setprecision(6)
            << data.timestamp << ","
            << data.devEui << ","
            << data.nodeId << ","
            << data.fcnt << ","
            << data.txPowerDbm << ","
            << static_cast<uint32_t>(data.sf) << ","
            << data.bw << ","
            << static_cast<uint32_t>(data.cr) << ","
            << data.txPos.x << ","
            << data.txPos.y << ","
            << data.txPos.z << ","
            << data.preTxRssiDbm << ","
            << data.ambientNoiseDbm << ","
            << data.gatewayRssiDbm << ","
            << data.gatewaySnrDb << ","
            << data.gatewayId << ","
            << (data.packetReceived ? 1 : 0) << ","
            << (data.collisionFlag ? 1 : 0) << ","
            << data.payloadBytes << ","
            << data.channel << "\n";
}

// Function to finalize and write all pending packets
void FinalizePendingPackets() {
    for (const auto &pending : pendingPackets) {
        WritePacketToCsv(pending.second);
    }
    csvFile.close();
}

int main(int argc, char *argv[]) {
    // Simulation parameters
    uint32_t nDevices = 10;
    uint32_t nGateways = 1;
    double simulationTime = 300.0; // seconds
    double appPeriodSeconds = 30.0; // packet transmission period
    std::string csvFileName = "lorawan_rssi_dataset.csv";
    double radius = 1000.0; // meters
    
    // Parse command line arguments
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("nGateways", "Number of gateways", nGateways);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Application packet period in seconds", appPeriodSeconds);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.AddValue("radius", "Simulation area radius in meters", radius);
    cmd.Parse(argc, argv);
    
    // Set up logging
    LogComponentEnable("PreTxRssiExample", LOG_LEVEL_INFO);
    
    // Open CSV file
    csvFile.open(csvFileName);
    WriteCsvHeader();
    
    // Create nodes
    NodeContainer endDevices;
    endDevices.Create(nDevices);
    
    NodeContainer gateways;
    gateways.Create(nGateways);
    
    // Set up mobility - random positions within radius
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    // Gateway at center
    for (uint32_t i = 0; i < nGateways; ++i) {
        allocator->Add(Vector(0.0, 0.0, 15.0)); // 15m height for gateway
    }
    
    // End devices randomly distributed
    Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable>();
    x->SetAttribute("Min", DoubleValue(-radius));
    x->SetAttribute("Max", DoubleValue(radius));
    
    Ptr<UniformRandomVariable> y = CreateObject<UniformRandomVariable>();
    y->SetAttribute("Min", DoubleValue(-radius));
    y->SetAttribute("Max", DoubleValue(radius));
    
    for (uint32_t i = 0; i < nDevices; ++i) {
        allocator->Add(Vector(x->GetValue(), y->GetValue(), 1.5)); // 1.5m height for devices
    }
    
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(allocator);
    mobility.Install(endDevices);
    mobility.Install(gateways);
    
    // Create the LoRa channel with realistic propagation model
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetPathLossExponent(3.76);
    loss->SetReference(1, 7.7);
    
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    
    Ptr<LoraChannel> channel = CreateObject<LoraChannel>(loss, delay);
    
    // Set up the physical layer
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(channel);
    
    // Set up MAC layer
    LorawanMacHelper macHelper = LorawanMacHelper();
    
    // Create the LoRaWAN helper
    LorawanHelper helper = LorawanHelper();
    
    // Install LoRaWAN stack on devices
    NetDeviceContainer endDevicesNetDevices = helper.Install(phyHelper, macHelper, endDevices);
    NetDeviceContainer gatewayNetDevices = helper.Install(phyHelper, macHelper, gateways);
    
    // Set device types
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    helper.Install(phyHelper, macHelper, endDevices);
    
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    helper.Install(phyHelper, macHelper, gateways);
    
    // Set TX powers for devices and store them
    for (uint32_t i = 0; i < nDevices; ++i) {
        Ptr<Node> node = endDevices.Get(i);
        uint32_t nodeId = node->GetId();
        
        // Randomize TX power between 2 and 14 dBm
        Ptr<UniformRandomVariable> txPowerRand = CreateObject<UniformRandomVariable>();
        txPowerRand->SetAttribute("Min", DoubleValue(2.0));
        txPowerRand->SetAttribute("Max", DoubleValue(14.0));
        double txPower = txPowerRand->GetValue();
        
        nodeTxPowers[nodeId] = txPower;
        activeTransmitters[nodeId] = false;
        
        Ptr<LoraNetDevice> loraNetDevice = node->GetDevice(0)->GetObject<LoraNetDevice>();
        Ptr<ClassAEndDeviceLorawanMac> mac = loraNetDevice->GetMac()->GetObject<ClassAEndDeviceLorawanMac>();
        mac->SetTxPower(txPower);
    }
    
    // Set up applications
    OneShotSenderHelper appHelper = OneShotSenderHelper();
    appHelper.SetSendTime(Seconds(1));
    ApplicationContainer appContainer = appHelper.Install(endDevices);
    
    // Set up periodic transmissions
    for (uint32_t i = 0; i < nDevices; ++i) {
        Ptr<UniformRandomVariable> rv = CreateObject<UniformRandomVariable>();
        rv->SetAttribute("Min", DoubleValue(1.0));
        rv->SetAttribute("Max", DoubleValue(appPeriodSeconds));
        
        for (double time = rv->GetValue(); time < simulationTime; time += appPeriodSeconds) {
            Simulator::Schedule(Seconds(time), &OnPacketSent, 
                              Create<Packet>(20), endDevices.Get(i)->GetId());
            // Schedule transmission end callback
            Simulator::Schedule(Seconds(time + 0.1), &OnPacketTransmissionEnd,
                              Create<Packet>(20), endDevices.Get(i)->GetId());
        }
    }
    
    // Connect to receive traces
    for (uint32_t i = 0; i < nGateways; ++i) {
        Ptr<Node> gateway = gateways.Get(i);
        Ptr<LoraNetDevice> netDevice = gateway->GetDevice(0)->GetObject<LoraNetDevice>();
        Ptr<LorawanMac> mac = netDevice->GetMac();
        mac->TraceConnectWithoutContext("ReceivedPacket",
                                      MakeBoundCallback(&OnPacketReceived, gateway->GetId()));
    }
    
    // Schedule simulation end callback to finalize CSV
    Simulator::Schedule(Seconds(simulationTime), &FinalizePendingPackets);
    
    std::cout << "Starting LoRaWAN simulation with " << nDevices << " devices and " 
              << nGateways << " gateways" << std::endl;
    std::cout << "Simulation time: " << simulationTime << " seconds" << std::endl;
    std::cout << "Output file: " << csvFileName << std::endl;
    
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "Simulation completed. Check " << csvFileName << " for results." << std::endl;
    
    return 0;
}
