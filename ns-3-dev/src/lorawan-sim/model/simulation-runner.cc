#include "simulation-runner.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/mobility-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/lora-phy-helper.h"
#include "ns3/lorawan-mac-helper.h"
#include "ns3/periodic-sender-helper.h"
#include "ns3/end-device-lora-phy.h"
#include "ns3/end-device-lorawan-mac.h"
#include "ns3/gateway-lora-phy.h"
#include "ns3/gateway-lorawan-mac.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/network-server-helper.h"
#include "ns3/forwarder-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/string.h"
#include <iostream>
#include <iomanip>

namespace ns3 {
namespace lorawan {

NS_LOG_COMPONENT_DEFINE("SimulationRunner");

// Static instance pointer for callbacks
SimulationRunner* SimulationRunner::s_instance = nullptr;

SimulationRunner::SimulationRunner(uint32_t nDevices, double simulationTime, double appPeriodSeconds,
                                  double radius, const std::string& csvFileName, ADRMethod adrMethod,
                                  uint32_t nWifiInterferers, bool enableEnvironmentalModeling)
    : m_nDevices(nDevices), m_simulationTime(simulationTime), m_appPeriodSeconds(appPeriodSeconds),
      m_radius(radius), m_csvFileName(csvFileName), m_adrMethod(adrMethod),
      m_nWifiInterferers(nWifiInterferers) {
    
    s_instance = this;
    m_environment.SetEnvironmentalModeling(enableEnvironmentalModeling);
}

SimulationRunner::~SimulationRunner() {
    s_instance = nullptr;
}

void SimulationRunner::Clear() {
    m_environment.Clear();
    m_recorder.Clear();
    m_endDevicesNetDevices.clear();
    m_ddqnAgents.clear();
    m_nodeCurrentTxPowers.clear();
    m_nodeCurrentDataRates.clear();
    m_nodeIdToDeviceIndex.clear();
    m_currentSF.clear();
    m_currentTP.clear();
}

void SimulationRunner::SetupNodes() {
    // Set fixed seed for reproducible results
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(1);
    
    // Setup environment
    if (m_environment.IsEnvironmentalModelingEnabled()) {
        m_environment.SetupUrbanEnvironment(m_radius);
        m_environment.SetupWiFiInterferers(m_radius, m_nWifiInterferers);
        m_environment.InitializeHardwareVariability(m_nDevices);
    }
    
    // Create nodes
    m_endDevices.Create(m_nDevices);
    m_gateways.Create(1);
    
    NodeContainer allNodes = NodeContainer(m_endDevices, m_gateways);
    
    // Setup mobility
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    
    // Gateway position
    allocator->Add(Vector(0, 0, 15));
    m_environment.SetNodePosition(m_gateways.Get(0)->GetId(), Vector(0, 0, 15));
    
    // End device positions
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-m_radius));
    xRand->SetAttribute("Max", DoubleValue(m_radius));
    xRand->SetStream(4001);
    
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-m_radius));
    yRand->SetAttribute("Max", DoubleValue(m_radius));
    yRand->SetStream(4002);
    
    Ptr<UniformRandomVariable> powerRand = CreateObject<UniformRandomVariable>();
    powerRand->SetAttribute("Min", DoubleValue(8.0));
    powerRand->SetAttribute("Max", DoubleValue(16.0));
    powerRand->SetStream(4003);
    
    for (uint32_t i = 0; i < m_nDevices; i++) {
        uint32_t nodeId = m_endDevices.Get(i)->GetId();
        Vector pos(xRand->GetValue(), yRand->GetValue(), 1.5);
        allocator->Add(pos);
        m_environment.SetNodePosition(nodeId, pos);
        m_environment.SetNodeTxPower(nodeId, powerRand->GetValue());
    }
    
    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);
}

void SimulationRunner::SetupChannel() {
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetAttribute("Exponent", DoubleValue(3.76));
    loss->SetAttribute("ReferenceLoss", DoubleValue(35.41));
    loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    
    if (m_environment.IsEnvironmentalModelingEnabled()) {
        Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
        shadowing->SetAttribute("Variable", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=64]"));
        loss->SetNext(shadowing);
    }
    
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    Ptr<LoraChannel> channel = CreateObject<LoraChannel>(loss, delay);
    m_environment.SetChannel(channel);
}

void SimulationRunner::SetupDevices() {
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(m_environment.GetChannel());
    
    LorawanMacHelper macHelper = LorawanMacHelper();
    LoraHelper helper = LoraHelper();
    
    // End devices
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    NetDeviceContainer endDevicesNetDevs = helper.Install(phyHelper, macHelper, m_endDevices);
    
    for (uint32_t i = 0; i < endDevicesNetDevs.GetN(); ++i) {
        m_endDevicesNetDevices.push_back(endDevicesNetDevs.Get(i)->GetObject<LoraNetDevice>());
        uint32_t nodeId = m_endDevices.Get(i)->GetId();
        m_currentSF[nodeId] = 7;
        m_currentTP[nodeId] = m_environment.GetNodeTxPower(nodeId);
        m_nodeIdToDeviceIndex[nodeId] = i;
        
        std::cout << "📋 Mapped Node ID " << nodeId << " → Device Index " << i << std::endl;
    }
    
    // Gateways
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    helper.Install(phyHelper, macHelper, m_gateways);
}

void SimulationRunner::SetupNetworkServer() {
    if (m_adrMethod == ADRMethod::ON) {
        NodeContainer networkServer;
        networkServer.Create(1);
        
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
        p2p.SetChannelAttribute("Delay", StringValue("2ms"));
        
        P2PGwRegistration_t gwRegistration;
        for (auto gw = m_gateways.Begin(); gw != m_gateways.End(); ++gw) {
            auto container = p2p.Install(networkServer.Get(0), *gw);
            auto serverP2PNetDev = DynamicCast<PointToPointNetDevice>(container.Get(0));
            gwRegistration.emplace_back(serverP2PNetDev, *gw);
        }
        
        NetworkServerHelper networkServerHelper;
        networkServerHelper.EnableAdr(true);
        networkServerHelper.SetAdr("ns3::AdrComponent");
        networkServerHelper.SetGatewaysP2P(gwRegistration);
        networkServerHelper.SetEndDevices(m_endDevices);
        networkServerHelper.Install(networkServer.Get(0));
        
        for (uint32_t i = 0; i < m_endDevicesNetDevices.size(); ++i) {
            Ptr<EndDeviceLorawanMac> endDeviceMac = DynamicCast<EndDeviceLorawanMac>(
                m_endDevicesNetDevices[i]->GetMac());
            if (endDeviceMac) {
                endDeviceMac->SetUplinkAdrBit(true);
                uint32_t nodeId = m_endDevices.Get(i)->GetId();
                std::cout << "ADR uplink bit enabled for device " << nodeId << std::endl;
            }
        }
        
        Config::Connect("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::EndDeviceLorawanMac/TxPower",
                       MakeCallback(&SimulationRunner::OnTxPowerChange));
        Config::Connect("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::EndDeviceLorawanMac/DataRate",
                       MakeCallback(&SimulationRunner::OnDataRateChange));
        
        ForwarderHelper forwarderHelper;
        forwarderHelper.Install(m_gateways);
        
        std::cout << "✅ Classical ADR enabled via ns-3 AdrComponent" << std::endl;
    } else if (m_adrMethod == ADRMethod::DDQN) {
        for (uint32_t i = 0; i < m_nDevices; ++i) {
            uint32_t nodeId = m_endDevices.Get(i)->GetId();
            m_ddqnAgents[nodeId] = std::make_unique<DDQNPERADRAgent>(
                0.005, 0.95, 0.9995, 0.2, 0.95, 100, 1.0, 0.1);
        }
        std::cout << "✅ DDQN-PER ADR enabled" << std::endl;
    } else {
        std::cout << "ADR disabled" << std::endl;
    }
}

void SimulationRunner::SetupApplications() {
    for (uint32_t i = 0; i < m_nDevices; i++) {
        PeriodicSenderHelper appHelper = PeriodicSenderHelper();
        appHelper.SetPeriod(Seconds(m_appPeriodSeconds));
        
        ApplicationContainer app = appHelper.Install(m_endDevices.Get(i));
        
        Ptr<UniformRandomVariable> randomStart = CreateObject<UniformRandomVariable>();
        randomStart->SetAttribute("Min", DoubleValue(0.0));
        randomStart->SetAttribute("Max", DoubleValue(m_appPeriodSeconds));
        randomStart->SetStream(5000 + i);
        app.Start(Seconds(randomStart->GetValue()));
    }
}

void SimulationRunner::SetupCallbacks() {
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Phy/$ns3::EndDeviceLoraPhy/StartSending",
        MakeCallback(&SimulationRunner::OnTransmissionStart, this));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::LoraNetDevice/Mac/$ns3::GatewayLorawanMac/ReceivedPacket",
        MakeCallback(&SimulationRunner::OnPacketReceived, this));
}

void SimulationRunner::ApplyADRDecision(uint32_t nodeId) {
    if (nodeId >= m_endDevicesNetDevices.size()) return;
    
    Ptr<EndDeviceLoraPhy> edPhy = DynamicCast<EndDeviceLoraPhy>(
        m_endDevicesNetDevices[nodeId]->GetPhy());
    Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
        m_endDevicesNetDevices[nodeId]->GetMac());
    
    if (!edPhy || !edMac) return;
    
    if (m_adrMethod == ADRMethod::DDQN) {
        auto it = m_ddqnAgents.find(nodeId);
        if (it != m_ddqnAgents.end()) {
            DeviceMetrics& metrics = m_recorder.GetDeviceMetrics(nodeId);
            if (metrics.packetsSent > 0) {
                std::vector<double> state = {
                    metrics.lastPDR,
                    metrics.lastEnergyPerPacket,
                    (double)edPhy->GetSpreadingFactor(),
                    edMac->GetTransmissionPowerDbm(),
                    0.0, 0.0
                };
                
                int action = it->second->selectAction(state);
                
                uint8_t newSF = 7 + (action % 6);
                double newTP = 5.0 + (action / 6) * 2.0;
                
                if (newSF != edPhy->GetSpreadingFactor() || 
                    std::abs(newTP - edMac->GetTransmissionPowerDbm()) > 0.1) {
                    
                    edPhy->SetSpreadingFactor(newSF);
                    edMac->SetTransmissionPowerDbm(newTP);
                    
                    m_currentSF[nodeId] = newSF;
                    m_currentTP[nodeId] = newTP;
                    
                    std::cout << "🤖 DDQN ADR: Device " << nodeId 
                              << " SF=" << (int)newSF << ", TP=" << newTP << " dBm" << std::endl;
                }
            }
        }
    }
}

void SimulationRunner::OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId) {
    uint32_t nodeId = systemId;
    Time currentTime = Simulator::Now();
    
    ApplyADRDecision(nodeId);
    RecordTransmissionParameters(packet, nodeId, currentTime);
}

void SimulationRunner::RecordTransmissionParameters(Ptr<const Packet> packet, uint32_t nodeId, Time transmissionTime) {
    RssiSample rssiSample = m_environment.SamplePreTxRssi(nodeId, transmissionTime);
    m_environment.SetTransmitterActive(nodeId, true);
    
    uint8_t actualSF;
    double actualTxPower;
    
    if (m_adrMethod == ADRMethod::ON) {
        actualSF = 7;
        actualTxPower = 14.0;
        
        if (m_nodeIdToDeviceIndex.find(nodeId) != m_nodeIdToDeviceIndex.end()) {
            if (m_nodeCurrentTxPowers.find(nodeId) != m_nodeCurrentTxPowers.end()) {
                actualTxPower = m_nodeCurrentTxPowers[nodeId];
            } else {
                actualTxPower = m_environment.GetNodeTxPower(nodeId);
            }
            
            if (m_nodeCurrentDataRates.find(nodeId) != m_nodeCurrentDataRates.end()) {
                uint8_t dataRate = m_nodeCurrentDataRates[nodeId];
                switch (dataRate) {
                    case 0: actualSF = 12; break;
                    case 1: actualSF = 11; break;
                    case 2: actualSF = 10; break;
                    case 3: actualSF = 9; break;
                    case 4: actualSF = 8; break;
                    case 5: actualSF = 7; break;
                    default: actualSF = 7; break;
                }
            }
            
            m_currentSF[nodeId] = actualSF;
            m_currentTP[nodeId] = actualTxPower;
        }
    } else {
        actualSF = m_currentSF.count(nodeId) ? m_currentSF[nodeId] : 7;
        actualTxPower = m_currentTP.count(nodeId) ? m_currentTP[nodeId] : m_environment.GetNodeTxPower(nodeId);
    }
    
    LoraTxParameters txParams;
    txParams.sf = actualSF;
    txParams.headerDisabled = false;
    txParams.codingRate = 1;
    txParams.bandwidthHz = 125000;
    txParams.nPreamble = 8;
    txParams.crcEnabled = true;
    txParams.lowDataRateOptimizationEnabled = (actualSF > 10);
    
    Time txDuration = LoraPhy::GetOnAirTime(packet->Copy(), txParams);
    m_environment.SetTransmissionEndTime(nodeId, transmissionTime + txDuration);
    
    PacketRecord record;
    record.timestamp = transmissionTime.GetSeconds();
    record.nodeId = nodeId;
    record.devEui = "DEV" + std::to_string(nodeId);
    record.fcnt = m_recorder.GetNextFrameCounter(nodeId);
    record.txPowerDbm = actualTxPower;
    record.sf = actualSF;
    record.bw = 125000;
    record.txPos = m_environment.GetNodePosition(nodeId);
    record.preTxRssiDbm = rssiSample.preTxRssiDbm;
    record.ambientNoiseDbm = rssiSample.ambientNoiseDbm;
    record.packetReceived = false;
    record.gatewayRssiDbm = -999.0;
    record.gatewaySnrDb = -999.0;
    record.payloadBytes = packet->GetSize();
    record.collisionFlag = false;
    record.energyConsumed = CalculateEnergyConsumption(actualSF, actualTxPower, record.payloadBytes);
    
    m_recorder.AddPacketRecord(record);
    
    Simulator::Schedule(txDuration, [this, nodeId]() {
        m_environment.SetTransmitterActive(nodeId, false);
    });
    
    Time packetTimeout = txDuration + Seconds(2.0);
    Simulator::Schedule(packetTimeout, &SimulationRunner::CheckPacketLoss, this, nodeId, transmissionTime.GetSeconds());
}

void SimulationRunner::CheckPacketLoss(uint32_t nodeId, double transmissionTime) {
    const auto& packets = m_recorder.GetCompletedPackets();
    for (const auto& record : packets) {
        if (record.nodeId == nodeId && 
            std::abs(record.timestamp - transmissionTime) < 0.1 && 
            !record.packetReceived) {
            
            m_recorder.UpdateDeviceMetrics(nodeId, false, record.energyConsumed);
            
            if (m_adrMethod == ADRMethod::DDQN) {
                UpdateDDQNADR(nodeId, -20.0, false);
            }
            
            NS_LOG_INFO("PACKET LOSS - Node " << nodeId << " at " << transmissionTime << "s");
            break;
        }
    }
}

void SimulationRunner::OnPacketReceived(Ptr<const Packet> packet) {
    Time rxTime = Simulator::Now();
    
    const auto& packets = m_recorder.GetCompletedPackets();
    for (auto it = packets.rbegin(); it != packets.rend(); ++it) {
        if (!it->packetReceived && rxTime.GetSeconds() >= it->timestamp &&
            std::abs(it->timestamp - rxTime.GetSeconds()) < 3.0) {
            
            Vector gatewayPos(0, 0, 15);
            double distance = CalculateDistance(it->txPos, gatewayPos);
            double pathLossDb = 20 * std::log10(distance) + 20 * std::log10(868e6) - 147.55;
            double gatewayRssi = it->txPowerDbm - pathLossDb;
            double gatewaySnr = gatewayRssi - it->ambientNoiseDbm;
            
            m_recorder.UpdatePacketReceived(it->nodeId, it->timestamp, gatewayRssi, gatewaySnr);
            m_recorder.UpdateDeviceMetrics(it->nodeId, true, it->energyConsumed);
            
            if (m_adrMethod == ADRMethod::DDQN) {
                UpdateDDQNADR(it->nodeId, gatewaySnr, true);
            }
            
            NS_LOG_INFO("RX SUCCESS - Node " << it->nodeId << " at " << rxTime.GetSeconds() 
                       << "s, SNR: " << gatewaySnr << " dB");
            return;
        }
    }
}

void SimulationRunner::UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess) {
    if (m_ddqnAgents.find(nodeId) == m_ddqnAgents.end()) return;
    
    DeviceMetrics& metrics = m_recorder.GetDeviceMetrics(nodeId);
    
    double currentPDR = (double)metrics.packetsReceived / std::max(1u, metrics.packetsSent);
    uint8_t currentSFVal = m_currentSF[nodeId];
    double currentTPVal = m_currentTP[nodeId];
    
    double energyThisPacket = CalculateEnergyConsumption(currentSFVal, currentTPVal, 20);
    double avgEnergyPerPacket = (metrics.totalEnergyConsumed + energyThisPacket) / std::max(1u, metrics.packetsSent);
    
    Vector nodePos = m_environment.GetNodePosition(nodeId);
    Vector gatewayPos(0, 0, 15);
    double distance = CalculateDistance(nodePos, gatewayPos);
    
    std::vector<double> currentState = {
        std::max(0.0, std::min(1.0, (snr + 25.0) / 50.0)),
        (currentSFVal - 7.0) / 5.0,
        std::min(1.0, std::max(0.0, currentPDR)),
        (currentTPVal - 2.0) / 15.0,
        std::min(1.0, avgEnergyPerPacket / 1.0),
        std::min(1.0, distance / 4000.0)
    };
    
    double reward = 0.0;
    if (metrics.packetsSent >= 20) {
        reward = m_ddqnAgents[nodeId]->calculateAdvancedReward(
            nodeId, currentPDR, avgEnergyPerPacket, snr, packetSuccess, 
            currentSFVal, currentTPVal, distance, metrics.packetsSent);
    }
    
    static std::map<uint32_t, std::vector<double>> previousStates;
    static std::map<uint32_t, int> previousActions;
    
    if (previousStates.find(nodeId) != previousStates.end() && metrics.packetsSent >= 2) {
        m_ddqnAgents[nodeId]->addExperience(previousStates[nodeId], previousActions[nodeId], 
                                           reward, currentState, false);
        m_ddqnAgents[nodeId]->train();
    }
    
    int action = m_ddqnAgents[nodeId]->selectAction(currentState);
    auto [newSF, newTP] = m_ddqnAgents[nodeId]->getParameters(action);
    
    if (newSF != -1) {
        m_currentSF[nodeId] = newSF;
        uint8_t newDataRate = 12 - newSF;
        
        Simulator::Schedule(Seconds(0.1), [this, nodeId, newDataRate, newSF]() {
            if (nodeId < m_endDevicesNetDevices.size()) {
                Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                    m_endDevicesNetDevices[nodeId]->GetMac());
                if (edMac) {
                    edMac->SetDataRate(newDataRate);
                }
            }
        });
    }
    
    if (newTP != -1) {
        m_currentTP[nodeId] = newTP;
        m_environment.SetNodeTxPower(nodeId, newTP);
        
        if (nodeId < m_endDevicesNetDevices.size()) {
            Ptr<EndDeviceLorawanMac> edMac = DynamicCast<EndDeviceLorawanMac>(
                m_endDevicesNetDevices[nodeId]->GetMac());
            if (edMac) {
                edMac->SetTransmissionPowerDbm(newTP);
            }
        }
    }
    
    previousStates[nodeId] = currentState;
    previousActions[nodeId] = action;
}

void SimulationRunner::OnTxPowerChange(std::string context, double oldValue, double newValue) {
    if (!s_instance) return;
    
    size_t nodeStart = context.find("/NodeList/") + 10;
    size_t nodeEnd = context.find("/", nodeStart);
    uint32_t nodeId = std::stoi(context.substr(nodeStart, nodeEnd - nodeStart));
    
    s_instance->m_nodeCurrentTxPowers[nodeId] = newValue;
    
    std::cout << "ADR TX POWER CHANGE - Node " << nodeId << ": " << oldValue << " -> " << newValue << " dBm" << std::endl;
}

void SimulationRunner::OnDataRateChange(std::string context, uint8_t oldValue, uint8_t newValue) {
    if (!s_instance) return;
    
    size_t nodeStart = context.find("/NodeList/") + 10;
    size_t nodeEnd = context.find("/", nodeStart);
    uint32_t nodeId = std::stoi(context.substr(nodeStart, nodeEnd - nodeStart));
    
    s_instance->m_nodeCurrentDataRates[nodeId] = newValue;
    
    std::cout << "ADR DATA RATE CHANGE - Node " << nodeId << ": " << (int)oldValue << " -> " << (int)newValue << std::endl;
}

void SimulationRunner::Run() {
    Clear();
    
    SetupNodes();
    SetupChannel();
    SetupDevices();
    SetupNetworkServer();
    SetupApplications();
    SetupCallbacks();
    
    std::cout << "=== Simulation Run Details ===" << std::endl;
    std::cout << "ADR Mode: " << (m_adrMethod == ADRMethod::ON ? "CLASSICAL" : 
                                  (m_adrMethod == ADRMethod::DDQN ? "DDQN-PER" : "OFF")) << std::endl;
    std::cout << "Output file: " << m_csvFileName << std::endl;
    
    Simulator::Schedule(Seconds(m_simulationTime), [this]() {
        m_recorder.WriteCSVOutput(m_csvFileName);
    });
    
    Simulator::Stop(Seconds(m_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "Simulation completed!" << std::endl;
}

} // namespace lorawan
} // namespace ns3
