#include "environment.h"
#include "ns3/log.h"
#include "ns3/lora-utils.h"
#include "ns3/constant-position-mobility-model.h"
#include <iostream>
#include <cmath>

namespace ns3 {
namespace lorawan {

NS_LOG_COMPONENT_DEFINE("EnvironmentManager");

EnvironmentManager::EnvironmentManager() 
    : m_enableEnvironmentalModeling(true) {
}

EnvironmentManager::~EnvironmentManager() {
}

void EnvironmentManager::SetEnvironmentalModeling(bool enable) {
    m_enableEnvironmentalModeling = enable;
}

bool EnvironmentManager::IsEnvironmentalModelingEnabled() const {
    return m_enableEnvironmentalModeling;
}

void EnvironmentManager::SetupUrbanEnvironment(double radius) {
    if (!m_enableEnvironmentalModeling) return;
    
    uint32_t obstaclesPerSide = static_cast<uint32_t>(radius / 80.0);
    double obstacleSpacing = (2.0 * radius) / obstaclesPerSide;
    
    Ptr<UniformRandomVariable> obstacleWidthRand = CreateObject<UniformRandomVariable>();
    obstacleWidthRand->SetAttribute("Min", DoubleValue(20.0));
    obstacleWidthRand->SetAttribute("Max", DoubleValue(120.0));
    obstacleWidthRand->SetStream(1001);
    
    Ptr<UniformRandomVariable> obstacleHeightRand = CreateObject<UniformRandomVariable>();
    obstacleHeightRand->SetAttribute("Min", DoubleValue(8.0));
    obstacleHeightRand->SetAttribute("Max", DoubleValue(80.0));
    obstacleHeightRand->SetStream(1002);
    
    Ptr<UniformRandomVariable> attenuationRand = CreateObject<UniformRandomVariable>();
    attenuationRand->SetAttribute("Min", DoubleValue(8.0));
    attenuationRand->SetAttribute("Max", DoubleValue(35.0));
    attenuationRand->SetStream(1003);
    
    Ptr<UniformRandomVariable> placementProbability = CreateObject<UniformRandomVariable>();
    placementProbability->SetAttribute("Min", DoubleValue(0.0));
    placementProbability->SetAttribute("Max", DoubleValue(1.0));
    placementProbability->SetStream(1004);
    
    Ptr<UniformRandomVariable> positionJitter = CreateObject<UniformRandomVariable>();
    positionJitter->SetAttribute("Min", DoubleValue(-15.0));
    positionJitter->SetAttribute("Max", DoubleValue(15.0));
    positionJitter->SetStream(1005);
    
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
            
            m_urbanObstacles.push_back(obstacle);
        }
    }
    
    NS_LOG_INFO("Created " << m_urbanObstacles.size() << " virtual obstacles for urban modeling");
    std::cout << "Urban environment: Created " << m_urbanObstacles.size() << " virtual obstacles" << std::endl;
}

void EnvironmentManager::SetupWiFiInterferers(double radius, uint32_t nInterferers) {
    if (!m_enableEnvironmentalModeling || nInterferers == 0) return;
    
    m_wifiInterferers.Create(nInterferers);
    
    MobilityHelper wifiMobility;
    Ptr<ListPositionAllocator> wifiAllocator = CreateObject<ListPositionAllocator>();
    
    Ptr<UniformRandomVariable> wifiXRand = CreateObject<UniformRandomVariable>();
    wifiXRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiXRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiXRand->SetStream(2001);
    
    Ptr<UniformRandomVariable> wifiYRand = CreateObject<UniformRandomVariable>();
    wifiYRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiYRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiYRand->SetStream(2002);
    
    for (uint32_t i = 0; i < nInterferers; i++) {
        Vector wifiPos(wifiXRand->GetValue(), wifiYRand->GetValue(), 3.0);
        wifiAllocator->Add(wifiPos);
    }
    
    wifiMobility.SetPositionAllocator(wifiAllocator);
    wifiMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    wifiMobility.Install(m_wifiInterferers);
    
    NS_LOG_INFO("Created " << nInterferers << " WiFi interferers (simplified model)");
    std::cout << "Interference modeling: Created " << nInterferers << " WiFi interferers" << std::endl;
}

void EnvironmentManager::InitializeHardwareVariability(uint32_t nDevices) {
    if (!m_enableEnvironmentalModeling) return;
    
    Ptr<NormalRandomVariable> antennaGainVar = CreateObject<NormalRandomVariable>();
    antennaGainVar->SetAttribute("Mean", DoubleValue(0.0));
    antennaGainVar->SetAttribute("Variance", DoubleValue(2.0));
    antennaGainVar->SetStream(3001);
    
    Ptr<NormalRandomVariable> noiseFigureVar = CreateObject<NormalRandomVariable>();
    noiseFigureVar->SetAttribute("Mean", DoubleValue(6.0));
    noiseFigureVar->SetAttribute("Variance", DoubleValue(1.5));
    noiseFigureVar->SetStream(3002);
    
    Ptr<NormalRandomVariable> hardwareVar = CreateObject<NormalRandomVariable>();
    hardwareVar->SetAttribute("Mean", DoubleValue(0.0));
    hardwareVar->SetAttribute("Variance", DoubleValue(1.0));
    hardwareVar->SetStream(3003);
    
    for (uint32_t i = 0; i < nDevices; i++) {
        m_nodeAntennaGains[i] = antennaGainVar->GetValue();
        m_nodeNoiseFigures[i] = noiseFigureVar->GetValue();
        m_nodeHardwareVariance[i] = hardwareVar->GetValue();
        
        Ptr<NormalRandomVariable> txPowerVar = CreateObject<NormalRandomVariable>();
        txPowerVar->SetAttribute("Mean", DoubleValue(0.0));
        txPowerVar->SetAttribute("Variance", DoubleValue(0.5));
        txPowerVar->SetStream(3100 + i);
        
        if (m_nodeTxPowers.find(i) != m_nodeTxPowers.end()) {
            m_nodeTxPowers[i] += txPowerVar->GetValue();
        }
    }
    
    NS_LOG_INFO("Initialized hardware variability for " << nDevices << " devices");
    std::cout << "Hardware modeling: Added manufacturing tolerances for " << nDevices << " devices" << std::endl;
}

void EnvironmentManager::SetChannel(Ptr<LoraChannel> channel) {
    m_channel = channel;
}

Ptr<LoraChannel> EnvironmentManager::GetChannel() const {
    return m_channel;
}

void EnvironmentManager::SetNodePosition(uint32_t nodeId, const Vector& position) {
    m_nodePositions[nodeId] = position;
}

void EnvironmentManager::SetNodeTxPower(uint32_t nodeId, double txPower) {
    m_nodeTxPowers[nodeId] = txPower;
}

Vector EnvironmentManager::GetNodePosition(uint32_t nodeId) const {
    auto it = m_nodePositions.find(nodeId);
    if (it != m_nodePositions.end()) {
        return it->second;
    }
    return Vector(0, 0, 0);
}

double EnvironmentManager::GetNodeTxPower(uint32_t nodeId) const {
    auto it = m_nodeTxPowers.find(nodeId);
    if (it != m_nodeTxPowers.end()) {
        return it->second;
    }
    return 14.0; // Default power
}

void EnvironmentManager::SetTransmitterActive(uint32_t nodeId, bool active) {
    m_activeTransmitters[nodeId] = active;
}

void EnvironmentManager::SetTransmissionEndTime(uint32_t nodeId, Time endTime) {
    m_transmissionEndTimes[nodeId] = endTime;
}

bool EnvironmentManager::IsTransmitterActive(uint32_t nodeId) const {
    auto it = m_activeTransmitters.find(nodeId);
    if (it != m_activeTransmitters.end()) {
        return it->second;
    }
    return false;
}

Time EnvironmentManager::GetTransmissionEndTime(uint32_t nodeId) const {
    auto it = m_transmissionEndTimes.find(nodeId);
    if (it != m_transmissionEndTimes.end()) {
        return it->second;
    }
    return Seconds(0);
}

RssiSample EnvironmentManager::SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    Vector txPos = GetNodePosition(txNodeId);
    
    Ptr<UniformRandomVariable> noiseVar = CreateObject<UniformRandomVariable>();
    
    if (m_enableEnvironmentalModeling) {
        double urbanNoiseIncrease = 5.0;
        double timeVaryingNoise = 2.0 * std::sin(currentTime.GetSeconds() / 60.0);
        
        noiseVar->SetAttribute("Min", DoubleValue(-115.0 + urbanNoiseIncrease));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0 + urbanNoiseIncrease + timeVaryingNoise));
    } else {
        noiseVar->SetAttribute("Min", DoubleValue(-115.0));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0));
    }
    
    double ambientNoise = noiseVar->GetValue();
    
    if (m_enableEnvironmentalModeling && m_nodeNoiseFigures.find(txNodeId) != m_nodeNoiseFigures.end()) {
        ambientNoise += (m_nodeNoiseFigures[txNodeId] - 6.0);
    }
    
    double totalPowerLinear = DbmToW(ambientNoise);
    
    for (const auto& pair : m_activeTransmitters) {
        uint32_t otherNodeId = pair.first;
        bool isTransmitting = pair.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) continue;
        if (currentTime >= GetTransmissionEndTime(otherNodeId)) continue;
        
        Vector interfererPos = GetNodePosition(otherNodeId);
        double interfererTxPower = GetNodeTxPower(otherNodeId);
        
        if (m_enableEnvironmentalModeling) {
            if (m_nodeAntennaGains.find(otherNodeId) != m_nodeAntennaGains.end()) {
                interfererTxPower += m_nodeAntennaGains[otherNodeId];
            }
            if (m_nodeAntennaGains.find(txNodeId) != m_nodeAntennaGains.end()) {
                interfererTxPower += m_nodeAntennaGains[txNodeId];
            }
        }
        
        Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel>();
        txMobility->SetPosition(txPos);
        
        Ptr<ConstantPositionMobilityModel> interfererMobility = CreateObject<ConstantPositionMobilityModel>();
        interfererMobility->SetPosition(interfererPos);
        
        double rxPowerDbm = m_channel->GetRxPower(interfererTxPower, interfererMobility, txMobility);
        
        if (m_enableEnvironmentalModeling && m_nodeHardwareVariance.find(txNodeId) != m_nodeHardwareVariance.end()) {
            rxPowerDbm += m_nodeHardwareVariance[txNodeId];
        }
        
        totalPowerLinear += DbmToW(rxPowerDbm);
    }
    
    if (m_enableEnvironmentalModeling && m_wifiInterferers.GetN() > 0) {
        Ptr<UniformRandomVariable> wifiInterferenceVar = CreateObject<UniformRandomVariable>();
        wifiInterferenceVar->SetAttribute("Min", DoubleValue(-130.0));
        wifiInterferenceVar->SetAttribute("Max", DoubleValue(-110.0));
        
        double wifiTrafficLevel = 0.5 + 0.5 * std::sin(currentTime.GetSeconds() / 30.0);
        double wifiInterferenceDbm = wifiInterferenceVar->GetValue() + 10.0 * std::log10(wifiTrafficLevel);
        
        totalPowerLinear += DbmToW(wifiInterferenceDbm);
    }
    
    if (m_enableEnvironmentalModeling) {
        Ptr<UniformRandomVariable> adjacentChannelVar = CreateObject<UniformRandomVariable>();
        adjacentChannelVar->SetAttribute("Min", DoubleValue(-140.0));
        adjacentChannelVar->SetAttribute("Max", DoubleValue(-120.0));
        
        double adjacentInterferenceDbm = adjacentChannelVar->GetValue();
        totalPowerLinear += DbmToW(adjacentInterferenceDbm);
    }
    
    double preTxRssiDbm = WToDbm(totalPowerLinear);
    
    Ptr<NormalRandomVariable> rssiNoise = CreateObject<NormalRandomVariable>();
    rssiNoise->SetAttribute("Mean", DoubleValue(0.0));
    
    if (m_enableEnvironmentalModeling) {
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

void EnvironmentManager::Clear() {
    m_urbanObstacles.clear();
    m_nodeAntennaGains.clear();
    m_nodeNoiseFigures.clear();
    m_nodeHardwareVariance.clear();
    m_wifiInterferers = NodeContainer();
    m_activeTransmitters.clear();
    m_nodePositions.clear();
    m_transmissionEndTimes.clear();
    m_nodeTxPowers.clear();
}

void EnvironmentManager::AddObstacle(const Vector& position, double width, double height, double attenuationDb) {
    UrbanObstacle obstacle;
    obstacle.position = position;
    obstacle.width = width;
    obstacle.height = height;
    obstacle.attenuationDb = attenuationDb;
    m_urbanObstacles.push_back(obstacle);
}

void EnvironmentManager::ClearObstacles() {
    m_urbanObstacles.clear();
}

uint32_t EnvironmentManager::GetObstacleCount() const {
    return m_urbanObstacles.size();
}

} // namespace lorawan
} // namespace ns3
