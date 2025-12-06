#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lora-channel.h"
#include <vector>
#include <map>

namespace ns3 {
namespace lorawan {

/**
 * Urban obstacle structure for environment modeling
 */
struct UrbanObstacle {
    Vector position;
    double width;
    double height;
    double attenuationDb;
};

/**
 * RSSI sampling results
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

/**
 * Environment manager class
 */
class EnvironmentManager {
private:
    std::vector<UrbanObstacle> m_urbanObstacles;
    std::map<uint32_t, double> m_nodeAntennaGains;
    std::map<uint32_t, double> m_nodeNoiseFigures;
    std::map<uint32_t, double> m_nodeHardwareVariance;
    NodeContainer m_wifiInterferers;
    bool m_enableEnvironmentalModeling;
    
    std::map<uint32_t, bool> m_activeTransmitters;
    std::map<uint32_t, Vector> m_nodePositions;
    std::map<uint32_t, Time> m_transmissionEndTimes;
    std::map<uint32_t, double> m_nodeTxPowers;
    
    Ptr<LoraChannel> m_channel;
    
public:
    EnvironmentManager();
    ~EnvironmentManager();
    
    void SetEnvironmentalModeling(bool enable);
    bool IsEnvironmentalModelingEnabled() const;
    
    void SetupUrbanEnvironment(double radius);
    void SetupWiFiInterferers(double radius, uint32_t nInterferers);
    void InitializeHardwareVariability(uint32_t nDevices);
    
    // Load obstacles from external list
    void AddObstacle(const Vector& position, double width, double height, double attenuationDb);
    void ClearObstacles();
    uint32_t GetObstacleCount() const;
    
    void SetChannel(Ptr<LoraChannel> channel);
    Ptr<LoraChannel> GetChannel() const;
    void SetNodePosition(uint32_t nodeId, const Vector& position);
    void SetNodeTxPower(uint32_t nodeId, double txPower);
    Vector GetNodePosition(uint32_t nodeId) const;
    double GetNodeTxPower(uint32_t nodeId) const;
    
    void SetTransmitterActive(uint32_t nodeId, bool active);
    void SetTransmissionEndTime(uint32_t nodeId, Time endTime);
    bool IsTransmitterActive(uint32_t nodeId) const;
    Time GetTransmissionEndTime(uint32_t nodeId) const;
    
    RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime);
    
    void Clear();
};

} // namespace lorawan
} // namespace ns3

#endif // ENVIRONMENT_H
