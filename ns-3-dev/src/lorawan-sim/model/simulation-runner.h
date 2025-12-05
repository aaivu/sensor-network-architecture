#ifndef SIMULATION_RUNNER_H
#define SIMULATION_RUNNER_H

#include "ns3/core-module.h"
#include "ns3/lora-helper.h"
#include "ns3/node-container.h"
#include "environment.h"
#include "packet-recorder.h"
#include "adr-ddqn.h"
#include <string>
#include <memory>
#include <map>

namespace ns3 {
namespace lorawan {

/**
 * ADR Method enumeration
 */
enum class ADRMethod { OFF, ON, DDQN };

/**
 * Main simulation runner class
 */
class SimulationRunner {
private:
    // Simulation parameters
    uint32_t m_nDevices;
    double m_simulationTime;
    double m_appPeriodSeconds;
    double m_radius;
    std::string m_csvFileName;
    ADRMethod m_adrMethod;
    uint32_t m_nWifiInterferers;
    std::string m_nodePositionsFile;
    std::string m_obstaclesFile;
    
    // Core components
    EnvironmentManager m_environment;
    PacketRecorder m_recorder;
    
    // ns-3 containers
    NodeContainer m_endDevices;
    NodeContainer m_gateways;
    std::vector<Ptr<LoraNetDevice>> m_endDevicesNetDevices;
    
    // DDQN agents
    std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>> m_ddqnAgents;
    
    // ADR tracking
    std::map<uint32_t, double> m_nodeCurrentTxPowers;
    std::map<uint32_t, uint8_t> m_nodeCurrentDataRates;
    std::map<uint32_t, uint32_t> m_nodeIdToDeviceIndex;
    std::map<uint32_t, uint8_t> m_currentSF;
    std::map<uint32_t, double> m_currentTP;
    
    // Callback functions
    void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId);
    void RecordTransmissionParameters(Ptr<const Packet> packet, uint32_t nodeId, Time transmissionTime);
    void OnPacketReceived(Ptr<const Packet> packet);
    void CheckPacketLoss(uint32_t nodeId, double transmissionTime);
    
    // ADR functions
    void ApplyADRDecision(uint32_t nodeId);
    void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
    static void OnTxPowerChange(std::string context, double oldValue, double newValue);
    static void OnDataRateChange(std::string context, uint8_t oldValue, uint8_t newValue);
    
    // Setup functions
    void SetupNodes();
    void SetupChannel();
    void SetupDevices();
    void SetupNetworkServer();
    void SetupApplications();
    void SetupCallbacks();
    
    // Utility
    void Clear();
    
    // Static instance for callbacks
    static SimulationRunner* s_instance;
    
public:
    SimulationRunner(uint32_t nDevices, double simulationTime, double appPeriodSeconds,
                    double radius, const std::string& csvFileName, ADRMethod adrMethod,
                    uint32_t nWifiInterferers, bool enableEnvironmentalModeling);
    ~SimulationRunner();
    
    // Set CSV files for node positions and obstacles
    void SetNodePositionsFile(const std::string& filePath);
    void SetObstaclesFile(const std::string& filePath);
    
    void Run();
    
    // Getters for callback access
    EnvironmentManager& GetEnvironment() { return m_environment; }
    PacketRecorder& GetRecorder() { return m_recorder; }
    std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>>& GetDDQNAgents() { return m_ddqnAgents; }
    std::map<uint32_t, uint8_t>& GetCurrentSF() { return m_currentSF; }
    std::map<uint32_t, double>& GetCurrentTP() { return m_currentTP; }
    std::vector<Ptr<LoraNetDevice>>& GetEndDevicesNetDevices() { return m_endDevicesNetDevices; }
    ADRMethod GetADRMethod() const { return m_adrMethod; }
    std::map<uint32_t, double>& GetNodeCurrentTxPowers() { return m_nodeCurrentTxPowers; }
    std::map<uint32_t, uint8_t>& GetNodeCurrentDataRates() { return m_nodeCurrentDataRates; }
    std::map<uint32_t, uint32_t>& GetNodeIdToDeviceIndex() { return m_nodeIdToDeviceIndex; }
};

} // namespace lorawan
} // namespace ns3

#endif // SIMULATION_RUNNER_H
