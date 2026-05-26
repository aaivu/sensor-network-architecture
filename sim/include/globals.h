// -*- mode: c++; -*-
#pragma once
// Global Variables: simulation state, UrbanObstacle, PacketRecord, DeviceMetrics, forward declarations

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
NodeContainer lteInterferers;
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
enum class ADRMethod { OFF, ON, DDQN, PPO, MARL, MAPPO_GAT, MARL_PPO };
ADRMethod currentADRMethod = ADRMethod::OFF;

// Global references for ADR functionality
std::vector<Ptr<LoraNetDevice>> endDevicesNetDevices;
std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>> ddqnAgents;

// ADR parameter tracking (updated via trace sources) - SAME AS WORKING VERSION
std::map<uint32_t, double> nodeCurrentTxPowers; // nodeId -> current ADR-adjusted TX power
std::map<uint32_t, uint8_t> nodeCurrentDataRates; // nodeId -> current ADR-adjusted data rate
std::map<uint32_t, uint32_t> nodeIdToDeviceIndex; // nodeId -> device index in endDevicesNetDevices

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

// **FUNCTION DECLARATIONS**
void ApplyADRDecision(uint32_t nodeId);
void UpdateDeviceMetrics(uint32_t nodeId, bool packetSuccess, double energyConsumed);
void CheckPacketLoss(uint32_t nodeId, double transmissionTime);
void OnTransmissionStart(Ptr<const Packet> packet, uint32_t systemId);
void RecordTransmissionParameters(Ptr<const Packet> packet, uint32_t nodeId, Time transmissionTime);
void OnPacketReceived(Ptr<const Packet> packet);
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
void UpdateOptimizedDDQN(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void UpdatePPOADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void UpdateMARLADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void UpdateMAPPOGATADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void UpdateMARLPPOADR(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);

// Forward declarations
void UpdateDDQNADR(uint32_t nodeId, double snr, bool packetSuccess);
void UpdateOptimizedDDQN(uint32_t nodeId, double snr, bool packetSuccess, double preTxRssi);
void WriteCSVOutput(const std::string& filename);
void WriteEnvironmentVisualization(const std::string& baseFilename, double radius, uint32_t nDevices);
void SetupLTEMobileInterferers(double radius, uint32_t nInterferers);
