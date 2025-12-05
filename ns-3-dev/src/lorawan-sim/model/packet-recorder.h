#ifndef PACKET_RECORDER_H
#define PACKET_RECORDER_H

#include "ns3/core-module.h"
#include "ns3/vector.h"
#include <vector>
#include <map>
#include <string>

namespace ns3 {
namespace lorawan {

/**
 * Packet record structure for tracking transmission and reception
 */
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

/**
 * Device performance metrics for ADR
 */
struct DeviceMetrics {
    uint32_t packetsSent = 0;
    uint32_t packetsReceived = 0;
    double totalEnergyConsumed = 0.0;
    double lastPDR = 0.0;
    double lastEnergyPerPacket = 0.0;
};

/**
 * Packet recorder class for tracking all transmissions
 */
class PacketRecorder {
private:
    std::vector<PacketRecord> m_completedPackets;
    std::map<uint32_t, uint32_t> m_deviceFrameCounters;
    std::map<uint32_t, DeviceMetrics> m_deviceMetrics;
    
public:
    PacketRecorder();
    ~PacketRecorder();
    
    void AddPacketRecord(const PacketRecord& record);
    void UpdatePacketReceived(uint32_t nodeId, double timestamp, double gatewayRssi, double gatewaySnr);
    
    uint32_t GetNextFrameCounter(uint32_t nodeId);
    
    DeviceMetrics& GetDeviceMetrics(uint32_t nodeId);
    void UpdateDeviceMetrics(uint32_t nodeId, bool packetSuccess, double energyConsumed);
    
    const std::vector<PacketRecord>& GetCompletedPackets() const;
    void WriteCSVOutput(const std::string& filename) const;
    
    void Clear();
};

/**
 * Calculate energy consumption for a packet transmission
 */
double CalculateEnergyConsumption(int sf, double txPowerDbm, uint32_t payloadBytes);

} // namespace lorawan
} // namespace ns3

#endif // PACKET_RECORDER_H
