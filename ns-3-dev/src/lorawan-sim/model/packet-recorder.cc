#include "packet-recorder.h"
#include "ns3/lora-phy.h"
#include "ns3/packet.h"
#include "ns3/lora-utils.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>

namespace ns3 {
namespace lorawan {

PacketRecorder::PacketRecorder() 
    : m_outputDir("lorawan_datasets") {
}

PacketRecorder::~PacketRecorder() {
}

void PacketRecorder::AddPacketRecord(const PacketRecord& record) {
    m_completedPackets.push_back(record);
}

void PacketRecorder::UpdatePacketReceived(uint32_t nodeId, double timestamp, double gatewayRssi, double gatewaySnr) {
    for (auto it = m_completedPackets.rbegin(); it != m_completedPackets.rend(); ++it) {
        if (!it->packetReceived && it->nodeId == nodeId &&
            std::abs(it->timestamp - timestamp) < 3.0) {
            it->packetReceived = true;
            it->gatewayRssiDbm = gatewayRssi;
            it->gatewaySnrDb = gatewaySnr;
            return;
        }
    }
}

uint32_t PacketRecorder::GetNextFrameCounter(uint32_t nodeId) {
    return ++m_deviceFrameCounters[nodeId];
}

DeviceMetrics& PacketRecorder::GetDeviceMetrics(uint32_t nodeId) {
    return m_deviceMetrics[nodeId];
}

void PacketRecorder::UpdateDeviceMetrics(uint32_t nodeId, bool packetSuccess, double energyConsumed) {
    DeviceMetrics& metrics = m_deviceMetrics[nodeId];
    metrics.packetsSent++;
    if (packetSuccess) {
        metrics.packetsReceived++;
    }
    metrics.totalEnergyConsumed += energyConsumed;
    
    if (metrics.packetsSent >= 10) {
        uint32_t recentReceived = 0;
        uint32_t recentSent = 0;
        
        for (auto it = m_completedPackets.rbegin(); it != m_completedPackets.rend() && recentSent < 10; ++it) {
            if (it->nodeId == nodeId) {
                recentSent++;
                if (it->packetReceived) {
                    recentReceived++;
                }
            }
        }
        metrics.lastPDR = (double)recentReceived / recentSent;
    } else {
        metrics.lastPDR = (double)metrics.packetsReceived / metrics.packetsSent;
    }
    
    metrics.lastEnergyPerPacket = metrics.totalEnergyConsumed / metrics.packetsSent;
    
    std::cout << "📊 Device " << nodeId << " metrics: PDR=" 
              << std::fixed << std::setprecision(3) << metrics.lastPDR 
              << ", Energy/pkt=" << metrics.lastEnergyPerPacket << " mJ" << std::endl;
}

const std::vector<PacketRecord>& PacketRecorder::GetCompletedPackets() const {
    return m_completedPackets;
}

void PacketRecorder::SetOutputDirectory(const std::string& dirPath) {
    m_outputDir = dirPath;
}

void PacketRecorder::WriteCSVOutput(const std::string& filename) const {
    std::string folderName = m_outputDir;
    system(("mkdir -p " + folderName).c_str());
    
    std::string baseFilename = filename;
    size_t lastDot = baseFilename.find_last_of('.');
    if (lastDot != std::string::npos) {
        baseFilename = baseFilename.substr(0, lastDot);
    }
    
    std::map<uint32_t, std::vector<PacketRecord>> devicePackets;
    for (const auto& record : m_completedPackets) {
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
    
    int totalPackets = m_completedPackets.size();
    int receivedPackets = 0;
    for (const auto& record : m_completedPackets) {
        if (record.packetReceived) {
            receivedPackets++;
        }
    }
    double successRate = (totalPackets > 0) ? (double)receivedPackets / totalPackets * 100.0 : 0.0;
    
    std::cout << "\n=== REALISTIC ns-3 LoRaWAN SIMULATION RESULTS ===" << std::endl;
    std::cout << "Total packets transmitted: " << totalPackets << std::endl;
    std::cout << "Successfully received: " << receivedPackets << " (" << successRate << "%)" << std::endl;
    std::cout << "Per-device datasets saved to: " << m_outputDir << "/ folder" << std::endl;
}

void PacketRecorder::Clear() {
    m_completedPackets.clear();
    m_deviceFrameCounters.clear();
    m_deviceMetrics.clear();
}

double CalculateEnergyConsumption(int sf, double txPowerDbm, uint32_t payloadBytes) {
    // Standard LoRa energy calculation - ALIGNED WITH advanced-ddqn-per-adr-example.cc
    double bandwidth = 125000.0;  // 125 kHz
    double symbolRate = bandwidth / std::pow(2, sf);
    
    // Preamble + header + payload symbols
    double preambleSymbols = 8;
    double headerSymbols = 4.25;
    double payloadSymbols = 8 + std::max(0.0, 
        std::ceil((8.0 * payloadBytes - 4.0 * sf + 28 + 16) / (4.0 * sf)) * 5);
    
    double totalSymbols = preambleSymbols + headerSymbols + payloadSymbols;
    double timeOnAir = totalSymbols / symbolRate;  // seconds
    
    // Current consumption based on TX power (realistic values)
    // SX1276: ~120mA at 17dBm, ~85mA at 2dBm
    double txCurrent_mA = 85.0 + (txPowerDbm - 2.0) * 2.3;  // Linear approximation
    
    // Energy = Power × Time = (Voltage × Current) × Time
    // Assuming 3.3V supply
    double voltage = 3.3;
    double energy_mJ = voltage * txCurrent_mA * timeOnAir;  // mJ
    
    // Add small RX window energy (fixed overhead)
    double rxEnergy_mJ = 0.033;  // ~10mA for 1ms
    
    return energy_mJ + rxEnergy_mJ;
}

} // namespace lorawan
} // namespace ns3
