/*
 * nbiot-helper.h
 *
 * Helper class for setting up NB-IoT simulations in ns-3.
 * Configures UEs, eNB, channel model, and energy model.
 */

#ifndef NBIOT_HELPER_H
#define NBIOT_HELPER_H

#include "ns3/nbiot-common.h"
#include "ns3/nbiot-channel.h"
#include "ns3/nbiot-net-device.h"
#include "ns3/nbiot-energy-model.h"

#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/object-factory.h"

#include <vector>

namespace ns3 {
namespace nbiot {

/**
 * NB-IoT Helper
 * Simplifies the creation and configuration of NB-IoT simulations.
 */
class NbiotHelper {
public:
    NbiotHelper();
    ~NbiotHelper();

    /**
     * Create and configure the NB-IoT channel
     * @param scenario  "urban_macro", "rural_macro", or "indoor"
     * @return Configured channel
     */
    Ptr<NbiotChannel> CreateChannel(const std::string& scenario = "urban_macro");

    /**
     * Install NB-IoT UE devices on nodes
     * @param channel   Pre-configured NB-IoT channel
     * @param ueNodes   Container of UE nodes
     * @return Vector of UE device pointers
     */
    std::vector<Ptr<NbiotUeDevice>> InstallUeDevices(
        Ptr<NbiotChannel> channel,
        NodeContainer ueNodes);

    /**
     * Install NB-IoT eNB device on node
     * @param channel   Pre-configured NB-IoT channel
     * @param enbNode   eNB node
     * @return eNB device pointer
     */
    Ptr<NbiotEnbDevice> InstallEnbDevice(
        Ptr<NbiotChannel> channel,
        Ptr<Node> enbNode);

    /**
     * Create an energy model
     * @return Configured energy model
     */
    Ptr<NbiotEnergyModel> CreateEnergyModel();

    // Configuration methods
    void SetUePowerClass(PowerClass pc);
    void SetEnbTxPower(double power_dBm);
    void SetEnbNoiseFigure(double nf_dB);
    void SetInterCellDistance(double distance_m);
    void SetNumNeighborCells(uint32_t nCells);

private:
    PowerClass m_uePowerClass;
    double     m_enbTxPower_dBm;
    double     m_enbNoiseFigure_dB;
    double     m_interCellDistance_m;
    uint32_t   m_numNeighborCells;
};

} // namespace nbiot
} // namespace ns3

#endif // NBIOT_HELPER_H
