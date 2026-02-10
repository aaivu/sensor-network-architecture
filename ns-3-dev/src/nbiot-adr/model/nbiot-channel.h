/*
 * nbiot-channel.h
 *
 * NB-IoT channel model using 3GPP path loss models.
 * Supports Urban Macro (UMa) and Rural Macro (RMa) scenarios.
 * Wraps ns-3 propagation loss models configured for NB-IoT bands.
 */

#ifndef NBIOT_CHANNEL_H
#define NBIOT_CHANNEL_H

#include "nbiot-common.h"

#include "ns3/object.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/mobility-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/vector.h"
#include "ns3/nstime.h"

#include <map>
#include <vector>

namespace ns3 {
namespace nbiot {

/**
 * NB-IoT Channel Model
 * 
 * Implements 3GPP TR 36.802 / 36.942 path loss for NB-IoT.
 * The model consists of:
 *   - Distance-dependent path loss (log-distance or Hata model)
 *   - Log-normal shadowing
 *   - Indoor penetration loss
 *   - Inter-cell interference modeling
 *   - Time-varying fading
 */
class NbiotChannel : public Object {
public:
    static TypeId GetTypeId();
    NbiotChannel();
    virtual ~NbiotChannel();

    /**
     * Set the propagation loss model
     */
    void SetPropagationLossModel(Ptr<PropagationLossModel> loss);

    /**
     * Set the propagation delay model
     */
    void SetPropagationDelayModel(Ptr<PropagationDelayModel> delay);

    /**
     * Calculate received power at eNB from a UE transmission
     * @param txPowerDbm   Transmit power in dBm
     * @param txMobility   UE mobility model (position)
     * @param rxMobility   eNB mobility model (position)
     * @return Received power in dBm
     */
    double GetRxPower(double txPowerDbm,
                      Ptr<MobilityModel> txMobility,
                      Ptr<MobilityModel> rxMobility) const;

    /**
     * Calculate coupling loss between UE and eNB
     * @param txMobility UE position
     * @param rxMobility eNB position
     * @return Coupling loss in dB
     */
    double GetCouplingLoss(Ptr<MobilityModel> txMobility,
                           Ptr<MobilityModel> rxMobility) const;

    /**
     * Calculate SINR at eNB for a given UE transmission
     * @param signalPowerDbm    Received signal power in dBm
     * @param interCellInterference_dBm  Inter-cell interference power in dBm
     * @param noiseFigure_dB    eNB noise figure in dB (typically 3-5 dB)
     * @return SINR in dB
     */
    double CalculateSINR(double signalPowerDbm,
                         double interCellInterference_dBm,
                         double noiseFigure_dB = 5.0) const;

    /**
     * Model inter-cell interference from neighboring cells
     * Simulates aggregate interference from N neighboring eNBs
     * @param uePosition  UE position
     * @param currentTime Simulation time for time-varying effects
     * @return Inter-cell interference power in dBm
     */
    double CalculateInterCellInterference(Vector uePosition, Time currentTime) const;

    // Configuration
    void SetInterCellDistance(double distance_m);
    void SetNumNeighborCells(uint32_t nCells);
    void SetIndoorPenetrationLoss(double loss_dB);
    void SetEnableTimeFading(bool enable);

private:
    Ptr<PropagationLossModel>  m_loss;
    Ptr<PropagationDelayModel> m_delay;

    // Inter-cell interference parameters
    double   m_interCellDistance;      // Distance between cell centers (meters)
    uint32_t m_numNeighborCells;       // Number of interfering neighbor cells
    double   m_neighborTxPower_dBm;    // Neighbor cell TX power
    double   m_indoorPenetrationLoss;  // Indoor penetration loss (dB)
    bool     m_enableTimeFading;       // Enable time-varying fading

    // Random variables for fading
    Ptr<NormalRandomVariable> m_shadowingRv;
    Ptr<UniformRandomVariable> m_fadingRv;
};

} // namespace nbiot
} // namespace ns3

#endif // NBIOT_CHANNEL_H
