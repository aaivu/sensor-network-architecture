/*
 * nbiot-energy-model.h
 *
 * NB-IoT energy consumption model based on UE state machine.
 * Models energy for: PSM sleep, eDRX, RACH, connected TX/RX.
 * Based on measurements from u-blox SARA-N2 and Quectel BC95.
 */

#ifndef NBIOT_ENERGY_MODEL_H
#define NBIOT_ENERGY_MODEL_H

#include "nbiot-common.h"

#include "ns3/object.h"
#include "ns3/nstime.h"

namespace ns3 {
namespace nbiot {

/**
 * NB-IoT Energy Model
 * Calculates energy consumption for each UE state and transition.
 * Uses a state-machine approach mirroring real NB-IoT module behavior.
 */
class NbiotEnergyModel : public Object {
public:
    static TypeId GetTypeId();
    NbiotEnergyModel();
    virtual ~NbiotEnergyModel();

    /**
     * Calculate total energy for a complete transmission cycle:
     * PSM wakeup → RACH → Data TX → Return to PSM
     *
     * @param txPower_dBm    Transmission power
     * @param mcs            Modulation and Coding Scheme index
     * @param repetitions    Number of data repetitions
     * @param ru             Resource unit type
     * @param rachAttempts   Number of RACH attempts needed
     * @param payloadBytes   Data payload size
     * @param psmSleepTime_s Time spent in PSM since last activity
     * @param ceLevel        Coverage enhancement level
     * @return Total energy consumed in mJ
     */
    double CalculateTransmissionCycleEnergy(
        double txPower_dBm,
        uint8_t mcs,
        int repetitions,
        ResourceUnit ru,
        int rachAttempts,
        uint32_t payloadBytes,
        double psmSleepTime_s,
        CELevel ceLevel) const;

    /**
     * Calculate energy for PSM sleep period
     * @param duration_s Duration in seconds
     * @return Energy in mJ
     */
    double CalculatePSMEnergy(double duration_s) const;

    /**
     * Calculate energy for eDRX cycle
     * @param cycleDuration_s  eDRX cycle duration
     * @param pagingDuration_ms Paging window duration
     * @return Energy per eDRX cycle in mJ
     */
    double CalculateEDRXEnergy(double cycleDuration_s, double pagingDuration_ms = 1.0) const;

    /**
     * Calculate energy for RACH procedure
     * @param rachAttempts Number of preamble attempts
     * @param ceLevel     CE level (affects preamble repetitions)
     * @return Energy in mJ
     */
    double CalculateRACHEnergy(int rachAttempts, CELevel ceLevel) const;

    /**
     * Calculate energy for uplink data transmission
     * @param txPower_dBm  TX power
     * @param airtime_ms   Total airtime (includes repetitions)
     * @return Energy in mJ
     */
    double CalculateUplinkTxEnergy(double txPower_dBm, double airtime_ms) const;

    /**
     * Calculate energy for downlink reception (RAR, scheduling grant, etc.)
     * @param rxDuration_ms  Duration of RX window
     * @return Energy in mJ
     */
    double CalculateDownlinkRxEnergy(double rxDuration_ms) const;

    /**
     * Calculate energy for connected mode idle (inactivity timer)
     * @param duration_ms  Duration in ms
     * @return Energy in mJ
     */
    double CalculateConnectedIdleEnergy(double duration_ms) const;

    // Configuration
    void SetSupplyVoltage(double voltage_V);
    void SetBatteryCapacity(double capacity_mAh);
    void SetCurrentConsumption(const CurrentConsumption& current);

private:
    CurrentConsumption m_current;
    double m_supplyVoltage_V;
    double m_batteryCapacity_mAh;

    // Wakeup transition energy (measured from real modules)
    double m_wakeupTransitionEnergy_mJ;  // ~0.05 mJ for PSM→Active transition
};

} // namespace nbiot
} // namespace ns3

#endif // NBIOT_ENERGY_MODEL_H
