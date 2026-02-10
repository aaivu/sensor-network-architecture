/*
 * nbiot-energy-model.cc
 *
 * NB-IoT energy consumption model implementation.
 * Based on measurements from commercial NB-IoT modules.
 */

#include "nbiot-energy-model.h"

#include "ns3/log.h"
#include "ns3/double.h"

namespace ns3 {
namespace nbiot {

NS_LOG_COMPONENT_DEFINE("NbiotEnergyModel");
NS_OBJECT_ENSURE_REGISTERED(NbiotEnergyModel);

TypeId
NbiotEnergyModel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::nbiot::NbiotEnergyModel")
        .SetParent<Object>()
        .AddConstructor<NbiotEnergyModel>();
    return tid;
}

NbiotEnergyModel::NbiotEnergyModel()
    : m_current(TYPICAL_CURRENT),
      m_supplyVoltage_V(3.6),
      m_batteryCapacity_mAh(2400.0),
      m_wakeupTransitionEnergy_mJ(0.05)
{
}

NbiotEnergyModel::~NbiotEnergyModel() {}

double
NbiotEnergyModel::CalculateTransmissionCycleEnergy(
    double txPower_dBm,
    uint8_t mcs,
    int repetitions,
    ResourceUnit ru,
    int rachAttempts,
    uint32_t payloadBytes,
    double psmSleepTime_s,
    CELevel ceLevel) const
{
    double totalEnergy_mJ = 0.0;

    // 1. PSM sleep energy (during sleep period before this transmission)
    totalEnergy_mJ += CalculatePSMEnergy(psmSleepTime_s);

    // 2. Wakeup transition energy
    totalEnergy_mJ += m_wakeupTransitionEnergy_mJ;

    // 3. RACH procedure energy
    totalEnergy_mJ += CalculateRACHEnergy(rachAttempts, ceLevel);

    // 4. Downlink RX energy (RAR + scheduling grant)
    double rarDuration_ms = 5.0;   // RAR window
    double grantDuration_ms = 2.0; // DCI/scheduling grant
    totalEnergy_mJ += CalculateDownlinkRxEnergy(rarDuration_ms + grantDuration_ms);

    // 5. Uplink TX energy
    double airtime_ms = CalculateAirtime_ms(mcs, repetitions, ru, payloadBytes);
    totalEnergy_mJ += CalculateUplinkTxEnergy(txPower_dBm, airtime_ms);

    // 6. Downlink RX for ACK
    totalEnergy_mJ += CalculateDownlinkRxEnergy(2.0);

    // 7. Inactivity timer energy (before returning to idle/PSM)
    // Typical inactivity timer: 10-20 seconds
    double inactivityTimer_ms = 10000.0; // 10 seconds default
    // In connected idle state during inactivity timer
    totalEnergy_mJ += CalculateConnectedIdleEnergy(inactivityTimer_ms);

    return totalEnergy_mJ;
}

double
NbiotEnergyModel::CalculatePSMEnergy(double duration_s) const
{
    // Energy = I_psm × V × t
    // I_psm is in μA, convert to mA
    double current_mA = m_current.psm_uA / 1000.0;
    // P(mW) = I(mA) × V(V), E(mJ) = P(mW) × t(s)
    double energy_mJ = current_mA * m_supplyVoltage_V * duration_s;
    return energy_mJ;
}

double
NbiotEnergyModel::CalculateEDRXEnergy(double cycleDuration_s, double pagingDuration_ms) const
{
    // Sleep phase energy: P(mW) = I(mA) × V(V), E(mJ) = P(mW) × t(s)
    double sleepDuration_s = cycleDuration_s - (pagingDuration_ms / 1000.0);
    double sleepCurrent_mA = m_current.edrx_sleep_uA / 1000.0;
    double sleepEnergy_mJ = sleepCurrent_mA * m_supplyVoltage_V * sleepDuration_s;

    // Paging window energy: E(mJ) = I(mA) × V(V) × t(ms) / 1000
    double pagingEnergy_mJ = m_current.edrx_paging_mA * m_supplyVoltage_V * pagingDuration_ms / 1000.0;

    return sleepEnergy_mJ + pagingEnergy_mJ;
}

double
NbiotEnergyModel::CalculateRACHEnergy(int rachAttempts, CELevel ceLevel) const
{
    double totalEnergy_mJ = 0.0;

    int preambleReps = CE_CONFIGS[static_cast<int>(ceLevel)].maxRepetitions;
    if (preambleReps < 1) preambleReps = 1;

    // Each RACH attempt: preamble TX + RAR wait
    for (int i = 0; i < rachAttempts; i++) {
        // Preamble duration with repetitions
        double preambleDuration_ms = DEFAULT_NPRACH_CONFIG.preambleDurationMs * preambleReps;

        // TX energy for preamble: P(mW) = I(mA) × V(V), E(mJ) = P(mW) × t(ms) / 1000
        double txCurrent_mA = m_current.rach_mA;
        totalEnergy_mJ += txCurrent_mA * m_supplyVoltage_V * preambleDuration_ms / 1000.0;

        // RAR wait energy (RX mode)
        double rarWindow_ms = DEFAULT_NPRACH_CONFIG.rarWindowMs;
        totalEnergy_mJ += m_current.connected_rx_mA * m_supplyVoltage_V * rarWindow_ms / 1000.0;

        // Backoff energy (idle mode)
        if (i < rachAttempts - 1) {
            double backoff_ms = DEFAULT_NPRACH_CONFIG.backoffMs;
            totalEnergy_mJ += m_current.connected_idle_mA * m_supplyVoltage_V * backoff_ms / 1000.0;
        }
    }

    return totalEnergy_mJ;
}

double
NbiotEnergyModel::CalculateUplinkTxEnergy(double txPower_dBm, double airtime_ms) const
{
    double txCurrent_mA = GetTxCurrent_mA(txPower_dBm);
    // Energy = I × V × t (current in mA, voltage in V, time in ms → energy in μJ, /1000 → mJ)
    // Actually: mA × V × ms = μW·s = μJ... wait
    // I (mA) × V (V) = P (mW), P (mW) × t (ms) = E (μJ) = E/1000 (mJ)
    // So: mA × V × ms / 1000 = mJ? No.
    // P = I × V = mA × V = mW
    // E = P × t = mW × ms = μJ != mJ
    // E in mJ = P(mW) × t(s) = I(mA) × V(V) × t(s)
    // If t is in ms: E(mJ) = I(mA) × V(V) × t(ms) / 1000
    // Wait no. E(mJ) = P(mW) × t(s) = I(mA) × V(V) × t(ms) / 1000
    // Actually: mW × s = mJ. So P(mW) × t_s = E(mJ).
    // t_s = t_ms / 1000
    // E(mJ) = I(mA) × V(V) × t(ms) / 1000
    return txCurrent_mA * m_supplyVoltage_V * airtime_ms / 1000.0;
}

double
NbiotEnergyModel::CalculateDownlinkRxEnergy(double rxDuration_ms) const
{
    return m_current.connected_rx_mA * m_supplyVoltage_V * rxDuration_ms / 1000.0;
}

double
NbiotEnergyModel::CalculateConnectedIdleEnergy(double duration_ms) const
{
    return m_current.connected_idle_mA * m_supplyVoltage_V * duration_ms / 1000.0;
}

void NbiotEnergyModel::SetSupplyVoltage(double voltage_V) { m_supplyVoltage_V = voltage_V; }
void NbiotEnergyModel::SetBatteryCapacity(double capacity_mAh) { m_batteryCapacity_mAh = capacity_mAh; }
void NbiotEnergyModel::SetCurrentConsumption(const CurrentConsumption& current) { m_current = current; }

} // namespace nbiot
} // namespace ns3
