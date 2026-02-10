/*
 * nbiot-channel.cc
 *
 * NB-IoT channel model implementation.
 * Uses 3GPP path loss models for NB-IoT licensed band operation.
 */

#include "nbiot-channel.h"

#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/constant-position-mobility-model.h"

namespace ns3 {
namespace nbiot {

NS_LOG_COMPONENT_DEFINE("NbiotChannel");
NS_OBJECT_ENSURE_REGISTERED(NbiotChannel);

TypeId
NbiotChannel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::nbiot::NbiotChannel")
        .SetParent<Object>()
        .AddConstructor<NbiotChannel>()
        .AddAttribute("InterCellDistance",
                       "Distance between cell centers in meters",
                       DoubleValue(1732.0),
                       MakeDoubleAccessor(&NbiotChannel::m_interCellDistance),
                       MakeDoubleChecker<double>(100.0, 50000.0))
        .AddAttribute("NumNeighborCells",
                       "Number of interfering neighbor cells",
                       UintegerValue(6),
                       MakeUintegerAccessor(&NbiotChannel::m_numNeighborCells),
                       MakeUintegerChecker<uint32_t>(0, 18))
        .AddAttribute("IndoorPenetrationLoss",
                       "Indoor penetration loss in dB",
                       DoubleValue(20.0),
                       MakeDoubleAccessor(&NbiotChannel::m_indoorPenetrationLoss),
                       MakeDoubleChecker<double>(0.0, 40.0));
    return tid;
}

NbiotChannel::NbiotChannel()
    : m_interCellDistance(1732.0),
      m_numNeighborCells(6),
      m_neighborTxPower_dBm(10.0),  // Equivalent UL interference per neighbor cell (NOT eNB DL power)
      m_indoorPenetrationLoss(20.0),
      m_enableTimeFading(true)
{
    m_shadowingRv = CreateObject<NormalRandomVariable>();
    m_shadowingRv->SetAttribute("Mean", DoubleValue(0.0));
    m_shadowingRv->SetAttribute("Variance", DoubleValue(16.0)); // 4 dB std dev (was 8, too high)

    m_fadingRv = CreateObject<UniformRandomVariable>();
    m_fadingRv->SetAttribute("Min", DoubleValue(0.0));
    m_fadingRv->SetAttribute("Max", DoubleValue(1.0));
}

NbiotChannel::~NbiotChannel()
{
}

void
NbiotChannel::SetPropagationLossModel(Ptr<PropagationLossModel> loss)
{
    m_loss = loss;
}

void
NbiotChannel::SetPropagationDelayModel(Ptr<PropagationDelayModel> delay)
{
    m_delay = delay;
}

double
NbiotChannel::GetRxPower(double txPowerDbm,
                          Ptr<MobilityModel> txMobility,
                          Ptr<MobilityModel> rxMobility) const
{
    if (m_loss) {
        return m_loss->CalcRxPower(txPowerDbm, txMobility, rxMobility);
    }

    // Fallback: 3GPP Urban Macro path loss model for 900 MHz
    // PL = 120.9 + 37.6 * log10(d_km)  (Hata model for urban macro)
    double distance = txMobility->GetDistanceFrom(rxMobility);
    if (distance < 1.0) distance = 1.0;

    double d_km = distance / 1000.0;
    double pathLoss = 120.9 + 37.6 * std::log10(d_km);

    return txPowerDbm - pathLoss;
}

double
NbiotChannel::GetCouplingLoss(Ptr<MobilityModel> txMobility,
                               Ptr<MobilityModel> rxMobility) const
{
    // Coupling loss = TX power - RX power (for 0 dBm TX)
    double rxPower = GetRxPower(0.0, txMobility, rxMobility);
    return -rxPower;  // Positive coupling loss in dB
}

double
NbiotChannel::CalculateSINR(double signalPowerDbm,
                             double interCellInterference_dBm,
                             double noiseFigure_dB) const
{
    // Noise floor = thermal noise + noise figure
    double thermalNoise_dBm = NbiotThermalNoise_dBm();  // ~-128.4 dBm for 180 kHz
    double noiseFloor_dBm = thermalNoise_dBm + noiseFigure_dB;

    // Total interference + noise
    double totalIN_W = DbmToW(noiseFloor_dBm) + DbmToW(interCellInterference_dBm);
    double totalIN_dBm = WToDbm(totalIN_W);

    // SINR = Signal / (Interference + Noise)
    double sinr_dB = signalPowerDbm - totalIN_dBm;

    return sinr_dB;
}

double
NbiotChannel::CalculateInterCellInterference(Vector uePosition, Time currentTime) const
{
    if (m_numNeighborCells == 0) {
        return -200.0; // No interference
    }

    double totalInterference_W = 0.0;

    // Model interfering neighbors placed at ISD distance in hexagonal layout
    for (uint32_t i = 0; i < m_numNeighborCells; i++) {
        double angle = 2.0 * M_PI * i / m_numNeighborCells;
        Vector neighborPos(
            m_interCellDistance * std::cos(angle),
            m_interCellDistance * std::sin(angle),
            30.0  // Typical eNB antenna height
        );

        double distance = std::sqrt(
            std::pow(uePosition.x - neighborPos.x, 2) +
            std::pow(uePosition.y - neighborPos.y, 2) +
            std::pow(uePosition.z - neighborPos.z, 2));

        if (distance < 10.0) distance = 10.0;

        // 3GPP path loss for 900 MHz
        double d_km = distance / 1000.0;
        double pathLoss = 120.9 + 37.6 * std::log10(d_km);

        // Neighbor cell UL interference power minus path loss
        double rxPower_dBm = m_neighborTxPower_dBm - pathLoss;

        // Add shadowing variation
        double shadowing = m_shadowingRv->GetValue() * 0.3; // Reduced variance for neighbors
        rxPower_dBm += shadowing;

        // Time-varying loading factor (simulates neighbor cell traffic)
        if (m_enableTimeFading) {
            double loadFactor = 0.3 + 0.2 * std::sin(currentTime.GetSeconds() / 45.0 + angle);
            rxPower_dBm += 10.0 * std::log10(std::max(0.01, loadFactor));
        }

        // NB-IoT frequency reuse: only ~10% chance of same subcarrier overlap
        double frequencyReuseFactor = 0.1;
        totalInterference_W += DbmToW(rxPower_dBm) * frequencyReuseFactor;
    }

    return WToDbm(totalInterference_W);
}

void NbiotChannel::SetInterCellDistance(double distance_m) { m_interCellDistance = distance_m; }
void NbiotChannel::SetNumNeighborCells(uint32_t nCells) { m_numNeighborCells = nCells; }
void NbiotChannel::SetIndoorPenetrationLoss(double loss_dB) { m_indoorPenetrationLoss = loss_dB; }
void NbiotChannel::SetEnableTimeFading(bool enable) { m_enableTimeFading = enable; }

} // namespace nbiot
} // namespace ns3
