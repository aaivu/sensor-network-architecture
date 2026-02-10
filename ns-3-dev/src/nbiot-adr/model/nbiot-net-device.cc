/*
 * nbiot-net-device.cc
 *
 * NB-IoT UE and eNB model implementations.
 */

#include "nbiot-net-device.h"

#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"

namespace ns3 {
namespace nbiot {

NS_LOG_COMPONENT_DEFINE("NbiotNetDevice");

// ============================================================================
// NbiotUeDevice Implementation
// ============================================================================

NS_OBJECT_ENSURE_REGISTERED(NbiotUeDevice);

TypeId
NbiotUeDevice::GetTypeId()
{
    static TypeId tid = TypeId("ns3::nbiot::NbiotUeDevice")
        .SetParent<Object>()
        .AddConstructor<NbiotUeDevice>();
    return tid;
}

NbiotUeDevice::NbiotUeDevice()
    : m_ueId(0),
      m_mcs(0),
      m_repetitions(1),
      m_ru(ResourceUnit::SINGLE_TONE_15KHZ),
      m_txPower_dBm(23.0),
      m_powerClass(PowerClass::CLASS_3),
      m_state(UEState::PSM_SLEEP),
      m_ceLevel(CELevel::CE0),
      m_psmActiveTimer_s(10.0),
      m_psmTauTimer_s(3600.0),
      m_edrxCycle_s(20.48),
      m_totalEnergy_mJ(0.0),
      m_batteryCapacity_mAh(2400.0),
      m_supplyVoltage_V(3.6),
      m_packetsSent(0),
      m_packetsDelivered(0)
{
}

NbiotUeDevice::~NbiotUeDevice() {}

void NbiotUeDevice::SetNode(Ptr<Node> node) { m_node = node; }
Ptr<Node> NbiotUeDevice::GetNode() const { return m_node; }
void NbiotUeDevice::SetChannel(Ptr<NbiotChannel> channel) { m_channel = channel; }
void NbiotUeDevice::SetUeId(uint32_t id) { m_ueId = id; }
uint32_t NbiotUeDevice::GetUeId() const { return m_ueId; }

void NbiotUeDevice::SetMCS(uint8_t mcs) { m_mcs = std::min(mcs, (uint8_t)10); }
uint8_t NbiotUeDevice::GetMCS() const { return m_mcs; }
void NbiotUeDevice::SetRepetitions(int reps) {
    // Clamp to valid repetition values
    int valid[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int closest = 1;
    for (int v : valid) {
        if (std::abs(v - reps) < std::abs(closest - reps)) closest = v;
    }
    m_repetitions = closest;
}
int NbiotUeDevice::GetRepetitions() const { return m_repetitions; }
void NbiotUeDevice::SetResourceUnit(ResourceUnit ru) { m_ru = ru; }
ResourceUnit NbiotUeDevice::GetResourceUnit() const { return m_ru; }
void NbiotUeDevice::SetTxPower(double power_dBm) {
    double maxP = POWER_CLASSES[static_cast<int>(m_powerClass)].maxPower_dBm;
    m_txPower_dBm = std::max(0.0, std::min(power_dBm, maxP));
}
double NbiotUeDevice::GetTxPower() const { return m_txPower_dBm; }
void NbiotUeDevice::SetPowerClass(PowerClass pc) { m_powerClass = pc; }

void NbiotUeDevice::SetPSMActiveTimer(double seconds) { m_psmActiveTimer_s = seconds; }
void NbiotUeDevice::SetPSMTauTimer(double seconds) { m_psmTauTimer_s = seconds; }
void NbiotUeDevice::SetEDRXCycle(double seconds) { m_edrxCycle_s = seconds; }

UEState NbiotUeDevice::GetState() const { return m_state; }
CELevel NbiotUeDevice::GetCELevel() const { return m_ceLevel; }
void NbiotUeDevice::SetCELevel(CELevel level) { m_ceLevel = level; }

double NbiotUeDevice::GetBatteryLevel() const {
    double totalCapacity_mJ = m_batteryCapacity_mAh * m_supplyVoltage_V * 3600.0;  // mAh * V * 3600 = mJ
    double remaining = 1.0 - (m_totalEnergy_mJ / totalCapacity_mJ);
    return std::max(0.0, std::min(1.0, remaining));
}

double NbiotUeDevice::GetTotalEnergyConsumed_mJ() const { return m_totalEnergy_mJ; }

uint32_t NbiotUeDevice::GetTotalPacketsSent() const { return m_packetsSent; }
uint32_t NbiotUeDevice::GetTotalPacketsDelivered() const { return m_packetsDelivered; }
double NbiotUeDevice::GetPDR() const {
    return m_packetsSent > 0 ? static_cast<double>(m_packetsDelivered) / m_packetsSent : 0.0;
}

void NbiotUeDevice::SetTxCompleteCallback(TxCompleteCallback cb) { m_txCompleteCb = cb; }


// ============================================================================
// NbiotEnbDevice Implementation
// ============================================================================

NS_OBJECT_ENSURE_REGISTERED(NbiotEnbDevice);

TypeId
NbiotEnbDevice::GetTypeId()
{
    static TypeId tid = TypeId("ns3::nbiot::NbiotEnbDevice")
        .SetParent<Object>()
        .AddConstructor<NbiotEnbDevice>();
    return tid;
}

NbiotEnbDevice::NbiotEnbDevice()
    : m_noiseFigure_dB(5.0),
      m_txPower_dBm(46.0),
      m_currentRachContenders(0),
      m_schedulerIndex(0),
      m_totalResourcesUsed(0.0),
      m_cellLoad(0.0)
{
}

NbiotEnbDevice::~NbiotEnbDevice() {}

void NbiotEnbDevice::SetNode(Ptr<Node> node) { m_node = node; }
Ptr<Node> NbiotEnbDevice::GetNode() const { return m_node; }
void NbiotEnbDevice::SetChannel(Ptr<NbiotChannel> channel) { m_channel = channel; }
void NbiotEnbDevice::SetNoiseFigure(double nf_dB) { m_noiseFigure_dB = nf_dB; }
void NbiotEnbDevice::SetTxPower(double power_dBm) { m_txPower_dBm = power_dBm; }

NbiotEnbDevice::RACHResult
NbiotEnbDevice::ProcessRACH(uint32_t ueId, double preamblePower_dBm,
                             CELevel ceLevel, Time currentTime)
{
    RACHResult result;

    int maxAttempts = CE_CONFIGS[static_cast<int>(ceLevel)].maxNPRACHAttempts;

    // Estimate concurrent RACH contenders based on registered UEs
    // Only a small fraction of UEs transmit in any given RACH occasion
    uint32_t concurrentContenders = std::max((uint32_t)1,
        (uint32_t)(m_registeredUEs.size() * 0.15)); // ~15% active at any time
    double collisionProb = CalculateRACHCollisionProb(concurrentContenders);

    // Track attempts per UE
    if (m_rachAttemptCount.find(ueId) == m_rachAttemptCount.end()) {
        m_rachAttemptCount[ueId] = 0;
    }

    result.attemptsUsed = 0;
    result.success = false;
    result.collisionDetected = false;

    // Simulate RACH procedure with multiple attempts
    for (int attempt = 0; attempt < maxAttempts; attempt++) {
        result.attemptsUsed++;

        // Random collision check
        double roll = static_cast<double>(rand()) / RAND_MAX;
        if (roll < collisionProb) {
            result.collisionDetected = true;
            continue; // Retry after backoff
        }

        // Power check: preamble must be received above threshold
        double threshold = NbiotThermalNoise_dBm() + m_noiseFigure_dB + 3.0; // 3 dB margin
        double effectivePower = preamblePower_dBm + GetRepetitionGainDb(
            CE_CONFIGS[static_cast<int>(ceLevel)].maxRepetitions);

        if (effectivePower >= threshold) {
            result.success = true;
            break;
        }
    }

    m_rachAttemptCount[ueId] = result.attemptsUsed;
    m_ueLastActivity[ueId] = currentTime;

    return result;
}

NbiotEnbDevice::UplinkResult
NbiotEnbDevice::ProcessUplink(uint32_t ueId, double rxPower_dBm,
                               uint8_t mcs, int repetitions,
                               ResourceUnit ru, Time currentTime)
{
    UplinkResult result;

    // Calculate inter-cell interference at eNB
    Vector enbPos(0.0, 0.0, 30.0); // eNB position (set in simulation)
    double interCellInterf = -200.0;
    if (m_channel) {
        interCellInterf = m_channel->CalculateInterCellInterference(enbPos, currentTime);
    }

    // Calculate SINR at eNB
    double thermalNoise = NbiotThermalNoise_dBm();
    double noiseFloor = thermalNoise + m_noiseFigure_dB;
    double totalIN_W = DbmToW(noiseFloor) + DbmToW(interCellInterf);

    // Apply repetition gain
    double effectiveRxPower = rxPower_dBm + GetRepetitionGainDb(repetitions);

    // Apply RU processing gain
    uint8_t ruIdx = static_cast<uint8_t>(ru);
    effectiveRxPower += RU_CONFIGS[ruIdx].processingGainDb;

    result.sinr_dB = effectiveRxPower - WToDbm(totalIN_W);
    result.rsrp_dBm = rxPower_dBm;

    // RSRQ = N × RSRP / RSSI (in linear), where N = number of RBs
    // For NB-IoT, simplified: RSRQ ≈ SINR - 3 dB
    result.rsrq_dB = std::max(-20.0, std::min(-3.0, result.sinr_dB - 3.0));

    // Check if decodable
    result.success = (result.sinr_dB >= MCS_TABLE[mcs].requiredSINR_dB);

    // Update cell load
    m_ueLastActivity[ueId] = currentTime;
    double ruUsage = RU_CONFIGS[ruIdx].durationMs * repetitions;
    m_totalResourcesUsed += ruUsage;

    return result;
}

double NbiotEnbDevice::GetCellLoad() const { return m_cellLoad; }
uint32_t NbiotEnbDevice::GetNumConnectedUEs() const { return m_registeredUEs.size(); }

void NbiotEnbDevice::RegisterUE(uint32_t ueId) {
    m_registeredUEs.push_back(ueId);
    // Update cell load based on number of connected UEs
    // Assume capacity ~50 simultaneous NB-IoT UEs per cell
    m_cellLoad = std::min(1.0, static_cast<double>(m_registeredUEs.size()) / 50.0);
}

void NbiotEnbDevice::DeregisterUE(uint32_t ueId) {
    m_registeredUEs.erase(
        std::remove(m_registeredUEs.begin(), m_registeredUEs.end(), ueId),
        m_registeredUEs.end());
    m_cellLoad = std::min(1.0, static_cast<double>(m_registeredUEs.size()) / 50.0);
}

NbiotEnbDevice::ScheduleGrant
NbiotEnbDevice::ScheduleUplink(uint32_t ueId, uint8_t requestedMCS,
                                int requestedReps, ResourceUnit requestedRU)
{
    ScheduleGrant grant;
    grant.assignedMCS = requestedMCS;
    grant.assignedRepetitions = requestedReps;
    grant.assignedRU = static_cast<uint8_t>(requestedRU);

    // Round-robin scheduling delay
    double schedulingDelay = 1.0; // 1 ms base scheduling delay
    if (!m_registeredUEs.empty()) {
        schedulingDelay += static_cast<double>(m_schedulerIndex % m_registeredUEs.size()) * 0.5;
    }
    grant.scheduledTime_ms = schedulingDelay;

    m_schedulerIndex++;
    return grant;
}

} // namespace nbiot
} // namespace ns3
