/*
 * nbiot-common.h
 *
 * NB-IoT common definitions, MCS tables, repetition configurations,
 * coverage enhancement levels, and utility functions.
 *
 * Based on 3GPP TS 36.211/36.213/36.321 for NB-IoT.
 */

#ifndef NBIOT_COMMON_H
#define NBIOT_COMMON_H

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace ns3 {
namespace nbiot {

// ============================================================================
// NB-IoT FREQUENCY BANDS
// ============================================================================
// NB-IoT operates in licensed LTE bands (in-band, guard-band, or standalone)
// Common deployments: Band 8 (900 MHz), Band 20 (800 MHz), Band 28 (700 MHz)

constexpr double NBIOT_CARRIER_FREQ_HZ = 900e6;      // 900 MHz (Band 8)
constexpr double NBIOT_BANDWIDTH_HZ    = 180e3;       // 180 kHz (one LTE PRB)
constexpr double NBIOT_SUBCARRIER_SPACING_15KHZ = 15e3;
constexpr double NBIOT_SUBCARRIER_SPACING_3_75KHZ = 3.75e3;

// ============================================================================
// MODULATION AND CODING SCHEME (MCS) TABLE
// ============================================================================
// 3GPP TS 36.213 Table 16.5.1.2-1 for NPUSCH
// MCS Index → {Modulation Order, TBS Index}

struct MCSEntry {
    uint8_t  index;
    uint8_t  modulationOrder;    // 1=pi/2-BPSK, 2=QPSK
    uint8_t  tbsIndex;
    double   requiredSINR_dB;    // Minimum SINR for 10% BLER (no repetitions)
    double   spectralEfficiency; // bits/s/Hz
    std::string modulationName;
};

// MCS table for NB-IoT NPUSCH (simplified from 3GPP)
static const MCSEntry MCS_TABLE[] = {
    { 0,  2,  0,  -12.6, 0.15, "QPSK"},   // Most robust
    { 1,  2,  1,  -10.5, 0.23, "QPSK"},
    { 2,  2,  2,   -8.4, 0.38, "QPSK"},
    { 3,  2,  3,   -6.3, 0.60, "QPSK"},
    { 4,  2,  4,   -4.2, 0.88, "QPSK"},
    { 5,  2,  5,   -2.1, 1.18, "QPSK"},
    { 6,  2,  6,    0.0, 1.48, "QPSK"},
    { 7,  2,  7,    2.1, 1.91, "QPSK"},
    { 8,  2,  8,    4.2, 2.41, "QPSK"},
    { 9,  2,  9,    6.0, 2.73, "QPSK"},
    { 10, 2, 10,    8.0, 3.32, "QPSK"},   // Highest rate
};
static const int NUM_MCS_LEVELS = 11;

// ============================================================================
// TRANSPORT BLOCK SIZE (TBS) TABLE
// ============================================================================
// 3GPP TS 36.213 Table 16.5.1.2-2 (simplified)
// TBS Index × Number of Resource Units → Transport Block Size (bits)

// For single-tone 15 kHz, 1 RU = 1 subcarrier × 1 ms slot
static const int TBS_TABLE_SINGLE_TONE[][8] = {
    // nRU:  1,    2,    3,    4,    5,    6,    8,   10
    /* I=0 */{ 16,   32,   56,   88,  120,  152,  208,  256},
    /* I=1 */{ 24,   56,   88,  144,  176,  208,  256,  344},
    /* I=2 */{ 32,   72,  144,  176,  208,  256,  392,  504},
    /* I=3 */{ 40,   88,  176,  208,  256,  328,  504,  680},
    /* I=4 */{ 56,  120,  208,  256,  328,  408,  680,  872},
    /* I=5 */{ 72,  144,  256,  328,  408,  504,  872, 1000},
    /* I=6 */{ 88,  176,  328,  408,  504,  680, 1000, 1224},
    /* I=7 */{104,  224,  392,  504,  680,  872, 1224, 1480},
    /* I=8 */{120,  256,  504,  680,  872, 1000, 1480, 1800},
    /* I=9 */{136,  296,  584,  776,  968, 1192, 1608, 2024},
    /* I=10*/{144,  328,  680,  872, 1128, 1352, 1800, 2280},
};

// ============================================================================
// REPETITION LEVELS
// ============================================================================
// 3GPP TS 36.213 - NPUSCH repetitions for coverage enhancement

enum class RepetitionLevel : uint8_t {
    REP_1   = 0,   // 1 repetition (no repetition)
    REP_2   = 1,   // 2 repetitions
    REP_4   = 2,   // 4 repetitions
    REP_8   = 3,   // 8 repetitions
    REP_16  = 4,   // 16 repetitions
    REP_32  = 5,   // 32 repetitions
    REP_64  = 6,   // 64 repetitions
    REP_128 = 7,   // 128 repetitions
};

static const int REPETITION_VALUES[] = {1, 2, 4, 8, 16, 32, 64, 128};
static const int NUM_REPETITION_LEVELS = 8;

// Repetition gain in dB (approximately 3 dB per doubling)
inline double GetRepetitionGainDb(int numRepetitions) {
    return 10.0 * std::log10(static_cast<double>(numRepetitions));
}

// ============================================================================
// COVERAGE ENHANCEMENT (CE) LEVELS
// ============================================================================
// 3GPP TS 36.331

enum class CELevel : uint8_t {
    CE0 = 0,   // Normal coverage (MCL up to 144 dB)
    CE1 = 1,   // Extended coverage (MCL up to 154 dB)
    CE2 = 2,   // Extreme coverage (MCL up to 164 dB)
};

struct CELevelConfig {
    CELevel level;
    double  maxCouplingLoss_dB;   // Maximum coupling loss
    int     maxRepetitions;       // Maximum repetitions allowed
    int     maxNPRACHAttempts;    // Maximum RACH preamble attempts
    double  rachPreamblePower_dBm; // NPRACH initial power
    std::string name;
};

static const CELevelConfig CE_CONFIGS[] = {
    {CELevel::CE0, 144.0,   1,  5, -10.0, "CE Level 0 (Normal)"},
    {CELevel::CE1, 154.0,  16, 10, -6.0,  "CE Level 1 (Extended)"},
    {CELevel::CE2, 164.0, 128, 15, -2.0,  "CE Level 2 (Extreme)"},
};

// ============================================================================
// RESOURCE UNIT (RU) ALLOCATION
// ============================================================================
// 3GPP TS 36.211 - NPUSCH resource unit definitions

enum class ResourceUnit : uint8_t {
    SINGLE_TONE_3_75KHZ = 0,  // 1 subcarrier × 3.75 kHz (max range)
    SINGLE_TONE_15KHZ   = 1,  // 1 subcarrier × 15 kHz
    THREE_TONE           = 2,  // 3 subcarriers × 15 kHz (45 kHz)
    SIX_TONE             = 3,  // 6 subcarriers × 15 kHz (90 kHz)
    TWELVE_TONE          = 4,  // 12 subcarriers × 15 kHz (180 kHz, full BW)
};

struct RUConfig {
    ResourceUnit type;
    uint8_t      numSubcarriers;
    double       spacingKHz;
    double       bandwidthKHz;
    double       durationMs;       // RU duration in ms
    double       processingGainDb; // Narrowband processing gain
    std::string  name;
};

static const RUConfig RU_CONFIGS[] = {
    {ResourceUnit::SINGLE_TONE_3_75KHZ,  1,  3.75,   3.75, 32.0, 16.8, "1-tone 3.75kHz"},
    {ResourceUnit::SINGLE_TONE_15KHZ,    1, 15.00,  15.00,  8.0, 10.8, "1-tone 15kHz"},
    {ResourceUnit::THREE_TONE,           3, 15.00,  45.00,  4.0,  6.0, "3-tone"},
    {ResourceUnit::SIX_TONE,             6, 15.00,  90.00,  2.0,  3.0, "6-tone"},
    {ResourceUnit::TWELVE_TONE,         12, 15.00, 180.00,  1.0,  0.0, "12-tone (full BW)"},
};
static const int NUM_RU_TYPES = 5;

// ============================================================================
// NB-IoT UE POWER CLASSES
// ============================================================================.

enum class PowerClass : uint8_t {
    CLASS_3 = 0,  // 23 dBm (200 mW) - Standard
    CLASS_5 = 1,  // 20 dBm (100 mW) - Reduced
    CLASS_6 = 2,  // 14 dBm (25 mW)  - Low power (NB-IoT specific)
};

struct PowerClassConfig {
    PowerClass cls;
    double maxPower_dBm;
    double minPower_dBm;
    double stepSize_dB;
    std::string name;
};

static const PowerClassConfig POWER_CLASSES[] = {
    {PowerClass::CLASS_3, 23.0, 0.0, 1.0, "Class 3 (23 dBm)"},
    {PowerClass::CLASS_5, 20.0, 0.0, 1.0, "Class 5 (20 dBm)"},
    {PowerClass::CLASS_6, 14.0, 0.0, 1.0, "Class 6 (14 dBm)"},
};

// ============================================================================
// POWER SAVING MODE (PSM) AND eDRX PARAMETERS
// ============================================================================

// PSM Active Timer (T3324) values in seconds
static const double PSM_ACTIVE_TIMER_VALUES[] = {
    0.0, 2.0, 10.0, 30.0, 60.0, 120.0, 360.0, 720.0
};

// PSM Extended TAU Timer (T3412) values in seconds
// Range: 10 min to 310 hours
static const double PSM_TAU_TIMER_VALUES[] = {
    600.0, 1800.0, 3600.0, 36000.0, 360000.0, 1116000.0
};

// eDRX cycle values in seconds
// Range: 5.12s to 2621.44s (~43 min)
static const double EDRX_CYCLE_VALUES[] = {
    5.12, 10.24, 20.48, 40.96, 81.92, 163.84, 327.68, 655.36,
    1310.72, 2621.44
};

// ============================================================================
// NB-IoT UE STATE MACHINE
// ============================================================================

enum class UEState : uint8_t {
    PSM_SLEEP       = 0,  // Power Saving Mode (deepest sleep, ~3 μA)
    EDRX_SLEEP      = 1,  // eDRX sleep phase
    EDRX_PAGING     = 2,  // eDRX paging occasion (listening)
    IDLE_DRX        = 3,  // RRC_IDLE with DRX
    RACH_PREAMBLE   = 4,  // Sending NPRACH preamble
    RACH_RAR_WAIT   = 5,  // Waiting for Random Access Response
    RACH_MSG3       = 6,  // Sending RRC Connection Request
    RACH_MSG4_WAIT  = 7,  // Waiting for Contention Resolution
    CONNECTED_TX    = 8,  // RRC_CONNECTED transmitting on NPUSCH
    CONNECTED_RX    = 9,  // RRC_CONNECTED receiving on NPDSCH
    CONNECTED_IDLE  = 10, // RRC_CONNECTED but idle (inactivity timer)
};

// ============================================================================
// NPRACH (NB-IoT PRACH) PARAMETERS
// ============================================================================

struct NPRACHConfig {
    uint8_t  preambleFormat;      // 0 or 1
    double   preambleDurationMs;  // 5.6 ms (format 0) or 6.4 ms (format 1)
    uint8_t  numPreambles;        // Number of available preamble sequences (default 64)
    uint8_t  maxAttempts;         // Maximum preamble transmission attempts
    double   backoffMs;           // Backoff window in ms
    double   rarWindowMs;         // RAR window size
    int      preambleRepetitions; // Number of preamble repetitions per CE level
};

static const NPRACHConfig DEFAULT_NPRACH_CONFIG = {
    0,      // format 0
    5.6,    // 5.6 ms per preamble
    64,     // 64 preamble sequences
    5,      // max 5 attempts (was 10, too many)
    20.0,   // 20 ms backoff
    5.0,    // 5 ms RAR window
    1       // 1 repetition (CE0)
};

// ============================================================================
// NB-IoT CURRENT CONSUMPTION (based on u-blox SARA-N2 / Quectel BC95)
// ============================================================================

struct CurrentConsumption {
    double psm_uA;              // PSM sleep current
    double edrx_sleep_uA;      // eDRX sleep current
    double edrx_paging_mA;     // eDRX paging window current
    double idle_drx_mA;        // RRC_IDLE DRX current
    double rach_mA;            // RACH procedure current
    double connected_rx_mA;    // Connected mode RX current
    double connected_idle_mA;  // Connected mode idle current
    double supply_voltage_V;   // Supply voltage
};

// Typical current consumption values
static const CurrentConsumption TYPICAL_CURRENT = {
    3.0,     // PSM: 3 μA
    6.0,     // eDRX sleep: 6 μA
    46.0,    // eDRX paging: 46 mA
    6.0,     // Idle DRX: 6 mA
    80.0,    // RACH: 80 mA
    46.0,    // Connected RX: 46 mA
    6.0,     // Connected idle: 6 mA
    3.6,     // Supply voltage
};

// TX current as function of TX power (based on datasheet measurements)
inline double GetTxCurrent_mA(double txPower_dBm) {
    if (txPower_dBm >= 23.0) return 220.0;      // 220 mA at 23 dBm
    if (txPower_dBm >= 20.0) return 150.0;      // 150 mA at 20 dBm
    if (txPower_dBm >= 17.0) return 110.0;      // 110 mA at 17 dBm
    if (txPower_dBm >= 14.0) return 80.0;       // 80 mA at 14 dBm
    if (txPower_dBm >= 10.0) return 55.0;       // 55 mA at 10 dBm
    if (txPower_dBm >= 5.0)  return 40.0;       // 40 mA at 5 dBm
    return 30.0;                                 // 30 mA at 0 dBm
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Convert dBm to Watts
inline double DbmToW(double dBm) {
    return std::pow(10.0, (dBm - 30.0) / 10.0);
}

// Convert Watts to dBm
inline double WToDbm(double w) {
    return 10.0 * std::log10(w) + 30.0;
}

// Calculate SINR given signal power and interference+noise
inline double CalculateSINR_dB(double signalPower_dBm, double interferenceNoise_dBm) {
    double signalW = DbmToW(signalPower_dBm);
    double interferenceW = DbmToW(interferenceNoise_dBm);
    return 10.0 * std::log10(signalW / interferenceW);
}

// Calculate thermal noise power for given bandwidth
inline double ThermalNoisePower_dBm(double bandwidth_Hz) {
    // N = kTB, k = 1.38e-23, T = 290K
    double noiseW = 1.38e-23 * 290.0 * bandwidth_Hz;
    return WToDbm(noiseW);
}

// NB-IoT thermal noise for 180 kHz bandwidth
inline double NbiotThermalNoise_dBm() {
    return ThermalNoisePower_dBm(NBIOT_BANDWIDTH_HZ); // ≈ -128.4 dBm
}

// Calculate airtime for NB-IoT uplink transmission
inline double CalculateAirtime_ms(uint8_t mcs, int numRepetitions, ResourceUnit ru, uint32_t payloadBytes) {
    uint8_t ruIdx = static_cast<uint8_t>(ru);
    double ruDuration = RU_CONFIGS[ruIdx].durationMs;
    
    // Number of RUs needed for payload
    int tbsIdx = MCS_TABLE[mcs].tbsIndex;
    int payloadBits = payloadBytes * 8;
    
    // Find minimum number of RUs needed
    int nRU = 1;
    for (int i = 0; i < 8; i++) {
        if (TBS_TABLE_SINGLE_TONE[tbsIdx][i] >= payloadBits) {
            nRU = i + 1;
            break;
        }
        if (i == 7) nRU = 10; // Maximum
    }
    
    // Total airtime = RU duration × number of RUs × repetitions
    return ruDuration * nRU * numRepetitions;
}

// Get optimal MCS for given SINR (without repetitions)
inline uint8_t GetOptimalMCS(double sinr_dB) {
    for (int mcs = NUM_MCS_LEVELS - 1; mcs >= 0; mcs--) {
        if (sinr_dB >= MCS_TABLE[mcs].requiredSINR_dB + 2.0) { // 2 dB margin
            return mcs;
        }
    }
    return 0; // Fallback to most robust MCS
}

// Get effective SINR with repetition gain
inline double GetEffectiveSINR_dB(double sinr_dB, int numRepetitions) {
    return sinr_dB + GetRepetitionGainDb(numRepetitions);
}

// Check if transmission can succeed with given parameters
inline bool CanDecodeWithParams(double sinr_dB, uint8_t mcs, int numRepetitions) {
    double effectiveSINR = GetEffectiveSINR_dB(sinr_dB, numRepetitions);
    return effectiveSINR >= MCS_TABLE[mcs].requiredSINR_dB;
}

// Determine coverage enhancement level from coupling loss
inline CELevel DetermineCELevel(double couplingLoss_dB) {
    if (couplingLoss_dB <= 144.0) return CELevel::CE0;
    if (couplingLoss_dB <= 154.0) return CELevel::CE1;
    return CELevel::CE2;
}

// Get maximum allowed repetitions for a CE level
inline int GetMaxRepetitions(CELevel level) {
    return CE_CONFIGS[static_cast<int>(level)].maxRepetitions;
}

// Calculate RACH collision probability (simplified)
// Based on the number of contending UEs and available preamble sequences
inline double CalculateRACHCollisionProb(uint32_t numContendingUEs, uint32_t numPreambles = 64) {
    if (numContendingUEs <= 1) return 0.0;
    // Probability that at least 2 UEs choose the same preamble
    // P(collision) ≈ 1 - (1 - 1/N)^(K-1) where N=preambles, K=contending UEs
    double pNoCollision = std::pow(1.0 - 1.0 / numPreambles, numContendingUEs - 1);
    return 1.0 - pNoCollision;
}

// Packet record for tracking (equivalent to LoRaWAN PacketRecord)
struct NbiotPacketRecord {
    double   timestamp;
    uint32_t ueId;
    uint32_t sequenceNum;
    double   txPowerDbm;
    uint8_t  mcs;
    int      repetitions;
    ResourceUnit ru;
    double   rsrp_dBm;        // Reference Signal Received Power
    double   rsrq_dB;         // Reference Signal Received Quality
    double   sinr_dB;         // Signal-to-Interference-plus-Noise Ratio
    double   preTxRssi_dBm;   // Pre-transmission RSSI measurement
    bool     rachSuccess;     // RACH procedure success
    int      rachAttempts;    // Number of RACH attempts needed
    bool     dataSuccess;     // Data transmission success
    double   airtime_ms;      // Total airtime including repetitions
    double   energyConsumed_mJ; // Energy for this transmission cycle
    CELevel  ceLevel;         // Coverage enhancement level
    double   couplingLoss_dB; // Path loss + antenna gains
    uint32_t payloadBytes;
    double   cellLoad;        // eNB cell load at time of transmission
    bool     collisionFlag;   // RACH collision detected
};

} // namespace nbiot
} // namespace ns3

#endif // NBIOT_COMMON_H
