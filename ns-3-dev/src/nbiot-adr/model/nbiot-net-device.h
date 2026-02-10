/*
 * nbiot-net-device.h
 *
 * NB-IoT UE and eNB model classes.
 * UE: State machine (PSM/eDRX/Connected), uplink transmission
 * eNB: Scheduler, RACH handling, cell load tracking
 */

#ifndef NBIOT_NET_DEVICE_H
#define NBIOT_NET_DEVICE_H

#include "nbiot-common.h"
#include "nbiot-channel.h"

#include "ns3/object.h"
#include "ns3/node.h"
#include "ns3/nstime.h"
#include "ns3/vector.h"
#include "ns3/callback.h"
#include "ns3/traced-value.h"
#include "ns3/random-variable-stream.h"

#include <deque>
#include <map>
#include <vector>
#include <functional>

namespace ns3 {
namespace nbiot {

/**
 * NB-IoT UE Network Device
 * Models an NB-IoT user equipment with:
 *   - Power saving state machine (PSM, eDRX, Connected)
 *   - RACH procedure
 *   - Uplink data transmission with MCS, repetitions, RU selection
 *   - Power control
 *   - Battery/energy tracking
 */
class NbiotUeDevice : public Object {
public:
    static TypeId GetTypeId();
    NbiotUeDevice();
    virtual ~NbiotUeDevice();

    // Configuration
    void SetNode(Ptr<Node> node);
    Ptr<Node> GetNode() const;
    void SetChannel(Ptr<NbiotChannel> channel);
    void SetUeId(uint32_t id);
    uint32_t GetUeId() const;

    // Transmission parameters (the "action space" for RL)
    void SetMCS(uint8_t mcs);
    uint8_t GetMCS() const;
    void SetRepetitions(int reps);
    int GetRepetitions() const;
    void SetResourceUnit(ResourceUnit ru);
    ResourceUnit GetResourceUnit() const;
    void SetTxPower(double power_dBm);
    double GetTxPower() const;
    void SetPowerClass(PowerClass pc);

    // PSM / eDRX configuration
    void SetPSMActiveTimer(double seconds);
    void SetPSMTauTimer(double seconds);
    void SetEDRXCycle(double seconds);

    // State
    UEState GetState() const;
    CELevel GetCELevel() const;
    void SetCELevel(CELevel level);

    // Metrics
    double GetBatteryLevel() const;         // 0.0 - 1.0
    double GetTotalEnergyConsumed_mJ() const;
    uint32_t GetTotalPacketsSent() const;
    uint32_t GetTotalPacketsDelivered() const;
    double GetPDR() const;

    // Transmission interface
    struct TxResult {
        bool     rachSuccess;
        int      rachAttempts;
        bool     dataSuccess;
        double   sinr_dB;
        double   rsrp_dBm;
        double   rsrq_dB;
        double   energyConsumed_mJ;
        double   airtime_ms;
        double   couplingLoss_dB;
    };

    // Callbacks
    typedef Callback<void, uint32_t, const TxResult&> TxCompleteCallback;
    void SetTxCompleteCallback(TxCompleteCallback cb);

private:
    Ptr<Node>          m_node;
    Ptr<NbiotChannel>  m_channel;
    uint32_t           m_ueId;

    // Current transmission parameters
    uint8_t      m_mcs;
    int          m_repetitions;
    ResourceUnit m_ru;
    double       m_txPower_dBm;
    PowerClass   m_powerClass;

    // State machine
    UEState m_state;
    CELevel m_ceLevel;

    // PSM / eDRX timers
    double m_psmActiveTimer_s;
    double m_psmTauTimer_s;
    double m_edrxCycle_s;

    // Energy tracking
    double   m_totalEnergy_mJ;
    double   m_batteryCapacity_mAh;   // e.g., 2400 mAh
    double   m_supplyVoltage_V;

    // Packet tracking
    uint32_t m_packetsSent;
    uint32_t m_packetsDelivered;

    // Callback
    TxCompleteCallback m_txCompleteCb;
};


/**
 * NB-IoT eNB (Base Station) Model
 * Models the eNB with:
 *   - Round-robin scheduler for NPUSCH
 *   - NPRACH resource management and collision handling
 *   - Cell load tracking
 *   - Downlink signaling (simplified)
 */
class NbiotEnbDevice : public Object {
public:
    static TypeId GetTypeId();
    NbiotEnbDevice();
    virtual ~NbiotEnbDevice();

    // Configuration
    void SetNode(Ptr<Node> node);
    Ptr<Node> GetNode() const;
    void SetChannel(Ptr<NbiotChannel> channel);
    void SetNoiseFigure(double nf_dB);
    void SetTxPower(double power_dBm);

    // RACH handling
    struct RACHResult {
        bool success;
        int  attemptsUsed;
        bool collisionDetected;
    };
    RACHResult ProcessRACH(uint32_t ueId, double preamblePower_dBm,
                           CELevel ceLevel, Time currentTime);

    // Uplink reception
    struct UplinkResult {
        bool   success;
        double sinr_dB;
        double rsrp_dBm;
        double rsrq_dB;
    };
    UplinkResult ProcessUplink(uint32_t ueId, double rxPower_dBm,
                               uint8_t mcs, int repetitions,
                               ResourceUnit ru, Time currentTime);

    // Cell load
    double GetCellLoad() const;  // 0.0 - 1.0
    uint32_t GetNumConnectedUEs() const;
    void RegisterUE(uint32_t ueId);
    void DeregisterUE(uint32_t ueId);

    // Scheduling
    struct ScheduleGrant {
        uint8_t      assignedRU;
        uint8_t      assignedMCS;
        int          assignedRepetitions;
        double       scheduledTime_ms;
    };
    ScheduleGrant ScheduleUplink(uint32_t ueId, uint8_t requestedMCS,
                                  int requestedReps, ResourceUnit requestedRU);

private:
    Ptr<Node>          m_node;
    Ptr<NbiotChannel>  m_channel;
    double             m_noiseFigure_dB;
    double             m_txPower_dBm;

    // Registered UEs
    std::vector<uint32_t> m_registeredUEs;
    std::map<uint32_t, Time> m_ueLastActivity;

    // RACH tracking
    std::map<uint32_t, int> m_rachAttemptCount;
    uint32_t m_currentRachContenders;

    // Scheduler state
    uint32_t m_schedulerIndex;     // Round-robin index
    double   m_totalResourcesUsed; // Resource utilization tracking

    // Cell load
    double m_cellLoad;
};

} // namespace nbiot
} // namespace ns3

#endif // NBIOT_NET_DEVICE_H
