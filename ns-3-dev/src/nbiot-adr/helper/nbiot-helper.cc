/*
 * nbiot-helper.cc
 *
 * NB-IoT helper implementation.
 */

#include "nbiot-helper.h"

#include "ns3/log.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/string.h"
#include "ns3/double.h"

namespace ns3 {
namespace nbiot {

NS_LOG_COMPONENT_DEFINE("NbiotHelper");

NbiotHelper::NbiotHelper()
    : m_uePowerClass(PowerClass::CLASS_3),
      m_enbTxPower_dBm(46.0),
      m_enbNoiseFigure_dB(5.0),
      m_interCellDistance_m(1732.0),
      m_numNeighborCells(6)
{
}

NbiotHelper::~NbiotHelper() {}

Ptr<NbiotChannel>
NbiotHelper::CreateChannel(const std::string& scenario)
{
    Ptr<NbiotChannel> channel = CreateObject<NbiotChannel>();

    // Create propagation loss model based on scenario
    // Using log-distance model configured for NB-IoT bands (900 MHz)
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();

    if (scenario == "urban_macro") {
        // 3GPP Urban Macro for 900 MHz
        // Reference: TR 36.942, Hata-COST231 model adapted
        loss->SetAttribute("Exponent", DoubleValue(3.76));
        loss->SetAttribute("ReferenceLoss", DoubleValue(32.45)); // Free space at 1m, 900 MHz
        loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));

        // Add log-normal shadowing (8 dB standard deviation)
        Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
        shadowing->SetAttribute("Variable",
            StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=64]"));
        loss->SetNext(shadowing);

    } else if (scenario == "rural_macro") {
        // Rural Macro: lower exponent, less shadowing
        loss->SetAttribute("Exponent", DoubleValue(3.2));
        loss->SetAttribute("ReferenceLoss", DoubleValue(32.45));
        loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));

        Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
        shadowing->SetAttribute("Variable",
            StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=36]")); // 6 dB std dev
        loss->SetNext(shadowing);

    } else if (scenario == "indoor") {
        // Indoor scenario: higher exponent + penetration loss
        loss->SetAttribute("Exponent", DoubleValue(4.0));
        loss->SetAttribute("ReferenceLoss", DoubleValue(32.45));
        loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));

        Ptr<RandomPropagationLossModel> shadowing = CreateObject<RandomPropagationLossModel>();
        shadowing->SetAttribute("Variable",
            StringValue("ns3::NormalRandomVariable[Mean=20.0|Variance=100]")); // 20 dB mean penetration
        loss->SetNext(shadowing);
    }

    Ptr<ConstantSpeedPropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();

    channel->SetPropagationLossModel(loss);
    channel->SetPropagationDelayModel(delay);
    channel->SetInterCellDistance(m_interCellDistance_m);
    channel->SetNumNeighborCells(m_numNeighborCells);
    channel->SetEnableTimeFading(true);

    return channel;
}

std::vector<Ptr<NbiotUeDevice>>
NbiotHelper::InstallUeDevices(Ptr<NbiotChannel> channel, NodeContainer ueNodes)
{
    std::vector<Ptr<NbiotUeDevice>> devices;

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<NbiotUeDevice> ue = CreateObject<NbiotUeDevice>();
        ue->SetNode(ueNodes.Get(i));
        ue->SetChannel(channel);
        ue->SetUeId(ueNodes.Get(i)->GetId());
        ue->SetPowerClass(m_uePowerClass);
        ue->SetTxPower(POWER_CLASSES[static_cast<int>(m_uePowerClass)].maxPower_dBm);
        ue->SetMCS(0);           // Start with most robust MCS
        ue->SetRepetitions(1);   // Start with no repetitions
        ue->SetResourceUnit(ResourceUnit::SINGLE_TONE_15KHZ); // Default RU

        devices.push_back(ue);
    }

    return devices;
}

Ptr<NbiotEnbDevice>
NbiotHelper::InstallEnbDevice(Ptr<NbiotChannel> channel, Ptr<Node> enbNode)
{
    Ptr<NbiotEnbDevice> enb = CreateObject<NbiotEnbDevice>();
    enb->SetNode(enbNode);
    enb->SetChannel(channel);
    enb->SetTxPower(m_enbTxPower_dBm);
    enb->SetNoiseFigure(m_enbNoiseFigure_dB);

    return enb;
}

Ptr<NbiotEnergyModel>
NbiotHelper::CreateEnergyModel()
{
    return CreateObject<NbiotEnergyModel>();
}

void NbiotHelper::SetUePowerClass(PowerClass pc) { m_uePowerClass = pc; }
void NbiotHelper::SetEnbTxPower(double power_dBm) { m_enbTxPower_dBm = power_dBm; }
void NbiotHelper::SetEnbNoiseFigure(double nf_dB) { m_enbNoiseFigure_dB = nf_dB; }
void NbiotHelper::SetInterCellDistance(double distance_m) { m_interCellDistance_m = distance_m; }
void NbiotHelper::SetNumNeighborCells(uint32_t nCells) { m_numNeighborCells = nCells; }

} // namespace nbiot
} // namespace ns3
