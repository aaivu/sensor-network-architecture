#include "ns3/command-line.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TestSimple");

int main(int argc, char* argv[]) {
    LogComponentEnable("TestSimple", LOG_LEVEL_INFO);
    
    CommandLine cmd;
    cmd.Parse(argc, argv);
    
    NS_LOG_INFO("Test simulation starting");
    
    Simulator::Stop(Seconds(1.0));
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Test simulation completed");
    return 0;
}