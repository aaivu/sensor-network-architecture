/*
 * lorawan-sim-example.cc
 * 
 * Clean example using the refactored LoRaWAN simulation module
 */

#include "ns3/command-line.h"
#include "ns3/log.h"
#include "ns3/simulation-runner.h"
#include <iostream>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("LoRaWANSimExample");

int main(int argc, char* argv[]) {
    uint32_t nDevices = 10;
    double simulationTime = 300.0;
    double appPeriodSeconds = 30.0;
    double radius = 1000.0;
    std::string csvFileName = "lorawan_sim_dataset.csv";
    std::string adrModeStr = "off";
    bool environmentalModeling = true;
    uint32_t nWifiInterferers = 12;
    std::string nodePositionsFile = "";
    std::string obstaclesFile = "";
    std::string outputDir = "lorawan_datasets";
    
    CommandLine cmd;
    cmd.AddValue("nDevices", "Number of end devices", nDevices);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Packet transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("radius", "Deployment radius in meters", radius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.AddValue("adr", "ADR mode: 'off', 'on', 'ddqn', or 'all'", adrModeStr);
    cmd.AddValue("environmental", "Enable environmental effects modeling", environmentalModeling);
    cmd.AddValue("wifiInterferers", "Number of WiFi interfering nodes", nWifiInterferers);
    cmd.AddValue("nodePositions", "CSV file with node positions (nodeId,x,y,z)", nodePositionsFile);
    cmd.AddValue("obstacles", "CSV file with obstacles (x,y,width,height,attenuationDb)", obstaclesFile);
    cmd.AddValue("outputDir", "Output directory for dataset files", outputDir);
    cmd.Parse(argc, argv);
    
    LogComponentEnable("LoRaWANSimExample", LOG_LEVEL_INFO);
    
    std::cout << "\n=== LoRaWAN Simulation Module ===" << std::endl;
    std::cout << "Environmental modeling: " << (environmentalModeling ? "ENABLED" : "DISABLED") << std::endl;
    if (environmentalModeling) {
        std::cout << "  - Urban obstacle modeling" << std::endl;
        std::cout << "  - WiFi interference modeling (" << nWifiInterferers << " nodes)" << std::endl;
        std::cout << "  - Hardware variability modeling" << std::endl;
    }
    
    if (adrModeStr == "all") {
        std::cout << "\n▶ Running simulation for NO ADR..." << std::endl;
        SimulationRunner simOff(nDevices, simulationTime, appPeriodSeconds, radius, 
                               "no_adr_" + csvFileName, ADRMethod::OFF, nWifiInterferers, 
                               environmentalModeling);
        simOff.SetOutputDirectory(outputDir);
        if (!nodePositionsFile.empty()) simOff.SetNodePositionsFile(nodePositionsFile);
        if (!obstaclesFile.empty()) simOff.SetObstaclesFile(obstaclesFile);
        simOff.Run();
        
        std::cout << "\n▶ Running simulation for CLASSICAL ADR..." << std::endl;
        SimulationRunner simOn(nDevices, simulationTime, appPeriodSeconds, radius, 
                              "adr_" + csvFileName, ADRMethod::ON, nWifiInterferers, 
                              environmentalModeling);
        simOn.SetOutputDirectory(outputDir);
        if (!nodePositionsFile.empty()) simOn.SetNodePositionsFile(nodePositionsFile);
        if (!obstaclesFile.empty()) simOn.SetObstaclesFile(obstaclesFile);
        simOn.Run();
        
        std::cout << "\n▶ Running simulation for DDQN-PER ADR..." << std::endl;
        SimulationRunner simDDQN(nDevices, simulationTime, appPeriodSeconds, radius, 
                                "ddqn_adr_" + csvFileName, ADRMethod::DDQN, nWifiInterferers, 
                                environmentalModeling);
        simDDQN.SetOutputDirectory(outputDir);
        if (!nodePositionsFile.empty()) simDDQN.SetNodePositionsFile(nodePositionsFile);
        if (!obstaclesFile.empty()) simDDQN.SetObstaclesFile(obstaclesFile);
        simDDQN.Run();
    } else {
        ADRMethod adrMethod;
        std::string filePrefix;
        if (adrModeStr == "on") {
            adrMethod = ADRMethod::ON;
            filePrefix = "adr_";
        } else if (adrModeStr == "ddqn") {
            adrMethod = ADRMethod::DDQN;
            filePrefix = "ddqn_adr_";
        } else {
            adrMethod = ADRMethod::OFF;
            filePrefix = "no_adr_";
        }
        
        SimulationRunner sim(nDevices, simulationTime, appPeriodSeconds, radius, 
                           filePrefix + csvFileName, adrMethod, nWifiInterferers, 
                           environmentalModeling);
        
        // Set output directory
        sim.SetOutputDirectory(outputDir);
        
        // Set CSV files if provided
        if (!nodePositionsFile.empty()) {
            std::cout << "📍 Using node positions from: " << nodePositionsFile << std::endl;
            sim.SetNodePositionsFile(nodePositionsFile);
        }
        if (!obstaclesFile.empty()) {
            std::cout << "🏢 Using obstacles from: " << obstaclesFile << std::endl;
            sim.SetObstaclesFile(obstaclesFile);
        }
        
        sim.Run();
    }
    
    return 0;
}
