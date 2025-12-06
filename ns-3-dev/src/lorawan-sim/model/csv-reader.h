#ifndef CSV_READER_H
#define CSV_READER_H

#include "ns3/core-module.h"
#include "environment.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace ns3 {
namespace lorawan {

/**
 * Structure to hold node position data from CSV
 */
struct NodePositionData {
    uint32_t nodeId;
    double x;
    double y;
    double z;
};

/**
 * Structure to hold obstacle data from CSV
 */
struct ObstacleData {
    double x;
    double y;
    double width;
    double height;
    double attenuationDb;
};

/**
 * Topology Loader utility class for loading simulation data from CSV files
 */
class TopologyLoader {
public:
    /**
     * Load node positions from CSV file
     * 
     * Expected CSV format:
     * nodeId,x,y,z
     * 0,100.5,200.3,1.5
     * 1,-50.2,150.7,1.5
     * ...
     * 
     * \param filePath Path to the CSV file
     * \return Vector of NodePositionData structures
     */
    static std::vector<NodePositionData> LoadNodePositions(const std::string& filePath);
    
    /**
     * Load obstacle positions from CSV file
     * 
     * Expected CSV format:
     * x,y,width,height,attenuationDb
     * 100.0,200.0,50.0,30.0,10.0
     * -150.0,300.0,60.0,40.0,12.0
     * ...
     * 
     * \param filePath Path to the CSV file
     * \return Vector of ObstacleData structures
     */
    static std::vector<ObstacleData> LoadObstacles(const std::string& filePath);
    
    /**
     * Check if a file exists
     * 
     * \param filePath Path to check
     * \return true if file exists
     */
    static bool FileExists(const std::string& filePath);
    
private:
    /**
     * Split a string by delimiter
     */
    static std::vector<std::string> Split(const std::string& s, char delimiter);
    
    /**
     * Trim whitespace from string
     */
    static std::string Trim(const std::string& str);
};

} // namespace lorawan
} // namespace ns3

#endif // CSV_READER_H
