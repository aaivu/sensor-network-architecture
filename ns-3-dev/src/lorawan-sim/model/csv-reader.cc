#include "csv-reader.h"
#include "ns3/log.h"
#include <algorithm>
#include <iostream>

namespace ns3 {
namespace lorawan {

NS_LOG_COMPONENT_DEFINE("TopologyLoader");

std::vector<std::string> TopologyLoader::Split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(Trim(token));
    }
    return tokens;
}

std::string TopologyLoader::Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

bool TopologyLoader::FileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

std::vector<NodePositionData> TopologyLoader::LoadNodePositions(const std::string& filePath) {
    std::vector<NodePositionData> positions;
    
    if (!FileExists(filePath)) {
        NS_LOG_WARN("Node positions CSV file not found: " << filePath);
        return positions;
    }
    
    std::ifstream file(filePath);
    std::string line;
    bool isHeader = true;
    uint32_t lineNumber = 0;
    
    NS_LOG_INFO("Loading node positions from: " << filePath);
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (Trim(line).empty()) {
            continue;
        }
        
        // Skip header line
        if (isHeader) {
            isHeader = false;
            NS_LOG_DEBUG("Header: " << line);
            continue;
        }
        
        // Parse CSV line
        std::vector<std::string> tokens = Split(line, ',');
        
        if (tokens.size() < 4) {
            NS_LOG_WARN("Line " << lineNumber << " has insufficient columns (expected 4): " << line);
            continue;
        }
        
        try {
            NodePositionData data;
            data.nodeId = std::stoul(tokens[0]);
            data.x = std::stod(tokens[1]);
            data.y = std::stod(tokens[2]);
            data.z = std::stod(tokens[3]);
            
            positions.push_back(data);
            
            NS_LOG_DEBUG("Loaded node " << data.nodeId << " at (" 
                        << data.x << ", " << data.y << ", " << data.z << ")");
        }
        catch (const std::exception& e) {
            NS_LOG_WARN("Error parsing line " << lineNumber << ": " << e.what());
            continue;
        }
    }
    
    file.close();
    NS_LOG_INFO("Loaded " << positions.size() << " node positions from CSV");
    
    return positions;
}

std::vector<ObstacleData> TopologyLoader::LoadObstacles(const std::string& filePath) {
    std::vector<ObstacleData> obstacles;
    
    if (!FileExists(filePath)) {
        NS_LOG_WARN("Obstacles CSV file not found: " << filePath);
        return obstacles;
    }
    
    std::ifstream file(filePath);
    std::string line;
    bool isHeader = true;
    uint32_t lineNumber = 0;
    
    NS_LOG_INFO("Loading obstacles from: " << filePath);
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (Trim(line).empty()) {
            continue;
        }
        
        // Skip header line
        if (isHeader) {
            isHeader = false;
            NS_LOG_DEBUG("Header: " << line);
            continue;
        }
        
        // Parse CSV line
        std::vector<std::string> tokens = Split(line, ',');
        
        if (tokens.size() < 5) {
            NS_LOG_WARN("Line " << lineNumber << " has insufficient columns (expected 5): " << line);
            continue;
        }
        
        try {
            ObstacleData data;
            data.x = std::stod(tokens[0]);
            data.y = std::stod(tokens[1]);
            data.width = std::stod(tokens[2]);
            data.height = std::stod(tokens[3]);
            data.attenuationDb = std::stod(tokens[4]);
            
            obstacles.push_back(data);
            
            NS_LOG_DEBUG("Loaded obstacle at (" << data.x << ", " << data.y 
                        << ") size: " << data.width << "x" << data.height 
                        << " attenuation: " << data.attenuationDb << " dB");
        }
        catch (const std::exception& e) {
            NS_LOG_WARN("Error parsing line " << lineNumber << ": " << e.what());
            continue;
        }
    }
    
    file.close();
    NS_LOG_INFO("Loaded " << obstacles.size() << " obstacles from CSV");
    
    return obstacles;
}

} // namespace lorawan
} // namespace ns3
