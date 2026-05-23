// -*- mode: c++; -*-
#pragma once
// Device History: DeviceHistory, CollisionIndicators, ChannelSelector + channelSelector instance

struct DeviceHistory {
    std::deque<double> rssiHistory;      // Last 20 RSSI readings
    std::deque<double> snrHistory;       // Last 20 SNR values
    std::deque<bool> packetResults;      // Last 20 packet outcomes
    std::deque<double> timestamps;       // Timestamps of packets
    uint32_t consecutiveLosses = 0;
    double lastSuccessTime = 0.0;
    uint8_t currentChannel = 0;
    uint8_t previousChannel = 0;
    
    void addRssi(double rssi) {
        rssiHistory.push_back(rssi);
        if (rssiHistory.size() > 20) rssiHistory.pop_front();
    }
    
    void addSnr(double snr) {
        snrHistory.push_back(snr);
        if (snrHistory.size() > 20) snrHistory.pop_front();
    }
    
    void addPacketResult(bool success, double timestamp) {
        packetResults.push_back(success);
        timestamps.push_back(timestamp);
        if (packetResults.size() > 20) {
            packetResults.pop_front();
            timestamps.pop_front();
        }
        
        if (success) {
            consecutiveLosses = 0;
            lastSuccessTime = timestamp;
        } else {
            consecutiveLosses++;
        }
    }
    
    void setChannel(uint8_t channel) {
        previousChannel = currentChannel;
        currentChannel = channel;
    }
    
    double getRssiVariance() const {
        if (rssiHistory.size() < 2) return 0.0;
        double mean = 0.0;
        for (double r : rssiHistory) mean += r;
        mean /= rssiHistory.size();
        
        double variance = 0.0;
        for (double r : rssiHistory) variance += (r - mean) * (r - mean);
        return variance / rssiHistory.size();
    }
    
    double getSnrTrend() const {
        if (snrHistory.size() < 5) return 0.0;
        // Compare recent average to older average
        double recentAvg = 0.0, oldAvg = 0.0;
        size_t mid = snrHistory.size() / 2;
        for (size_t i = 0; i < mid; i++) oldAvg += snrHistory[i];
        for (size_t i = mid; i < snrHistory.size(); i++) recentAvg += snrHistory[i];
        oldAvg /= mid;
        recentAvg /= (snrHistory.size() - mid);
        return recentAvg - oldAvg;
    }
    
    double getRecentPdr() const {
        if (packetResults.empty()) return 0.5;
        int successes = std::count(packetResults.begin(), packetResults.end(), true);
        return (double)successes / packetResults.size();
    }
    
    double getPdrTrend() const {
        if (packetResults.size() < 10) return 0.0;
        // Compare last 5 to previous 5
        int recent = 0, old = 0;
        size_t n = packetResults.size();
        for (size_t i = n - 5; i < n; i++) recent += packetResults[i] ? 1 : 0;
        for (size_t i = n - 10; i < n - 5; i++) old += packetResults[i] ? 1 : 0;
        return (recent - old) / 5.0;
    }
    
    double getLastRssi() const {
        return rssiHistory.empty() ? -110.0 : rssiHistory.back();
    }
    
    double getLastSnr() const {
        return snrHistory.empty() ? 0.0 : snrHistory.back();
    }
};

// Global storage for device histories
std::map<uint32_t, DeviceHistory> deviceHistories;

/**
 * COLLISION DETECTION HEURISTIC
 * Infers collision probability from observable metrics
 */
struct CollisionIndicators {
    double collisionProbability;
    bool likelyCollision;
    bool likelyWeakSignal;
    bool likelyInterference;
    
    static CollisionIndicators analyze(
        double preTxRssi,
        double rxSnr,
        bool packetSuccess,
        double rssiVariance,
        uint32_t consecutiveLosses,
        double recentPdr)
    {
        CollisionIndicators result;
        result.collisionProbability = 0.0;
        result.likelyCollision = false;
        result.likelyWeakSignal = false;
        result.likelyInterference = false;
        
        if (!packetSuccess) {
            // Case 1: High pre-TX RSSI + packet loss = likely collision
            if (preTxRssi > -105.0) {
                result.collisionProbability += 0.4;
                result.likelyCollision = true;
            }
            
            // Case 2: High RSSI variance + loss = interference
            if (rssiVariance > 8.0) {
                result.collisionProbability += 0.3;
                result.likelyInterference = true;
            }
            
            // Case 3: Good SNR history but sudden loss = collision
            if (rxSnr > 5.0 && consecutiveLosses < 3) {
                result.collisionProbability += 0.2;
                result.likelyCollision = true;
            }
            
            // Case 4: Low SNR = weak signal, not collision
            if (rxSnr < -5.0) {
                result.likelyWeakSignal = true;
                result.collisionProbability *= 0.5;
            }
            
            // Case 5: Pattern of intermittent losses = periodic interference
            if (recentPdr > 0.3 && recentPdr < 0.7 && rssiVariance > 5.0) {
                result.likelyInterference = true;
            }
        }
        
        result.collisionProbability = std::min(1.0, result.collisionProbability);
        return result;
    }
};

/**
 * CHANNEL SELECTOR - Tracks interference per channel and recommends best channel
 */
class ChannelSelector {
private:
    static const int NUM_CHANNELS = 8;
    
    struct ChannelStats {
        std::deque<double> rssiHistory;
        std::deque<bool> successHistory;
        double avgRssi = -110.0;
        double successRate = 0.5;
        double lastUsedTime = 0.0;
        
        void addSample(double rssi, bool success, double timestamp) {
            rssiHistory.push_back(rssi);
            successHistory.push_back(success);
            lastUsedTime = timestamp;
            
            if (rssiHistory.size() > 10) {
                rssiHistory.pop_front();
                successHistory.pop_front();
            }
            
            // Update averages
            avgRssi = 0.0;
            int successes = 0;
            for (size_t i = 0; i < rssiHistory.size(); i++) {
                avgRssi += rssiHistory[i];
                if (successHistory[i]) successes++;
            }
            avgRssi /= rssiHistory.size();
            successRate = (double)successes / successHistory.size();
        }
    };
    
    std::map<uint32_t, std::array<ChannelStats, NUM_CHANNELS>> deviceChannelStats;
    
public:
    void recordChannelUsage(uint32_t nodeId, uint8_t channel, double rssi, bool success, double timestamp) {
        if (channel >= NUM_CHANNELS) return;
        deviceChannelStats[nodeId][channel].addSample(rssi, success, timestamp);
    }
    
    uint8_t getBestChannel(uint32_t nodeId, double currentTime) {
        auto& stats = deviceChannelStats[nodeId];
        
        uint8_t bestChannel = 0;
        double bestScore = -1000.0;
        
        for (uint8_t ch = 0; ch < NUM_CHANNELS; ch++) {
            // Score = success rate + (lower RSSI is better) + time diversity
            double timeSinceUsed = currentTime - stats[ch].lastUsedTime;
            double timeBonus = std::min(1.0, timeSinceUsed / 60.0);
            
            double score = stats[ch].successRate * 100.0 
                         - (stats[ch].avgRssi + 110.0)
                         + timeBonus * 10.0;
            
            if (score > bestScore) {
                bestScore = score;
                bestChannel = ch;
            }
        }
        
        return bestChannel;
    }
    
    bool shouldChangeChannel(uint32_t nodeId, uint8_t currentChannel) {
        auto& stats = deviceChannelStats[nodeId];
        
        if (stats[currentChannel].successHistory.size() < 5) return false;
        
        if (stats[currentChannel].successRate < 0.4) {
            return true;
        }
        
        if (stats[currentChannel].avgRssi > -100.0) {
            return true;
        }
        
        return false;
    }
};

// Global channel selector instance
ChannelSelector channelSelector;

