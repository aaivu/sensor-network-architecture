/*
 * advanced-ddqn-per-adr-nbiot.cc
 *
 * NB-IoT Adaptive Link Adaptation using Reinforcement Learning.
 * Mirrors the LoRaWAN DDQN-PER ADR simulation but adapted for NB-IoT.
 *
 * This file implements:
 * 1. No link adaptation (baseline)
 * 2. Default NB-IoT link adaptation (3GPP standard)
 * 3. DDQN-PER based link adaptation
 * 4. PPO based link adaptation
 * 5. MARL (Multi-Agent RL) based link adaptation
 *
 * Key NB-IoT differences from LoRaWAN:
 * - MCS (0-10) instead of SF (7-12)
 * - Repetitions (1-128) for coverage enhancement
 * - Resource Unit (RU) allocation (1-tone to 12-tone)
 * - RACH procedure (scheduled access, not ALOHA)
 * - PSM/eDRX power saving modes
 * - eNB scheduler (central controller, not passive gateway)
 * - Licensed spectrum (inter-cell interference, not WiFi/LoRa collision)
 *
 * Based on 3GPP TS 36.211/36.213/36.321 for NB-IoT.
 */

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/position-allocator.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/random-walk-2d-mobility-model.h"
#include "ns3/rectangle.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simulator.h"
#include "ns3/string.h"

// NB-IoT module headers
#include "ns3/nbiot-channel.h"
#include "ns3/nbiot-common.h"
#include "ns3/nbiot-energy-model.h"
#include "ns3/nbiot-helper.h"
#include "ns3/nbiot-net-device.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <vector>

using namespace ns3;
using namespace ns3::nbiot;

NS_LOG_COMPONENT_DEFINE("AdvancedDDQNPERADRNbiotExample");

// ============================================================================
// DDQN-PER IMPLEMENTATION (Adapted for NB-IoT)
// ============================================================================

/**
 * Experience structure for Prioritized Experience Replay
 */
struct Experience
{
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> nextState;
    bool done;
    double priority;
    double tdError;

    Experience(const std::vector<double>& s,
               int a,
               double r,
               const std::vector<double>& ns,
               bool d,
               double p = 1.0)
        : state(s),
          action(a),
          reward(r),
          nextState(ns),
          done(d),
          priority(p),
          tdError(0.0)
    {
    }
};

/**
 * Simple Neural Network for Q-value approximation
 * Adapted for NB-IoT: INPUT=8, OUTPUT=18 (6 MCS × 3 power levels)
 */
class SimpleQNetwork
{
  private:
    static const int INPUT_SIZE = 8;
    static const int HIDDEN1_SIZE = 128;
    static const int HIDDEN2_SIZE = 64;
    static const int OUTPUT_SIZE = 18;

    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    std::vector<double> lastInput, lastHidden1, lastHidden2, lastOutput;
    double learningRate;

    double relu(double x)
    {
        return x > 0 ? x : 0;
    }

    double reluDerivative(double x)
    {
        return x > 0 ? 1.0 : 0.0;
    }

  public:
    SimpleQNetwork(double lr = 0.001)
        : learningRate(lr)
    {
        initializeWeights();
    }

    void initializeWeights()
    {
        std::mt19937 rng(42);

        // Xavier initialization
        auto xavier = [&](int fanIn, int fanOut) -> double {
            std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (fanIn + fanOut)));
            return dist(rng);
        };

        weights1.resize(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
        bias1.resize(HIDDEN1_SIZE, 0.0);
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                weights1[i][j] = xavier(INPUT_SIZE, HIDDEN1_SIZE);
            }
        }

        weights2.resize(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE));
        bias2.resize(HIDDEN2_SIZE, 0.0);
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN1_SIZE; j++)
            {
                weights2[i][j] = xavier(HIDDEN1_SIZE, HIDDEN2_SIZE);
            }
        }

        weights3.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN2_SIZE));
        bias3.resize(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                weights3[i][j] = xavier(HIDDEN2_SIZE, OUTPUT_SIZE);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input)
    {
        lastInput = input;

        // Layer 1
        lastHidden1.resize(HIDDEN1_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += weights1[i][j] * input[j];
            }
            lastHidden1[i] = relu(sum);
        }

        // Layer 2
        lastHidden2.resize(HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN1_SIZE; j++)
            {
                sum += weights2[i][j] * lastHidden1[j];
            }
            lastHidden2[i] = relu(sum);
        }

        // Output layer
        lastOutput.resize(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double sum = bias3[i];
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                sum += weights3[i][j] * lastHidden2[j];
            }
            lastOutput[i] = sum;
        }

        return lastOutput;
    }

    void updateWeights(const std::vector<double>& targetOutput)
    {
        // Output layer gradients
        std::vector<double> outputGrad(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            outputGrad[i] = targetOutput[i] - lastOutput[i];
        }

        // Update output weights
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                weights3[i][j] += learningRate * outputGrad[i] * lastHidden2[j];
            }
            bias3[i] += learningRate * outputGrad[i];
        }

        // Hidden2 gradients
        std::vector<double> hidden2Grad(HIDDEN2_SIZE);
        for (int j = 0; j < HIDDEN2_SIZE; j++)
        {
            double error = 0;
            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                error += outputGrad[i] * weights3[i][j];
            }
            hidden2Grad[j] = error * reluDerivative(lastHidden2[j]);
        }

        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN1_SIZE; j++)
            {
                weights2[i][j] += learningRate * hidden2Grad[i] * lastHidden1[j];
            }
            bias2[i] += learningRate * hidden2Grad[i];
        }

        // Hidden1 gradients
        std::vector<double> hidden1Grad(HIDDEN1_SIZE);
        for (int j = 0; j < HIDDEN1_SIZE; j++)
        {
            double error = 0;
            for (int i = 0; i < HIDDEN2_SIZE; i++)
            {
                error += hidden2Grad[i] * weights2[i][j];
            }
            hidden1Grad[j] = error * reluDerivative(lastHidden1[j]);
        }

        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                weights1[i][j] += learningRate * hidden1Grad[i] * lastInput[j];
            }
            bias1[i] += learningRate * hidden1Grad[i];
        }
    }

    void copyFrom(const SimpleQNetwork& other)
    {
        weights1 = other.weights1;
        bias1 = other.bias1;
        weights2 = other.weights2;
        bias2 = other.bias2;
        weights3 = other.weights3;
        bias3 = other.bias3;
    }
};

/**
 * Prioritized Experience Replay Buffer
 */
class PrioritizedReplayBuffer
{
  private:
    std::vector<Experience> buffer;
    std::vector<double> priorities;
    size_t maxSize, currentSize, position;
    double alpha, beta, epsilon;

  public:
    PrioritizedReplayBuffer(size_t maxSize,
                            double alpha = 0.6,
                            double beta = 0.4,
                            double epsilon = 1e-6)
        : maxSize(maxSize),
          currentSize(0),
          position(0),
          alpha(alpha),
          beta(beta),
          epsilon(epsilon)
    {
        buffer.reserve(maxSize);
        priorities.reserve(maxSize);
    }

    void add(const Experience& experience)
    {
        double maxPriority = 1.0;
        if (!priorities.empty())
        {
            maxPriority = *std::max_element(priorities.begin(), priorities.begin() + currentSize);
        }

        if (currentSize < maxSize)
        {
            buffer.push_back(experience);
            priorities.push_back(std::pow(maxPriority, alpha));
            currentSize++;
        }
        else
        {
            buffer[position] = experience;
            priorities[position] = std::pow(maxPriority, alpha);
        }
        position = (position + 1) % maxSize;
    }

    std::vector<Experience> sample(size_t batchSize,
                                   std::vector<size_t>& indices,
                                   std::vector<double>& weights)
    {
        std::vector<Experience> batch;
        indices.clear();
        weights.clear();

        if (currentSize < batchSize)
        {
            batchSize = currentSize;
        }

        double totalPriority = 0;
        for (size_t i = 0; i < currentSize; i++)
        {
            totalPriority += priorities[i];
        }

        double maxWeight = 0;
        std::vector<double> probabilities(currentSize);
        for (size_t i = 0; i < currentSize; i++)
        {
            probabilities[i] = priorities[i] / totalPriority;
        }

        // Sample based on priorities
        std::mt19937 rng(std::random_device{}());
        for (size_t i = 0; i < batchSize; i++)
        {
            double r = std::uniform_real_distribution<double>(0, 1)(rng);
            double cumSum = 0;
            size_t idx = 0;
            for (size_t j = 0; j < currentSize; j++)
            {
                cumSum += probabilities[j];
                if (cumSum >= r)
                {
                    idx = j;
                    break;
                }
            }

            batch.push_back(buffer[idx]);
            indices.push_back(idx);

            double weight = std::pow(currentSize * probabilities[idx], -beta);
            weights.push_back(weight);
            if (weight > maxWeight)
            {
                maxWeight = weight;
            }
        }

        // Normalize weights
        for (auto& w : weights)
        {
            w /= maxWeight;
        }

        return batch;
    }

    void updatePriorities(const std::vector<size_t>& indices, const std::vector<double>& tdErrors)
    {
        for (size_t i = 0; i < indices.size(); i++)
        {
            priorities[indices[i]] = std::pow(std::abs(tdErrors[i]) + epsilon, alpha);
        }
    }

    size_t size() const
    {
        return currentSize;
    }
};

/**
 * DDQN-PER Agent for NB-IoT Link Adaptation
 * Action space: 18 actions (6 MCS levels × 3 TX power levels)
 */
class DDQNPERADRAgent
{
  private:
    SimpleQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    double epsilon, epsilonDecay, epsilonMin, gamma;
    int targetUpdateFreq, updateCounter;

    // NB-IoT action space
    std::vector<int> mcsActions = {0, 2, 4, 6, 8, 10}; // MCS indices
    std::vector<int> tpActions = {14, 20, 23};         // TX power in dBm

    double pdrWeight, energyWeight;
    std::mt19937 rng;

    double getRequiredSINR(uint8_t mcs)
    {
        if (mcs > 10)
        {
            mcs = 10;
        }
        return MCS_TABLE[mcs].requiredSINR_dB;
    }

  public:
    DDQNPERADRAgent(double lr = 0.001,
                    double eps = 1.0,
                    double epsDecay = 0.995,
                    double epsMin = 0.01,
                    double gam = 0.95,
                    int targetFreq = 100,
                    double pdrW = 1.0,
                    double energyW = 0.5)
        : mainNetwork(lr),
          targetNetwork(lr),
          replayBuffer(10000),
          epsilon(eps),
          epsilonDecay(epsDecay),
          epsilonMin(epsMin),
          gamma(gam),
          targetUpdateFreq(targetFreq),
          updateCounter(0),
          pdrWeight(pdrW),
          energyWeight(energyW),
          rng(std::random_device{}())
    {
        targetNetwork.copyFrom(mainNetwork);
    }

    void increaseExploration()
    {
        epsilon = std::min(1.0, epsilon * 2.0);
    }

    int selectAction(const std::vector<double>& state)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon)
        {
            // Random exploration
            std::uniform_int_distribution<int> actionDist(0, 17);
            return actionDist(rng);
        }
        // Greedy: select action with highest Q-value
        auto qValues = mainNetwork.forward(state);
        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }

    void addExperience(const std::vector<double>& state,
                       int action,
                       double reward,
                       const std::vector<double>& nextState,
                       bool done)
    {
        replayBuffer.add(Experience(state, action, reward, nextState, done));
    }

    void train(size_t batchSize = 32)
    {
        if (replayBuffer.size() < batchSize)
        {
            return;
        }

        std::vector<size_t> indices;
        std::vector<double> isWeights;
        auto batch = replayBuffer.sample(batchSize, indices, isWeights);

        std::vector<double> tdErrors;
        for (size_t i = 0; i < batch.size(); i++)
        {
            auto& exp = batch[i];

            // DDQN: use main network for action selection, target for evaluation
            auto mainQNext = mainNetwork.forward(exp.nextState);
            int bestAction = std::distance(mainQNext.begin(),
                                           std::max_element(mainQNext.begin(), mainQNext.end()));

            auto targetQNext = targetNetwork.forward(exp.nextState);
            double targetValue = exp.done ? 0.0 : gamma * targetQNext[bestAction];
            double target = exp.reward + targetValue;

            auto currentQ = mainNetwork.forward(exp.state);
            double tdError = target - currentQ[exp.action];
            tdErrors.push_back(tdError);

            // Update current Q
            auto targetQ = currentQ;
            targetQ[exp.action] = target;
            mainNetwork.updateWeights(targetQ);
        }

        replayBuffer.updatePriorities(indices, tdErrors);

        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0)
        {
            targetNetwork.copyFrom(mainNetwork);
        }

        epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    }

    /**
     * Decode action to NB-IoT parameters
     * @return {MCS, TX Power in dBm}
     */
    std::pair<int, int> getParameters(int action)
    {
        int mcsIdx = action / 3;
        int tpIdx = action % 3;
        mcsIdx = std::min(mcsIdx, (int)mcsActions.size() - 1);
        tpIdx = std::min(tpIdx, (int)tpActions.size() - 1);
        return {mcsActions[mcsIdx], tpActions[tpIdx]};
    }

    /**
     * NB-IoT reward function
     * Balances reliability (PDR) vs energy efficiency
     * Includes RACH success and coverage enhancement considerations
     */
    double calculateAdvancedReward(uint32_t ueId,
                                   double pdr,
                                   double energyConsumption_mJ,
                                   double sinr,
                                   bool packetSuccess,
                                   int currentMCS,
                                   double currentTP,
                                   int repetitions,
                                   uint32_t packetsSent,
                                   bool rachSuccess,
                                   int rachAttempts)
    {
        double reward = 0.0;

        // Component 1: Packet delivery (primary objective)
        if (packetSuccess)
        {
            reward += 1.0;
            // Bonus for PDR maintenance
            if (pdr > 0.9)
            {
                reward += 0.3;
            }
            else if (pdr > 0.7)
            {
                reward += 0.1;
            }
        }
        else
        {
            reward -= 1.0;
            // Extra penalty for continued failures
            if (pdr < 0.3)
            {
                reward -= 0.5;
            }
        }

        // Component 2: RACH efficiency
        if (!rachSuccess)
        {
            reward -= 0.5; // RACH failure penalty
        }
        else if (rachAttempts > 3)
        {
            reward -= 0.1 * (rachAttempts - 3); // Penalize excessive RACH attempts
        }

        // Component 3: Energy efficiency (secondary objective)
        if (packetSuccess)
        {
            // Reward lower energy consumption
            double energyNorm = energyConsumption_mJ / 100.0; // Normalize
            reward += 0.3 * (1.0 - std::min(1.0, energyNorm));

            // Reward using fewer repetitions when SINR allows
            if (repetitions == 1 && sinr > 0.0)
            {
                reward += 0.2;
            }
        }

        // Component 4: MCS efficiency
        if (packetSuccess && currentMCS >= 4)
        {
            reward += 0.15; // Bonus for using higher MCS (more efficient)
        }

        // Component 5: SINR margin awareness
        double requiredSINR = MCS_TABLE[currentMCS].requiredSINR_dB;
        double effectiveSINR = sinr + GetRepetitionGainDb(repetitions);
        double margin = effectiveSINR - requiredSINR;

        if (margin > 15.0 && currentMCS < 10)
        {
            reward -= 0.1; // Penalize over-provisioning (could use higher MCS)
        }
        else if (margin < 2.0 && margin > 0.0)
        {
            reward += 0.1; // Good: operating near optimal efficiency
        }

        return reward;
    }

    double getEpsilon() const
    {
        return epsilon;
    }
};

// ============================================================================
// ENHANCED DDQN COMPONENTS FOR NB-IoT
// ============================================================================

/**
 * Device History - tracks observable metrics for realistic state representation
 * Adapted for NB-IoT: RSRP instead of RSSI, SINR instead of SNR,
 * plus RACH tracking and battery level
 */
struct DeviceHistory
{
    std::deque<double> rsrpHistory; // Last 20 RSRP readings
    std::deque<double> sinrHistory; // Last 20 SINR values
    std::deque<bool> packetResults; // Last 20 packet outcomes
    std::deque<double> timestamps;  // Timestamps of packets
    uint32_t consecutiveLosses = 0;
    double lastSuccessTime = 0.0;

    // NB-IoT specific tracking
    uint32_t rachAttempts = 0;
    uint32_t rachFailures = 0;
    std::deque<bool> rachResults;        // Last 20 RACH outcomes
    std::deque<int> rachAttemptsHistory; // Number of attempts per RACH
    uint8_t currentRU = 1;               // Current Resource Unit index
    uint8_t previousRU = 1;
    int currentRepetitions = 1;
    double batteryLevel = 1.0;

    void addRsrp(double rsrp)
    {
        rsrpHistory.push_back(rsrp);
        if (rsrpHistory.size() > 20)
        {
            rsrpHistory.pop_front();
        }
    }

    void addSinr(double sinr)
    {
        sinrHistory.push_back(sinr);
        if (sinrHistory.size() > 20)
        {
            sinrHistory.pop_front();
        }
    }

    void addPacketResult(bool success, double timestamp)
    {
        packetResults.push_back(success);
        if (packetResults.size() > 20)
        {
            packetResults.pop_front();
        }
        timestamps.push_back(timestamp);
        if (timestamps.size() > 20)
        {
            timestamps.pop_front();
        }
        if (success)
        {
            consecutiveLosses = 0;
            lastSuccessTime = timestamp;
        }
        else
        {
            consecutiveLosses++;
        }
    }

    void addRachResult(bool success, int attempts)
    {
        rachResults.push_back(success);
        rachAttemptsHistory.push_back(attempts);
        if (rachResults.size() > 20)
        {
            rachResults.pop_front();
        }
        if (rachAttemptsHistory.size() > 20)
        {
            rachAttemptsHistory.pop_front();
        }
        rachAttempts += attempts;
        if (!success)
        {
            rachFailures++;
        }
    }

    void setRU(uint8_t ru)
    {
        previousRU = currentRU;
        currentRU = ru;
    }

    double getRsrpVariance() const
    {
        if (rsrpHistory.size() < 2)
        {
            return 0.0;
        }
        double mean = 0;
        for (auto v : rsrpHistory)
        {
            mean += v;
        }
        mean /= rsrpHistory.size();
        double variance = 0;
        for (auto v : rsrpHistory)
        {
            variance += (v - mean) * (v - mean);
        }
        return variance / rsrpHistory.size();
    }

    double getSinrTrend() const
    {
        if (sinrHistory.size() < 5)
        {
            return 0.0;
        }
        double recentAvg = 0, oldAvg = 0;
        size_t half = sinrHistory.size() / 2;
        for (size_t i = 0; i < half; i++)
        {
            oldAvg += sinrHistory[i];
        }
        for (size_t i = half; i < sinrHistory.size(); i++)
        {
            recentAvg += sinrHistory[i];
        }
        oldAvg /= half;
        recentAvg /= (sinrHistory.size() - half);
        return recentAvg - oldAvg;
    }

    double getRecentPdr() const
    {
        if (packetResults.empty())
        {
            return 0.5;
        }
        int success = std::count(packetResults.begin(), packetResults.end(), true);
        return static_cast<double>(success) / packetResults.size();
    }

    double getPdrTrend() const
    {
        if (packetResults.size() < 10)
        {
            return 0.0;
        }
        int half = packetResults.size() / 2;
        int oldSuccess = 0, recentSuccess = 0;
        for (int i = 0; i < half; i++)
        {
            if (packetResults[i])
            {
                oldSuccess++;
            }
        }
        for (size_t i = half; i < packetResults.size(); i++)
        {
            if (packetResults[i])
            {
                recentSuccess++;
            }
        }
        double oldPdr = static_cast<double>(oldSuccess) / half;
        double recentPdr = static_cast<double>(recentSuccess) / (packetResults.size() - half);
        return recentPdr - oldPdr;
    }

    double getLastRsrp() const
    {
        return rsrpHistory.empty() ? -120.0 : rsrpHistory.back();
    }

    double getLastSinr() const
    {
        return sinrHistory.empty() ? -10.0 : sinrHistory.back();
    }

    double getRachSuccessRate() const
    {
        if (rachResults.empty())
        {
            return 1.0;
        }
        int success = std::count(rachResults.begin(), rachResults.end(), true);
        return static_cast<double>(success) / rachResults.size();
    }

    double getAvgRachAttempts() const
    {
        if (rachAttemptsHistory.empty())
        {
            return 1.0;
        }
        double sum = 0;
        for (auto a : rachAttemptsHistory)
        {
            sum += a;
        }
        return sum / rachAttemptsHistory.size();
    }
};

// Global device histories
std::map<uint32_t, DeviceHistory> deviceHistories;

/**
 * RACH Collision Heuristic for NB-IoT
 * Infers collision probability from observables
 */
struct RACHCollisionIndicators
{
    double collisionProbability;
    bool likelyCollision;
    bool likelyWeakSignal;
    bool likelyCongestion;

    static RACHCollisionIndicators analyze(double rsrp,
                                           double sinr,
                                           bool packetSuccess,
                                           bool rachSuccess,
                                           int rachAttempts,
                                           double rsrpVariance,
                                           uint32_t consecutiveLosses,
                                           double recentPdr,
                                           double cellLoad)
    {
        RACHCollisionIndicators ind;
        ind.collisionProbability = 0.0;
        ind.likelyCollision = false;
        ind.likelyWeakSignal = false;
        ind.likelyCongestion = false;

        // Weak signal detection
        if (rsrp < -130.0 || sinr < -10.0)
        {
            ind.likelyWeakSignal = true;
        }

        // RACH collision detection
        if (!rachSuccess && !ind.likelyWeakSignal)
        {
            ind.likelyCollision = true;
            ind.collisionProbability = 0.7;
        }
        else if (rachAttempts > 3 && rsrp > -120.0)
        {
            ind.likelyCollision = true;
            ind.collisionProbability = 0.5;
        }

        // Cell congestion detection
        if (cellLoad > 0.7)
        {
            ind.likelyCongestion = true;
            ind.collisionProbability = std::max(ind.collisionProbability, cellLoad * 0.6);
        }

        // PDR-based inference
        if (recentPdr < 0.5 && rsrp > -115.0 && sinr > -5.0)
        {
            // Good signal but poor delivery → likely collision/congestion
            ind.likelyCongestion = true;
            ind.collisionProbability = std::max(ind.collisionProbability, 0.6);
        }

        return ind;
    }
};

/**
 * Extended Action Space for NB-IoT
 * 60 actions: 5 MCS deltas × 4 repetition changes × 3 RU changes
 */
class ExtendedActionSpace
{
  public:
    static const int NUM_ACTIONS = 60;

    struct ActionResult
    {
        int newMCS;
        int newRepetitions;
        int newRU; // ResourceUnit index
        double newTP;
    };

    static ActionResult decodeAction(int action,
                                     uint8_t currentMCS,
                                     int currentReps,
                                     ResourceUnit currentRU,
                                     double currentTP)
    {
        ActionResult result;

        // Decompose action index
        int mcsAction = action / 12;      // 0-4: MCS delta
        int repAction = (action / 3) % 4; // 0-3: repetition change
        int ruAction = action % 3;        // 0-2: RU change

        // MCS delta: -2, -1, 0, +1, +2
        int mcsDelta[] = {-2, -1, 0, 1, 2};
        result.newMCS = std::max(0, std::min(10, (int)currentMCS + mcsDelta[mcsAction]));

        // Repetition change: /4, /2, same, *2
        switch (repAction)
        {
        case 0:
            result.newRepetitions = std::max(1, currentReps / 4);
            break;
        case 1:
            result.newRepetitions = std::max(1, currentReps / 2);
            break;
        case 2:
            result.newRepetitions = currentReps;
            break;
        case 3:
            result.newRepetitions = std::min(128, currentReps * 2);
            break;
        default:
            result.newRepetitions = currentReps;
            break;
        }

        // RU change: narrower, same, wider
        int ruIdx = static_cast<int>(currentRU);
        switch (ruAction)
        {
        case 0:
            result.newRU = std::max(0, ruIdx - 1);
            break; // Narrower (more robust)
        case 1:
            result.newRU = ruIdx;
            break; // Same
        case 2:
            result.newRU = std::min(4, ruIdx + 1);
            break; // Wider (more efficient)
        default:
            result.newRU = ruIdx;
            break;
        }

        // TX power stays the same in extended space (managed separately by power control)
        result.newTP = currentTP;

        return result;
    }
};

/**
 * Dueling DQN Network for NB-IoT
 * Separates state value and advantage estimation
 */
class DuelingQNetwork
{
  private:
    static const int INPUT_SIZE = 14; // NB-IoT realistic state size
    static const int HIDDEN_SIZE = 64;
    static const int VALUE_SIZE = 32;
    static const int ADVANTAGE_SIZE = 32;
    static const int OUTPUT_SIZE = 60; // ExtendedActionSpace::NUM_ACTIONS

    // Shared feature layers
    std::vector<std::vector<double>> sharedWeights1, sharedWeights2;
    std::vector<double> sharedBias1, sharedBias2;

    // Value stream
    std::vector<std::vector<double>> valueWeights1, valueWeights2;
    std::vector<double> valueBias1, valueBias2;

    // Advantage stream
    std::vector<std::vector<double>> advWeights1, advWeights2;
    std::vector<double> advBias1, advBias2;

    double learningRate;

    double leakyRelu(double x, double alpha = 0.01)
    {
        return x > 0 ? x : alpha * x;
    }

  public:
    DuelingQNetwork(double lr = 0.0005)
        : learningRate(lr)
    {
        initializeWeights();
    }

    void initializeWeights()
    {
        std::mt19937 rng(42);
        auto xavier = [&](int fanIn, int fanOut) -> double {
            std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (fanIn + fanOut)));
            return dist(rng);
        };

        // Shared layers
        sharedWeights1.resize(HIDDEN_SIZE, std::vector<double>(INPUT_SIZE));
        sharedBias1.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sharedWeights1[i][j] = xavier(INPUT_SIZE, HIDDEN_SIZE);
            }
        }

        sharedWeights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        sharedBias2.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sharedWeights2[i][j] = xavier(HIDDEN_SIZE, HIDDEN_SIZE);
            }
        }

        // Value stream
        valueWeights1.resize(VALUE_SIZE, std::vector<double>(HIDDEN_SIZE));
        valueBias1.resize(VALUE_SIZE, 0.0);
        valueWeights2.resize(1, std::vector<double>(VALUE_SIZE));
        valueBias2.resize(1, 0.0);
        for (int i = 0; i < VALUE_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                valueWeights1[i][j] = xavier(HIDDEN_SIZE, VALUE_SIZE);
            }
        }
        for (int j = 0; j < VALUE_SIZE; j++)
        {
            valueWeights2[0][j] = xavier(VALUE_SIZE, 1);
        }

        // Advantage stream
        advWeights1.resize(ADVANTAGE_SIZE, std::vector<double>(HIDDEN_SIZE));
        advBias1.resize(ADVANTAGE_SIZE, 0.0);
        advWeights2.resize(OUTPUT_SIZE, std::vector<double>(ADVANTAGE_SIZE));
        advBias2.resize(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < ADVANTAGE_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                advWeights1[i][j] = xavier(HIDDEN_SIZE, ADVANTAGE_SIZE);
            }
        }
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < ADVANTAGE_SIZE; j++)
            {
                advWeights2[i][j] = xavier(ADVANTAGE_SIZE, OUTPUT_SIZE);
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input)
    {
        // Shared layers
        std::vector<double> h1(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = sharedBias1[i];
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += sharedWeights1[i][j] * input[j];
            }
            h1[i] = leakyRelu(sum);
        }

        std::vector<double> h2(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = sharedBias2[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += sharedWeights2[i][j] * h1[j];
            }
            h2[i] = leakyRelu(sum);
        }

        // Value stream
        std::vector<double> v1(VALUE_SIZE);
        for (int i = 0; i < VALUE_SIZE; i++)
        {
            double sum = valueBias1[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += valueWeights1[i][j] * h2[j];
            }
            v1[i] = leakyRelu(sum);
        }
        double value = valueBias2[0];
        for (int j = 0; j < VALUE_SIZE; j++)
        {
            value += valueWeights2[0][j] * v1[j];
        }

        // Advantage stream
        std::vector<double> a1(ADVANTAGE_SIZE);
        for (int i = 0; i < ADVANTAGE_SIZE; i++)
        {
            double sum = advBias1[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += advWeights1[i][j] * h2[j];
            }
            a1[i] = leakyRelu(sum);
        }
        std::vector<double> advantages(OUTPUT_SIZE);
        double advMean = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double sum = advBias2[i];
            for (int j = 0; j < ADVANTAGE_SIZE; j++)
            {
                sum += advWeights2[i][j] * a1[j];
            }
            advantages[i] = sum;
            advMean += sum;
        }
        advMean /= OUTPUT_SIZE;

        // Q = V + (A - mean(A))
        std::vector<double> output(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            output[i] = value + advantages[i] - advMean;
        }

        return output;
    }

    void copyFrom(const DuelingQNetwork& other)
    {
        sharedWeights1 = other.sharedWeights1;
        sharedBias1 = other.sharedBias1;
        sharedWeights2 = other.sharedWeights2;
        sharedBias2 = other.sharedBias2;
        valueWeights1 = other.valueWeights1;
        valueBias1 = other.valueBias1;
        valueWeights2 = other.valueWeights2;
        valueBias2 = other.valueBias2;
        advWeights1 = other.advWeights1;
        advBias1 = other.advBias1;
        advWeights2 = other.advWeights2;
        advBias2 = other.advBias2;
    }
};

/**
 * Optimized DDQN Agent for NB-IoT
 * Uses Dueling DQN architecture with realistic state representation
 */
class OptimizedDDQNAgent
{
  private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;

    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;

    // Curiosity-driven exploration
    std::map<std::string, int> stateVisitCounts;
    double curiosityBonus = 0.1;

    std::mt19937 rng;

    std::string discretizeState(const std::vector<double>& state)
    {
        std::string key;
        for (auto v : state)
        {
            key += std::to_string(static_cast<int>(v * 10)) + ",";
        }
        return key;
    }

  public:
    OptimizedDDQNAgent(double lr = 0.0005,
                       double eps = 0.3,
                       double epsDecay = 0.999,
                       double epsMin = 0.05,
                       double g = 0.95,
                       int targetFreq = 50)
        : mainNetwork(lr),
          targetNetwork(lr),
          replayBuffer(20000),
          epsilon(eps),
          epsilonDecay(epsDecay),
          epsilonMin(epsMin),
          gamma(g),
          targetUpdateFreq(targetFreq),
          updateCounter(0),
          rng(std::random_device{}())
    {
        targetNetwork.copyFrom(mainNetwork);
    }

    /**
     * Build realistic state vector for NB-IoT
     * 14 features observable by UE and eNB feedback
     */
    std::vector<double> buildRealisticState(const DeviceHistory& history,
                                            uint8_t currentMCS,
                                            int currentReps,
                                            double currentTP,
                                            uint8_t currentRU,
                                            double cellLoad,
                                            double batteryLevel,
                                            double currentTime)
    {
        std::vector<double> state(14);

        // Feature 0: Normalized RSRP (-140 to -44 dBm → 0-1)
        state[0] = std::max(0.0, std::min(1.0, (history.getLastRsrp() + 140.0) / 96.0));

        // Feature 1: Normalized SINR (-12 to 30 dB → 0-1)
        state[1] = std::max(0.0, std::min(1.0, (history.getLastSinr() + 12.0) / 42.0));

        // Feature 2: RSRP variance (channel stability)
        state[2] = std::min(1.0, history.getRsrpVariance() / 20.0);

        // Feature 3: SINR trend
        state[3] = std::max(-1.0, std::min(1.0, history.getSinrTrend() / 10.0));

        // Feature 4: Current MCS normalized
        state[4] = currentMCS / 10.0;

        // Feature 5: Current repetitions (log2 normalized)
        state[5] = std::log2(currentReps) / 7.0;

        // Feature 6: Current TX power normalized
        state[6] = currentTP / 23.0;

        // Feature 7: Current RU normalized
        state[7] = currentRU / 4.0;

        // Feature 8: Recent PDR
        state[8] = history.getRecentPdr();

        // Feature 9: PDR trend
        state[9] = std::max(-1.0, std::min(1.0, history.getPdrTrend()));

        // Feature 10: Cell load
        state[10] = cellLoad;

        // Feature 11: Battery level
        state[11] = batteryLevel;

        // Feature 12: RACH success rate
        state[12] = history.getRachSuccessRate();

        // Feature 13: Consecutive failures normalized
        state[13] = std::min(1.0, history.consecutiveLosses / 10.0);

        return state;
    }

    int selectAction(const std::vector<double>& state,
                     uint32_t ueId,
                     const DeviceHistory& history,
                     double currentTime,
                     double cellLoad = 0.5)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(rng) < epsilon)
        {
            // Smart exploration using collision/coverage heuristics
            auto indicators = RACHCollisionIndicators::analyze(
                history.getLastRsrp(),
                history.getLastSinr(),
                !history.packetResults.empty() && history.packetResults.back(),
                !history.rachResults.empty() && history.rachResults.back(),
                history.rachAttemptsHistory.empty() ? 1 : history.rachAttemptsHistory.back(),
                history.getRsrpVariance(),
                history.consecutiveLosses,
                history.getRecentPdr(),
                cellLoad);

            if (indicators.likelyWeakSignal)
            {
                // Increase robustness: lower MCS (-2), more reps (*2), narrower RU
                int action = 0 * 12 + 3 * 3 + 0; // MCS delta=-2, reps*2, RU narrower
                return std::min(action, ExtendedActionSpace::NUM_ACTIONS - 1);
            }

            if (indicators.likelyCongestion && cellLoad > 0.6)
            {
                // Try to reduce resource usage to decrease congestion
                int action = 3 * 12 + 1 * 3 + 2; // MCS delta=+1, reps/2, RU wider
                return std::min(action, ExtendedActionSpace::NUM_ACTIONS - 1);
            }

            // Random exploration
            std::uniform_int_distribution<int> actionDist(0, ExtendedActionSpace::NUM_ACTIONS - 1);
            return actionDist(rng);
        }

        // Greedy action selection
        auto qValues = mainNetwork.forward(state);

        // Apply curiosity bonus
        std::string stateKey = discretizeState(state);
        int visitCount = stateVisitCounts[stateKey]++;
        for (int i = 0; i < ExtendedActionSpace::NUM_ACTIONS; i++)
        {
            qValues[i] += curiosityBonus / (1.0 + visitCount);
        }

        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }

    void addExperience(const std::vector<double>& state,
                       int action,
                       double reward,
                       const std::vector<double>& nextState,
                       bool done)
    {
        replayBuffer.add(Experience(state, action, reward, nextState, done));
    }

    void train(size_t batchSize = 32)
    {
        if (replayBuffer.size() < batchSize)
        {
            return;
        }

        std::vector<size_t> indices;
        std::vector<double> isWeights;
        auto batch = replayBuffer.sample(batchSize, indices, isWeights);

        std::vector<double> tdErrors;
        for (size_t i = 0; i < batch.size(); i++)
        {
            auto& exp = batch[i];
            auto mainQNext = mainNetwork.forward(exp.nextState);
            int bestAction = std::distance(mainQNext.begin(),
                                           std::max_element(mainQNext.begin(), mainQNext.end()));
            auto targetQNext = targetNetwork.forward(exp.nextState);
            double targetValue = exp.done ? 0.0 : gamma * targetQNext[bestAction];
            double target = exp.reward + targetValue;

            auto currentQ = mainNetwork.forward(exp.state);
            double tdError = target - currentQ[exp.action];
            tdErrors.push_back(tdError);

            auto targetQ = currentQ;
            targetQ[exp.action] = target;
            // Scale by IS weight
            for (int j = 0; j < ExtendedActionSpace::NUM_ACTIONS; j++)
            {
                targetQ[j] = currentQ[j] + isWeights[i] * (targetQ[j] - currentQ[j]);
            }
        }

        replayBuffer.updatePriorities(indices, tdErrors);

        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0)
        {
            targetNetwork.copyFrom(mainNetwork);
        }

        epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    }

    double getEpsilon() const
    {
        return epsilon;
    }

    void increaseExploration()
    {
        epsilon = std::min(1.0, epsilon * 2.0);
    }
};

/**
 * Estimate cell congestion from observable NB-IoT metrics
 * NB-IoT congestion is different from LoRaWAN: scheduled access means
 * congestion shows up as RACH collisions and scheduling delays
 */
double
estimateCellCongestion(double rsrp,
                       double rsrpVariance,
                       double recentPdr,
                       uint32_t numActiveUEs,
                       double avgRachAttempts,
                       double cellLoad)
{
    double congestion = 0.0;

    // Factor 1: Cell load (most direct indicator in licensed spectrum)
    congestion += 0.4 * cellLoad;

    // Factor 2: Number of active UEs
    double ueLoadRatio = std::min(1.0, numActiveUEs / 50.0); // 50 UEs = full load
    congestion += 0.2 * ueLoadRatio;

    // Factor 3: RACH attempt rate (high attempts = congested RACH)
    double rachCongestion = std::min(1.0, (avgRachAttempts - 1.0) / 5.0);
    congestion += 0.2 * std::max(0.0, rachCongestion);

    // Factor 4: Low PDR with good signal = scheduling congestion
    if (recentPdr < 0.5 && rsrp > -115.0)
    {
        congestion += 0.2;
    }

    return std::min(1.0, congestion);
}

/**
 * MCS ceiling based on SINR - returns maximum MCS allowed
 */
uint8_t
getMandatoryMCSCeiling(double sinr, int repetitions)
{
    double effectiveSINR = sinr + GetRepetitionGainDb(repetitions);
    for (int mcs = 10; mcs >= 0; mcs--)
    {
        if (effectiveSINR >= MCS_TABLE[mcs].requiredSINR_dB + 2.0)
        { // 2 dB margin
            return mcs;
        }
    }
    return 0;
}

/**
 * Congestion-Adaptive Reward Function for NB-IoT
 */
double
calculateCongestionAdaptiveReward(uint32_t ueId,
                                  bool packetSuccess,
                                  bool rachSuccess,
                                  int rachAttempts,
                                  double rsrp,
                                  double sinr,
                                  double rsrpVariance,
                                  uint32_t consecutiveLosses,
                                  uint8_t currentMCS,
                                  double currentTP,
                                  int currentReps,
                                  uint8_t currentRU,
                                  double recentPdr,
                                  double pdrTrend,
                                  double timeSinceLastSuccess,
                                  double congestionLevel,
                                  double energyConsumed_mJ)
{
    double reward = 0.0;

    // ========================================
    // COMPONENT 1: Packet Outcome (PRIMARY)
    // ========================================
    double outcomeWeight = 0.35 + 0.25 * (1.0 - congestionLevel);

    if (packetSuccess)
    {
        reward += outcomeWeight * 1.0;
        if (recentPdr >= 0.9)
        {
            reward += outcomeWeight * 0.3;
        }
    }
    else
    {
        reward -= outcomeWeight * 1.0;
        if (consecutiveLosses > 5)
        {
            reward -= outcomeWeight * 0.5;
        }
    }

    // ========================================
    // COMPONENT 2: RACH Efficiency
    // More important when cell is congested
    // ========================================
    double rachWeight = 0.15 + 0.15 * congestionLevel;

    if (!rachSuccess)
    {
        reward -= rachWeight * 1.0;
    }
    else if (rachAttempts == 1)
    {
        reward += rachWeight * 0.3;
    }
    else if (rachAttempts > 3)
    {
        reward -= rachWeight * 0.2 * (rachAttempts - 3);
    }

    // ========================================
    // COMPONENT 3: Energy Efficiency
    // More important when congestion is low (can optimize)
    // ========================================
    double energyWeight = 0.15 + 0.2 * (1.0 - congestionLevel);

    if (packetSuccess)
    {
        // Reward lower energy
        double energyNorm = std::min(1.0, energyConsumed_mJ / 200.0);
        reward += energyWeight * (1.0 - energyNorm);

        // Reward using fewer repetitions
        double repsNorm = std::log2(currentReps) / 7.0;
        reward += energyWeight * 0.5 * (1.0 - repsNorm);
    }

    // ========================================
    // COMPONENT 4: MCS Efficiency
    // Reward using higher MCS when SINR allows
    // ========================================
    double mcsWeight = 0.1 + 0.1 * (1.0 - congestionLevel);

    uint8_t optimalMCS = getMandatoryMCSCeiling(sinr, currentReps);
    if (packetSuccess && currentMCS == optimalMCS)
    {
        reward += mcsWeight * 0.5;
    }
    else if (currentMCS < optimalMCS - 2)
    {
        reward -= mcsWeight * 0.3; // Under-utilizing SINR margin
    }

    // ========================================
    // COMPONENT 5: PDR Stability
    // ========================================
    double stabilityWeight = 0.15;

    if (pdrTrend > 0.1)
    {
        reward += stabilityWeight * 0.5; // Improving
    }
    else if (pdrTrend < -0.1)
    {
        reward -= stabilityWeight * 0.5; // Degrading
    }

    // ========================================
    // COMPONENT 6: Coverage Class Appropriateness
    // ========================================
    double effectiveSINR = sinr + GetRepetitionGainDb(currentReps);
    if (effectiveSINR > 15.0 && currentReps > 4)
    {
        reward -= 0.1; // Over-provisioned repetitions
    }
    if (effectiveSINR < 0.0 && currentReps < 8)
    {
        reward -= 0.1; // Under-provisioned for poor coverage
    }

    return reward;
}

// Global agent storage
std::map<uint32_t, std::unique_ptr<DDQNPERADRAgent>> ddqnAgents;
std::map<uint32_t, std::unique_ptr<OptimizedDDQNAgent>> optimizedAgents;

// ============================================================================
// PPO AGENT FOR NB-IoT
// ============================================================================

/**
 * Policy Network for PPO (Actor)
 * Outputs mean and log_std for continuous MCS/Repetition actions
 */
class PolicyNetwork
{
  private:
    static const int INPUT_SIZE = 14;
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 3; // mean_mcs, mean_log2_reps, mean_tp

    std::vector<std::vector<double>> weights1, weights2, weightsMean, weightsLogStd;
    std::vector<double> bias1, bias2, biasMean, biasLogStd;
    double learningRate;
    std::mt19937 rng;

    double tanh_fn(double x)
    {
        return std::tanh(x);
    }

    double relu(double x)
    {
        return x > 0 ? x : 0;
    }

  public:
    PolicyNetwork(double lr = 0.0003)
        : learningRate(lr),
          rng(std::random_device{}())
    {
        initializeWeights();
    }

    void initializeWeights()
    {
        std::mt19937 initRng(42);
        auto xavier = [&](int fi, int fo) -> double {
            std::normal_distribution<double> d(0.0, std::sqrt(2.0 / (fi + fo)));
            return d(initRng);
        };

        weights1.resize(HIDDEN_SIZE, std::vector<double>(INPUT_SIZE));
        bias1.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                weights1[i][j] = xavier(INPUT_SIZE, HIDDEN_SIZE);
            }
        }

        weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        bias2.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                weights2[i][j] = xavier(HIDDEN_SIZE, HIDDEN_SIZE);
            }
        }

        weightsMean.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        biasMean.resize(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                weightsMean[i][j] = xavier(HIDDEN_SIZE, OUTPUT_SIZE);
            }
        }
        // Pragmatic initialization toward a reasonable operating profile:
        // MCS≈5, log2(repetitions)≈2 (reps=4), TX power≈18 dBm.
        biasMean[0] = 5.0;
        biasMean[1] = 2.0;
        biasMean[2] = 18.0;

        weightsLogStd.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_SIZE));
        biasLogStd.resize(OUTPUT_SIZE, -1.0); // Start with lower std
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                weightsLogStd[i][j] = xavier(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1;
            }
        }
    }

    /**
     * Forward pass: outputs (mean, logStd) for each action dimension
     * Action 0: MCS (0-10)
     * Action 1: log2(repetitions) (0-7)
     * Action 2: TX power (0-23 dBm)
     */
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input)
    {
        // Layer 1
        std::vector<double> h1(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += weights1[i][j] * input[j];
            }
            h1[i] = tanh_fn(sum);
        }

        // Layer 2
        std::vector<double> h2(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += weights2[i][j] * h1[j];
            }
            h2[i] = tanh_fn(sum);
        }

        // Mean output
        std::vector<double> mean(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double sum = biasMean[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += weightsMean[i][j] * h2[j];
            }
            mean[i] = sum;
        }

        // Log std output
        std::vector<double> logStd(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double sum = biasLogStd[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += weightsLogStd[i][j] * h2[j];
            }
            logStd[i] = std::max(-2.0, std::min(0.5, sum)); // Clamp
        }

        return {mean, logStd};
    }

    /**
     * Sample action from policy distribution
     * Returns: {MCS, log2(repetitions), TX power}
     */
    std::vector<double> sampleAction(const std::vector<double>& input)
    {
        auto [mean, logStd] = forward(input);
        std::vector<double> action(OUTPUT_SIZE);

        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double std = std::exp(logStd[i]);
            std::normal_distribution<double> dist(mean[i], std);
            action[i] = dist(rng);
        }

        // Clamp to valid NB-IoT ranges
        action[0] = std::max(0.0, std::min(10.0, action[0]));  // MCS 0-10
        action[1] = std::max(0.0, std::min(7.0, action[1]));   // log2(reps) 0-7
        action[2] = std::max(14.0, std::min(23.0, action[2])); // TP 14-23 dBm (Class 6 min)

        return action;
    }

    double computeLogProb(const std::vector<double>& input, const std::vector<double>& action)
    {
        auto [mean, logStd] = forward(input);
        constexpr double kMinVar = 1e-8;
        const double twoPi = 2.0 * std::acos(-1.0);

        double logProb = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double std = std::exp(logStd[i]);
            double var = std * std + kMinVar;
            double diff = action[i] - mean[i];
            logProb += -0.5 * ((diff * diff) / var + std::log(twoPi * var));
        }
        return logProb;
    }

    void updatePolicy(const std::vector<double>& input,
                      const std::vector<double>& action,
                      double oldLogProb,
                      double advantage,
                      double clip,
                      double entropyCoef)
    {
        auto [mean, logStd] = forward(input);
        constexpr double kMinVar = 1e-8;

        double currentLogProb = computeLogProb(input, action);
        double logRatio = std::max(-20.0, std::min(20.0, currentLogProb - oldLogProb));
        double ratio = std::exp(logRatio);
        double clippedRatio = std::max(1.0 - clip, std::min(1.0 + clip, ratio));

        // Gradient scale for PPO clipped surrogate (ascent).
        double unclippedObj = ratio * advantage;
        double clippedObj = clippedRatio * advantage;
        double gradientScale = (unclippedObj <= clippedObj) ? ratio * advantage : 0.0;

        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            double std = std::exp(logStd[i]);
            double var = std * std + kMinVar;
            double diff = action[i] - mean[i];

            double gradMean = diff / var;
            double gradLogStd = (diff * diff) / var - 1.0;

            // Lightweight policy update on output heads (stable approximation).
            biasMean[i] += learningRate * 0.05 * gradientScale * gradMean;
            biasLogStd[i] += learningRate * 0.02 * (gradientScale * gradLogStd + entropyCoef);
            biasLogStd[i] = std::max(-2.0, std::min(0.5, biasLogStd[i]));
        }
    }
};

/**
 * Value Network for PPO (Critic)
 */
class ValueNetwork
{
  private:
    static const int INPUT_SIZE = 14;
    static const int HIDDEN_SIZE = 64;

    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    double learningRate;

    double tanh_fn(double x)
    {
        return std::tanh(x);
    }

  public:
    ValueNetwork(double lr = 0.001)
        : learningRate(lr)
    {
        initializeWeights();
    }

    void initializeWeights()
    {
        std::mt19937 rng(42);
        auto xavier = [&](int fi, int fo) -> double {
            std::normal_distribution<double> d(0.0, std::sqrt(2.0 / (fi + fo)));
            return d(rng);
        };

        weights1.resize(HIDDEN_SIZE, std::vector<double>(INPUT_SIZE));
        bias1.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                weights1[i][j] = xavier(INPUT_SIZE, HIDDEN_SIZE);
            }
        }

        weights2.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE));
        bias2.resize(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                weights2[i][j] = xavier(HIDDEN_SIZE, HIDDEN_SIZE);
            }
        }

        weights3.resize(1, std::vector<double>(HIDDEN_SIZE));
        bias3.resize(1, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            weights3[0][j] = xavier(HIDDEN_SIZE, 1);
        }
    }

    double forward(const std::vector<double>& input)
    {
        std::vector<double> h1(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += weights1[i][j] * input[j];
            }
            h1[i] = tanh_fn(sum);
        }

        std::vector<double> h2(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                sum += weights2[i][j] * h1[j];
            }
            h2[i] = tanh_fn(sum);
        }

        double value = bias3[0];
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            value += weights3[0][j] * h2[j];
        }

        return value;
    }

    void update(const std::vector<double>& input, double target)
    {
        double pred = forward(input);
        double error = target - pred;
        // Simplified gradient update (same approach as LoRaWAN version)
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            weights3[0][j] += learningRate * error * 0.01;
        }
        bias3[0] += learningRate * error * 0.01;
    }
};

/**
 * PPO Agent for NB-IoT Link Adaptation
 * Continuous action space: MCS, repetitions, TX power
 */
class PPOAgent
{
  private:
    PolicyNetwork actor;
    ValueNetwork critic;

    double clipEpsilon = 0.2;
    double gamma = 0.95;
    double gaeLambda = 0.95;
    double entropyCoeff = 0.01;

    std::mt19937 rng;

    struct Transition
    {
        std::vector<double> state;
        std::vector<double> action;
        double reward;
        std::vector<double> nextState;
        bool done;
        double value;
        double logProb;
    };

    std::vector<Transition> trajectoryBuffer;
    int updateEvery = 16;
    int stepCount = 0;

  public:
    PPOAgent()
        : rng(std::random_device{}())
    {
    }

    /**
     * Select action: returns {MCS, repetitions, TX power}
     */
    struct NbiotAction
    {
        uint8_t mcs;
        int repetitions;
        double txPower;
    };

    NbiotAction selectAction(const std::vector<double>& state)
    {
        auto rawAction = actor.sampleAction(state);

        NbiotAction action;
        action.mcs = static_cast<uint8_t>(std::round(rawAction[0]));
        action.mcs = std::max((uint8_t)0, std::min((uint8_t)10, action.mcs));

        int repIdx = static_cast<int>(std::round(rawAction[1]));
        repIdx = std::max(0, std::min(7, repIdx));
        action.repetitions = REPETITION_VALUES[repIdx];

        // TX power: map from raw [0,23] to discrete NB-IoT power classes
        // Minimum 14 dBm (Class 6), typical 20-23 dBm
        action.txPower = std::round(rawAction[2]);
        action.txPower = std::max(14.0, std::min(23.0, action.txPower));

        return action;
    }

    double evaluateLogProb(const std::vector<double>& state, const std::vector<double>& action)
    {
        return actor.computeLogProb(state, action);
    }

    void addTransition(const std::vector<double>& state,
                       const std::vector<double>& action,
                       double reward,
                       const std::vector<double>& nextState,
                       bool done)
    {
        Transition t;
        t.state = state;
        t.action = action;
        t.reward = reward;
        t.nextState = nextState;
        t.done = done;
        t.value = critic.forward(state);
        t.logProb = action.size() > 3 ? action[3] : actor.computeLogProb(state, t.action);

        if (t.action.size() > 3)
        {
            t.action.resize(3);
        }

        trajectoryBuffer.push_back(t);
        stepCount++;

        if (stepCount % updateEvery == 0 && trajectoryBuffer.size() >= 16)
        {
            train();
            trajectoryBuffer.clear();
        }
    }

    void train()
    {
        if (trajectoryBuffer.empty())
        {
            return;
        }

        // Calculate returns and advantages (GAE)
        std::vector<double> returns(trajectoryBuffer.size());
        std::vector<double> advantages(trajectoryBuffer.size());

        double lastValue = 0.0;
        if (!trajectoryBuffer.back().done)
        {
            lastValue = critic.forward(trajectoryBuffer.back().nextState);
        }

        double gae = 0.0;
        for (int t = trajectoryBuffer.size() - 1; t >= 0; t--)
        {
            double nextValue =
                (t == (int)trajectoryBuffer.size() - 1) ? lastValue : trajectoryBuffer[t + 1].value;
            double delta =
                trajectoryBuffer[t].reward + gamma * nextValue - trajectoryBuffer[t].value;
            gae = delta + gamma * gaeLambda * gae;
            advantages[t] = gae;
            returns[t] = advantages[t] + trajectoryBuffer[t].value;
        }

        // Normalize advantages for stable actor updates.
        double advMean = std::accumulate(advantages.begin(), advantages.end(), 0.0) /
                         std::max<size_t>(1, advantages.size());
        double advVar = 0.0;
        for (double a : advantages)
        {
            double d = a - advMean;
            advVar += d * d;
        }
        advVar /= std::max<size_t>(1, advantages.size());
        double advStd = std::sqrt(advVar + 1e-8);

        // Update critic
        for (size_t i = 0; i < trajectoryBuffer.size(); i++)
        {
            critic.update(trajectoryBuffer[i].state, returns[i]);
            double normAdv = (advantages[i] - advMean) / advStd;
            actor.updatePolicy(trajectoryBuffer[i].state,
                               trajectoryBuffer[i].action,
                               trajectoryBuffer[i].logProb,
                               normAdv,
                               clipEpsilon,
                               entropyCoeff);
        }
    }
};

// Global PPO agents
std::map<uint32_t, std::unique_ptr<PPOAgent>> ppoAgents;

// ============================================================================
// MARL AGENT FOR NB-IoT (Independent Q-Learning + Coordination)
// ============================================================================

/**
 * MARL Agent for NB-IoT
 * Each UE learns independently but receives coordination signals from eNB
 * In NB-IoT, the eNB provides much more state info than a LoRaWAN gateway
 */
class MARLAgent
{
  private:
    DuelingQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;

    double epsilon, epsilonDecay, epsilonMin;
    double gamma;
    int targetUpdateFreq, updateCounter;

    std::mt19937 rng;

    // Coordination: tracks other agents' parameters (provided by eNB)
    std::map<uint32_t, int> otherAgentActions;
    std::map<uint32_t, uint8_t> otherAgentMCS;
    std::map<uint32_t, int> otherAgentReps;

  public:
    MARLAgent(double lr = 0.0005,
              double eps = 0.3,
              double epsDecay = 0.999,
              double epsMin = 0.05,
              double g = 0.95,
              int targetFreq = 50)
        : mainNetwork(lr),
          targetNetwork(lr),
          replayBuffer(20000),
          epsilon(eps),
          epsilonDecay(epsDecay),
          epsilonMin(epsMin),
          gamma(g),
          targetUpdateFreq(targetFreq),
          updateCounter(0),
          rng(std::random_device{}())
    {
        targetNetwork.copyFrom(mainNetwork);
    }

    void updateCoordination(uint32_t otherId, int action, uint8_t mcs, int reps)
    {
        otherAgentActions[otherId] = action;
        otherAgentMCS[otherId] = mcs;
        otherAgentReps[otherId] = reps;
    }

    /**
     * Build MARL state: includes coordination features from eNB
     * The eNB can provide cell-wide metrics that individual UEs cannot observe
     */
    std::vector<double> buildMARLState(const DeviceHistory& history,
                                       uint8_t currentMCS,
                                       int currentReps,
                                       double currentTP,
                                       uint8_t currentRU,
                                       double cellLoad,
                                       double batteryLevel,
                                       double currentTime,
                                       uint32_t numActiveUEs,
                                       double avgCellMCS,
                                       double avgCellReps)
    {
        std::vector<double> state(14);

        // Features 0-9: same as individual agent
        state[0] = std::max(0.0, std::min(1.0, (history.getLastRsrp() + 140.0) / 96.0));
        state[1] = std::max(0.0, std::min(1.0, (history.getLastSinr() + 12.0) / 42.0));
        state[2] = std::min(1.0, history.getRsrpVariance() / 20.0);
        state[3] = std::max(-1.0, std::min(1.0, history.getSinrTrend() / 10.0));
        state[4] = currentMCS / 10.0;
        state[5] = std::log2(currentReps) / 7.0;
        state[6] = currentTP / 23.0;
        state[7] = history.getRecentPdr();
        state[8] = history.getRachSuccessRate();
        state[9] = std::min(1.0, history.consecutiveLosses / 10.0);

        // Features 10-13: MARL coordination features (from eNB)
        state[10] = cellLoad;
        state[11] = std::min(1.0, numActiveUEs / 50.0);
        state[12] = avgCellMCS / 10.0; // Average MCS across all UEs
        state[13] = std::log2(std::max(1.0, avgCellReps)) / 7.0;

        return state;
    }

    int selectAction(const std::vector<double>& state,
                     uint32_t ueId,
                     const DeviceHistory& history,
                     double currentTime,
                     double cellLoad = 0.5)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(rng) < epsilon)
        {
            // MARL smart exploration: try to differentiate from other agents
            auto indicators = RACHCollisionIndicators::analyze(
                history.getLastRsrp(),
                history.getLastSinr(),
                !history.packetResults.empty() && history.packetResults.back(),
                !history.rachResults.empty() && history.rachResults.back(),
                history.rachAttemptsHistory.empty() ? 1 : history.rachAttemptsHistory.back(),
                history.getRsrpVariance(),
                history.consecutiveLosses,
                history.getRecentPdr(),
                cellLoad);

            if (indicators.likelyCongestion)
            {
                // In congestion, choose action that differs from majority of other agents
                std::map<int, int> actionCounts;
                for (auto& [id, act] : otherAgentActions)
                {
                    actionCounts[act]++;
                }

                // Select least-used action in extended space
                int leastUsedAction = 0;
                int minCount = INT_MAX;
                for (int a = 0; a < ExtendedActionSpace::NUM_ACTIONS; a++)
                {
                    if (actionCounts[a] < minCount)
                    {
                        minCount = actionCounts[a];
                        leastUsedAction = a;
                    }
                }
                return leastUsedAction;
            }

            std::uniform_int_distribution<int> actionDist(0, ExtendedActionSpace::NUM_ACTIONS - 1);
            return actionDist(rng);
        }

        auto qValues = mainNetwork.forward(state);
        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }

    void addExperience(const std::vector<double>& state,
                       int action,
                       double reward,
                       const std::vector<double>& nextState,
                       bool done)
    {
        replayBuffer.add(Experience(state, action, reward, nextState, done));
    }

    void train(size_t batchSize = 32)
    {
        if (replayBuffer.size() < batchSize)
        {
            return;
        }

        std::vector<size_t> indices;
        std::vector<double> isWeights;
        auto batch = replayBuffer.sample(batchSize, indices, isWeights);

        std::vector<double> tdErrors;
        for (size_t i = 0; i < batch.size(); i++)
        {
            auto& exp = batch[i];
            auto mainQNext = mainNetwork.forward(exp.nextState);
            int bestAction = std::distance(mainQNext.begin(),
                                           std::max_element(mainQNext.begin(), mainQNext.end()));
            auto targetQNext = targetNetwork.forward(exp.nextState);
            double targetValue = exp.done ? 0.0 : gamma * targetQNext[bestAction];
            double target = exp.reward + targetValue;
            auto currentQ = mainNetwork.forward(exp.state);
            double tdError = target - currentQ[exp.action];
            tdErrors.push_back(tdError);
        }

        replayBuffer.updatePriorities(indices, tdErrors);

        updateCounter++;
        if (updateCounter % targetUpdateFreq == 0)
        {
            targetNetwork.copyFrom(mainNetwork);
        }

        epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
    }

    double getEpsilon() const
    {
        return epsilon;
    }

    void increaseExploration()
    {
        epsilon = std::min(1.0, epsilon * 2.0);
    }
};

// Global MARL agents
std::map<uint32_t, std::unique_ptr<MARLAgent>> marlAgents;

// ============================================================================
// NB-IoT ENVIRONMENT & SIMULATION LOGIC
// ============================================================================

// ADR method enum
enum class ADRMethod
{
    OFF,
    ON,
    DDQN,
    PPO,
    MARL
};
ADRMethod currentADRMethod = ADRMethod::OFF;

// Global simulation state
Ptr<NbiotChannel> globalChannel;
Ptr<NbiotEnbDevice> globalEnb;
Ptr<NbiotEnergyModel> globalEnergyModel;
std::vector<Ptr<NbiotUeDevice>> ueDevices;

// Node positions
std::map<uint32_t, Vector> nodePositions;
std::map<uint32_t, double> nodeTxPowers;

// Current NB-IoT parameters per UE
std::map<uint32_t, uint8_t> currentMCS;
std::map<uint32_t, int> currentRepetitions;
std::map<uint32_t, double> currentTP;
std::map<uint32_t, uint8_t> currentRU;

// UE performance tracking
struct UEMetrics
{
    uint32_t packetsSent = 0;
    uint32_t packetsReceived = 0;
    double totalEnergyConsumed_mJ = 0.0;
    double lastPDR = 0.0;
    uint32_t totalRachAttempts = 0;
    uint32_t rachFailures = 0;
};

std::map<uint32_t, UEMetrics> ueMetrics;

// Completed packet records for CSV output
std::vector<NbiotPacketRecord> completedPackets;
std::map<uint32_t, uint32_t> ueSequenceCounters;

// Active transmission tracking
std::map<uint32_t, bool> activeTransmitters;
std::map<uint32_t, Time> transmissionEndTimes;

// Global parameters
double globalAppPeriodSeconds = 30.0;
bool enableEnvironmentalModeling = true;

// Inter-cell interferer nodes
NodeContainer interCellInterferers;

// LTE data traffic interferers (mobile users causing in-band interference)
NodeContainer lteDataInterferers;

// ============================================================================
// ENVIRONMENT SETUP FUNCTIONS
// ============================================================================

/**
 * Set up inter-cell interference sources
 * Models neighboring eNBs at standard ISD distances
 */
void
SetupInterCellInterference(double cellRadius, uint32_t nNeighborCells)
{
    std::cout << "  Setting up " << nNeighborCells << " neighboring cells for interference..."
              << std::endl;

    double isd = cellRadius * 2.0 * std::sqrt(3.0); // Inter-site distance for hexagonal layout

    for (uint32_t i = 0; i < nNeighborCells; i++)
    {
        double angle = 2.0 * M_PI * i / nNeighborCells;
        Vector pos(isd * std::cos(angle), isd * std::sin(angle), 30.0);
        std::cout << "    Neighbor cell " << i << " at (" << pos.x << ", " << pos.y << ")"
                  << std::endl;
    }

    // Configure channel with inter-cell parameters
    if (globalChannel)
    {
        globalChannel->SetNumNeighborCells(nNeighborCells);
        globalChannel->SetInterCellDistance(isd);
    }
}

/**
 * Set up mobile LTE data traffic interferers
 * Models other mobile devices in the cell causing in-band interference
 * These represent "normal" LTE traffic sharing the band with NB-IoT
 */
void
SetupLTEDataInterferers(double cellRadius, uint32_t nInterferers)
{
    lteDataInterferers.Create(nInterferers);

    MobilityHelper mobility;
    mobility.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel",
        "Bounds",
        RectangleValue(Rectangle(-cellRadius, cellRadius, -cellRadius, cellRadius)),
        "Speed",
        StringValue("ns3::UniformRandomVariable[Min=0.5|Max=1.5]"),
        "Distance",
        DoubleValue(20.0));

    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-cellRadius * 0.8));
    xRand->SetAttribute("Max", DoubleValue(cellRadius * 0.8));
    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-cellRadius * 0.8));
    yRand->SetAttribute("Max", DoubleValue(cellRadius * 0.8));

    for (uint32_t i = 0; i < nInterferers; i++)
    {
        allocator->Add(Vector(xRand->GetValue(), yRand->GetValue(), 1.5));
    }

    mobility.SetPositionAllocator(allocator);
    mobility.Install(lteDataInterferers);

    std::cout << "  Deployed " << nInterferers << " mobile LTE data interferers (human mobility)"
              << std::endl;
}

/**
 * Initialize hardware variability per UE
 * Models real-world differences in NB-IoT module performance
 */
std::map<uint32_t, double> ueAntennaGains;
std::map<uint32_t, double> ueNoiseFigures;
std::map<uint32_t, double> ueHardwareVariance;

void
InitializeHardwareVariability(uint32_t nUEs, NodeContainer& ueNodes)
{
    Ptr<UniformRandomVariable> gainVar = CreateObject<UniformRandomVariable>();
    gainVar->SetAttribute("Min", DoubleValue(-2.0));
    gainVar->SetAttribute("Max", DoubleValue(2.0));

    Ptr<NormalRandomVariable> nfVar = CreateObject<NormalRandomVariable>();
    nfVar->SetAttribute("Mean", DoubleValue(6.0));
    nfVar->SetAttribute("Variance", DoubleValue(1.0));

    Ptr<NormalRandomVariable> hwVar = CreateObject<NormalRandomVariable>();
    hwVar->SetAttribute("Mean", DoubleValue(0.0));
    hwVar->SetAttribute("Variance", DoubleValue(0.5));

    for (uint32_t i = 0; i < nUEs; i++)
    {
        uint32_t nodeId = ueNodes.Get(i)->GetId();
        ueAntennaGains[nodeId] = gainVar->GetValue();
        ueNoiseFigures[nodeId] = nfVar->GetValue();
        ueHardwareVariance[nodeId] = hwVar->GetValue();
    }
}

// ============================================================================
// PRE-TX RSRP MEASUREMENT (Equivalent to LoRaWAN's SamplePreTxRssi)
// ============================================================================

/**
 * Sample RSRP/RSRQ at UE location before uplink transmission
 * In NB-IoT, the UE measures Reference Signal (RS) from eNB
 * and can estimate interference from other cells
 */
struct RsrpSample
{
    double rsrp_dBm;      // Reference Signal Received Power
    double rsrq_dB;       // Reference Signal Received Quality
    double totalRssi_dBm; // Total RSSI (signal + interference + noise)
};

RsrpSample
SamplePreTxRsrp(uint32_t ueNodeId, Time currentTime)
{
    Vector uePos = nodePositions[ueNodeId];
    Vector enbPos = nodePositions[0]; // eNB at node 0

    // Calculate path loss from serving eNB
    Ptr<ConstantPositionMobilityModel> ueMobility = CreateObject<ConstantPositionMobilityModel>();
    ueMobility->SetPosition(uePos);
    Ptr<ConstantPositionMobilityModel> enbMobility = CreateObject<ConstantPositionMobilityModel>();
    enbMobility->SetPosition(enbPos);

    // RSRP from serving cell (eNB TX power for RS is lower than total)
    double enbRsPower_dBm = 15.3; // Reference signal power per RE (typical)
    double rsrp = globalChannel->GetRxPower(enbRsPower_dBm, enbMobility, ueMobility);

    // Apply hardware variability
    if (enableEnvironmentalModeling &&
        ueHardwareVariance.find(ueNodeId) != ueHardwareVariance.end())
    {
        rsrp += ueHardwareVariance[ueNodeId];
    }

    // Calculate total RSSI (includes interference + noise)
    double thermalNoise_dBm = NbiotThermalNoise_dBm(); // ~-128.4 dBm for 180 kHz
    double totalRssi_W = DbmToW(thermalNoise_dBm);

    // Add inter-cell interference
    double interCell_dBm = globalChannel->CalculateInterCellInterference(uePos, currentTime);
    totalRssi_W += DbmToW(interCell_dBm);

    // Add in-band LTE data traffic interference
    if (enableEnvironmentalModeling && lteDataInterferers.GetN() > 0)
    {
        Ptr<UniformRandomVariable> lteTrafficVar = CreateObject<UniformRandomVariable>();
        lteTrafficVar->SetAttribute("Min", DoubleValue(-135.0));
        lteTrafficVar->SetAttribute("Max", DoubleValue(-115.0));

        double loadFactor = 0.5 + 0.3 * std::sin(currentTime.GetSeconds() / 60.0);
        double lteInterference = lteTrafficVar->GetValue() + 10.0 * std::log10(loadFactor);
        totalRssi_W += DbmToW(lteInterference);
    }

    // Add serving cell signal to total RSSI
    totalRssi_W += DbmToW(rsrp);

    double totalRssi_dBm = WToDbm(totalRssi_W);

    // RSRQ = N × RSRP / RSSI (N=1 for NB-IoT single PRB)
    double rsrq = rsrp - totalRssi_dBm;           // In dB: RSRQ_dB = RSRP_dBm - RSSI_dBm (for N=1)
    rsrq = std::max(-20.0, std::min(-3.0, rsrq)); // 3GPP range

    // Add measurement noise
    Ptr<NormalRandomVariable> noise = CreateObject<NormalRandomVariable>();
    noise->SetAttribute("Mean", DoubleValue(0.0));
    noise->SetAttribute("Variance", DoubleValue(1.0));

    RsrpSample sample;
    sample.rsrp_dBm = rsrp + noise->GetValue() * 0.5;
    sample.rsrq_dB = rsrq;
    sample.totalRssi_dBm = totalRssi_dBm;

    return sample;
}

// ============================================================================
// ENERGY CALCULATION
// ============================================================================

/**
 * Calculate energy for one NB-IoT transmission cycle
 * Includes: PSM wakeup, RACH, data TX, ACK RX, inactivity timeout
 */
double
CalculateNbiotEnergy(uint8_t mcs,
                     int repetitions,
                     double txPower_dBm,
                     ResourceUnit ru,
                     int rachAttempts,
                     uint32_t payloadBytes,
                     double psmSleepTime_s,
                     CELevel ceLevel)
{
    if (globalEnergyModel)
    {
        return globalEnergyModel->CalculateTransmissionCycleEnergy(txPower_dBm,
                                                                   mcs,
                                                                   repetitions,
                                                                   ru,
                                                                   rachAttempts,
                                                                   payloadBytes,
                                                                   psmSleepTime_s,
                                                                   ceLevel);
    }

    // Fallback calculation
    double airtime_ms = CalculateAirtime_ms(mcs, repetitions, ru, payloadBytes);
    double txCurrent = GetTxCurrent_mA(txPower_dBm);
    double voltage = 3.6;

    // TX energy: P(mW) = I(mA) × V(V), E(mJ) = P(mW) × t(ms) / 1000
    double txEnergy = txCurrent * voltage * airtime_ms / 1000.0;

    // RACH energy: 80 mA × 3.6V × 5.6ms per attempt
    double rachEnergy = 80.0 * voltage * 5.6 * rachAttempts / 1000.0;

    // RX energy (RAR + ACK)
    double rxEnergy = 46.0 * voltage * 10.0 / 1000.0;

    // Inactivity timer energy: 6 mA × 3.6V × 10s
    double idleEnergy = 6.0 * voltage * 10.0; // 10s in seconds, result in mJ

    return txEnergy + rachEnergy + rxEnergy + idleEnergy;
}

// ============================================================================
// LINK ADAPTATION UPDATE FUNCTIONS
// ============================================================================

/**
 * Default NB-IoT Link Adaptation (3GPP standard)
 * Uses SINR-based MCS selection and CE level management
 * Equivalent to Classical ADR in LoRaWAN
 */
void
UpdateDefaultLinkAdaptation(uint32_t ueId,
                            double sinr,
                            bool packetSuccess,
                            double rsrp,
                            bool rachSuccess,
                            int rachAttempts)
{
    constexpr double kMcsSelectionMarginDb = 5.0;
    constexpr double kTargetSinrMarginDb = 8.0;
    constexpr double kRepetitionReductionMarginDb = 12.0;
    constexpr double kMinTxPowerDbm = 14.0;

    // Track SINR history for averaging
    static std::map<uint32_t, std::deque<double>> sinrHistMap;
    auto& sinrHist = sinrHistMap[ueId];
    sinrHist.push_back(sinr);
    if (sinrHist.size() > 20)
    {
        sinrHist.pop_front();
    }

    // Average SINR over last 20 measurements
    double avgSINR = 0;
    for (auto s : sinrHist)
    {
        avgSINR += s;
    }
    avgSINR /= sinrHist.size();

    // MCS selection based on average SINR + repetition gain
    int reps = currentRepetitions[ueId];
    double effectiveSINR = avgSINR + GetRepetitionGainDb(reps);

    // Select MCS with an explicit reliability margin rather than chasing
    // the highest rate that barely fits the current average SINR.
    uint8_t newMCS = 0;
    for (int m = 10; m >= 0; m--)
    {
        if (effectiveSINR >= MCS_TABLE[m].requiredSINR_dB + kMcsSelectionMarginDb)
        {
            newMCS = m;
            break;
        }
    }

    // Adjust repetitions based on CE level
    if (!packetSuccess && sinrHist.size() >= 10)
    {
        double recentPdr = 0;
        int count = 0;
        for (auto& h : deviceHistories[ueId].packetResults)
        {
            if (h)
            {
                recentPdr += 1.0;
            }
            count++;
        }
        recentPdr = count > 0 ? recentPdr / count : 0.5;

        if (recentPdr < 0.7)
        {
            // Increase repetitions
            reps = std::min(128, reps * 2);
        }
    }
    else if (packetSuccess &&
             effectiveSINR > MCS_TABLE[newMCS].requiredSINR_dB + kRepetitionReductionMarginDb)
    {
        // Large SINR margin: decrease repetitions
        reps = std::max(1, reps / 2);
    }

    // Power control: aim for target SINR
    double targetSINR = MCS_TABLE[newMCS].requiredSINR_dB + kTargetSinrMarginDb;
    double sinrDiff = targetSINR - avgSINR;
    double newTP = currentTP[ueId] + sinrDiff * 0.5; // Proportional control
    newTP = std::max(kMinTxPowerDbm, std::min(23.0, newTP));

    currentMCS[ueId] = newMCS;
    currentRepetitions[ueId] = reps;
    currentTP[ueId] = newTP;
    nodeTxPowers[ueId] = newTP;

    NS_LOG_INFO("Default LA - UE "
                << ueId << ": MCS=" << (int)newMCS << " Reps=" << reps << " TP=" << newTP << " dBm"
                << " avgSINR=" << std::fixed << std::setprecision(1) << avgSINR << " dB");
}

/**
 * DDQN-PER Link Adaptation for NB-IoT
 * Uses optimized DDQN with extended action space
 */
void
UpdateDDQNADR(uint32_t ueId,
              double sinr,
              bool packetSuccess,
              double rsrp,
              bool rachSuccess,
              int rachAttempts)
{
    auto& history = deviceHistories[ueId];
    auto& metrics = ueMetrics[ueId];

    // Update history
    history.addRsrp(rsrp);
    history.addSinr(sinr);
    history.addPacketResult(packetSuccess, Simulator::Now().GetSeconds());
    history.addRachResult(rachSuccess, rachAttempts);

    // Calculate RSRQ from RSRP and interference
    RsrpSample rsrpSample = SamplePreTxRsrp(ueId, Simulator::Now());

    double cellLoad = globalEnb ? globalEnb->GetCellLoad() : 0.3;
    double batteryLevel = 1.0 - (metrics.totalEnergyConsumed_mJ / (2400.0 * 3.6 * 3600.0));
    batteryLevel = std::max(0.0, std::min(1.0, batteryLevel));

    // Build state
    auto& agent = optimizedAgents[ueId];
    auto state = agent->buildRealisticState(history,
                                            currentMCS[ueId],
                                            currentRepetitions[ueId],
                                            currentTP[ueId],
                                            currentRU[ueId],
                                            cellLoad,
                                            batteryLevel,
                                            Simulator::Now().GetSeconds());

    // Estimate congestion
    double congestion = estimateCellCongestion(rsrp,
                                               history.getRsrpVariance(),
                                               history.getRecentPdr(),
                                               ueMetrics.size(),
                                               history.getAvgRachAttempts(),
                                               cellLoad);

    // Calculate energy for this transmission
    double energy = CalculateNbiotEnergy(currentMCS[ueId],
                                         currentRepetitions[ueId],
                                         currentTP[ueId],
                                         static_cast<ResourceUnit>(currentRU[ueId]),
                                         rachAttempts,
                                         20,
                                         globalAppPeriodSeconds,
                                         CELevel::CE0);

    // Calculate reward
    double reward =
        calculateCongestionAdaptiveReward(ueId,
                                          packetSuccess,
                                          rachSuccess,
                                          rachAttempts,
                                          rsrp,
                                          sinr,
                                          history.getRsrpVariance(),
                                          history.consecutiveLosses,
                                          currentMCS[ueId],
                                          currentTP[ueId],
                                          currentRepetitions[ueId],
                                          currentRU[ueId],
                                          history.getRecentPdr(),
                                          history.getPdrTrend(),
                                          Simulator::Now().GetSeconds() - history.lastSuccessTime,
                                          congestion,
                                          energy);

    // Select action
    int action = agent->selectAction(state, ueId, history, Simulator::Now().GetSeconds(), cellLoad);

    // Decode action to NB-IoT parameters
    auto result = ExtendedActionSpace::decodeAction(action,
                                                    currentMCS[ueId],
                                                    currentRepetitions[ueId],
                                                    static_cast<ResourceUnit>(currentRU[ueId]),
                                                    currentTP[ueId]);

    // Apply MCS ceiling based on SINR
    uint8_t mcsCeiling = getMandatoryMCSCeiling(sinr, result.newRepetitions);
    if (result.newMCS > mcsCeiling)
    {
        NS_LOG_INFO("DDQN MCS ceiling enforced: " << result.newMCS << " → " << (int)mcsCeiling);
        result.newMCS = mcsCeiling;
    }

    // Store previous state for experience
    static std::map<uint32_t, std::vector<double>> previousStates;
    static std::map<uint32_t, int> previousActions;

    if (previousStates.find(ueId) != previousStates.end())
    {
        agent->addExperience(previousStates[ueId], previousActions[ueId], reward, state, false);
        agent->train(32);
    }

    previousStates[ueId] = state;
    previousActions[ueId] = action;

    // Apply new parameters
    currentMCS[ueId] = static_cast<uint8_t>(result.newMCS);
    currentRepetitions[ueId] = result.newRepetitions;
    currentRU[ueId] = static_cast<uint8_t>(result.newRU);
    // TX power managed by extended action space result
    if (result.newTP >= 0.0 && result.newTP <= 23.0)
    {
        currentTP[ueId] = result.newTP;
        nodeTxPowers[ueId] = result.newTP;
    }

    NS_LOG_INFO("DDQN LA - UE " << ueId << ": MCS=" << (int)currentMCS[ueId]
                                << " Reps=" << currentRepetitions[ueId]
                                << " RU=" << (int)currentRU[ueId] << " TP=" << currentTP[ueId]
                                << " ε=" << std::fixed << std::setprecision(3)
                                << agent->getEpsilon() << " R=" << std::setprecision(2) << reward);
}

/**
 * PPO Link Adaptation for NB-IoT
 */
void
UpdatePPOADR(uint32_t ueId,
             double sinr,
             bool packetSuccess,
             double rsrp,
             bool rachSuccess,
             int rachAttempts)
{
    auto& history = deviceHistories[ueId];
    auto& metrics = ueMetrics[ueId];

    // Update history
    history.addRsrp(rsrp);
    history.addSinr(sinr);
    history.addPacketResult(packetSuccess, Simulator::Now().GetSeconds());
    history.addRachResult(rachSuccess, rachAttempts);

    double cellLoad = globalEnb ? globalEnb->GetCellLoad() : 0.3;
    double batteryLevel = 1.0 - (metrics.totalEnergyConsumed_mJ / (2400.0 * 3.6 * 3600.0));
    batteryLevel = std::max(0.0, std::min(1.0, batteryLevel));

    // Build state (same 14-feature vector)
    std::vector<double> state(14);
    state[0] = std::max(0.0, std::min(1.0, (history.getLastRsrp() + 140.0) / 96.0));
    state[1] = std::max(0.0, std::min(1.0, (history.getLastSinr() + 12.0) / 42.0));
    state[2] = std::min(1.0, history.getRsrpVariance() / 20.0);
    state[3] = std::max(-1.0, std::min(1.0, history.getSinrTrend() / 10.0));
    state[4] = currentMCS[ueId] / 10.0;
    state[5] = std::log2(currentRepetitions[ueId]) / 7.0;
    state[6] = currentTP[ueId] / 23.0;
    state[7] = currentRU[ueId] / 4.0;
    state[8] = history.getRecentPdr();
    state[9] = std::max(-1.0, std::min(1.0, history.getPdrTrend()));
    state[10] = cellLoad;
    state[11] = batteryLevel;
    state[12] = history.getRachSuccessRate();
    state[13] = std::min(1.0, history.consecutiveLosses / 10.0);

    // Congestion estimate
    double congestion = estimateCellCongestion(rsrp,
                                               history.getRsrpVariance(),
                                               history.getRecentPdr(),
                                               ueMetrics.size(),
                                               history.getAvgRachAttempts(),
                                               cellLoad);

    // Energy for this cycle
    double energy = CalculateNbiotEnergy(currentMCS[ueId],
                                         currentRepetitions[ueId],
                                         currentTP[ueId],
                                         static_cast<ResourceUnit>(currentRU[ueId]),
                                         rachAttempts,
                                         20,
                                         globalAppPeriodSeconds,
                                         CELevel::CE0);

    // Calculate reward
    double reward =
        calculateCongestionAdaptiveReward(ueId,
                                          packetSuccess,
                                          rachSuccess,
                                          rachAttempts,
                                          rsrp,
                                          sinr,
                                          history.getRsrpVariance(),
                                          history.consecutiveLosses,
                                          currentMCS[ueId],
                                          currentTP[ueId],
                                          currentRepetitions[ueId],
                                          currentRU[ueId],
                                          history.getRecentPdr(),
                                          history.getPdrTrend(),
                                          Simulator::Now().GetSeconds() - history.lastSuccessTime,
                                          congestion,
                                          energy);

    // Select action
    auto& ppoAgent = ppoAgents[ueId];
    auto action = ppoAgent->selectAction(state);

    // Apply MCS ceiling
    uint8_t mcsCeiling = getMandatoryMCSCeiling(sinr, action.repetitions);
    if (action.mcs > mcsCeiling)
    {
        action.mcs = mcsCeiling;
    }

    // Store transition
    std::vector<double> actionVec = {static_cast<double>(action.mcs),
                                     std::log2(static_cast<double>(action.repetitions)),
                                     action.txPower};
    actionVec.push_back(
        ppoAgent->evaluateLogProb(state, {actionVec[0], actionVec[1], actionVec[2]}));

    static std::map<uint32_t, std::vector<double>> prevStates;
    static std::map<uint32_t, std::vector<double>> prevActions;

    if (prevStates.find(ueId) != prevStates.end())
    {
        ppoAgent->addTransition(prevStates[ueId], prevActions[ueId], reward, state, false);
    }

    prevStates[ueId] = state;
    prevActions[ueId] = actionVec;

    // Apply parameters
    currentMCS[ueId] = action.mcs;
    currentRepetitions[ueId] = action.repetitions;
    currentTP[ueId] = action.txPower;
    nodeTxPowers[ueId] = action.txPower;

    NS_LOG_INFO("PPO LA - UE " << ueId << ": MCS=" << (int)action.mcs
                               << " Reps=" << action.repetitions << " TP=" << action.txPower
                               << " R=" << std::fixed << std::setprecision(2) << reward);
}

/**
 * MARL Link Adaptation for NB-IoT
 * Leverages eNB coordination signals
 */
void
UpdateMARLADR(uint32_t ueId,
              double sinr,
              bool packetSuccess,
              double rsrp,
              bool rachSuccess,
              int rachAttempts)
{
    auto& history = deviceHistories[ueId];
    auto& metrics = ueMetrics[ueId];

    // Update history
    history.addRsrp(rsrp);
    history.addSinr(sinr);
    history.addPacketResult(packetSuccess, Simulator::Now().GetSeconds());
    history.addRachResult(rachSuccess, rachAttempts);

    double cellLoad = globalEnb ? globalEnb->GetCellLoad() : 0.3;
    double batteryLevel = 1.0 - (metrics.totalEnergyConsumed_mJ / (2400.0 * 3.6 * 3600.0));
    batteryLevel = std::max(0.0, std::min(1.0, batteryLevel));

    // Calculate cell-wide averages for MARL coordination
    double avgCellMCS = 0.0;
    double avgCellReps = 0.0;
    uint32_t numActive = 0;
    for (auto& [id, mcs] : currentMCS)
    {
        avgCellMCS += mcs;
        avgCellReps += currentRepetitions[id];
        numActive++;
    }
    if (numActive > 0)
    {
        avgCellMCS /= numActive;
        avgCellReps /= numActive;
    }

    // Build MARL state with coordination features
    auto& agent = marlAgents[ueId];
    auto state = agent->buildMARLState(history,
                                       currentMCS[ueId],
                                       currentRepetitions[ueId],
                                       currentTP[ueId],
                                       currentRU[ueId],
                                       cellLoad,
                                       batteryLevel,
                                       Simulator::Now().GetSeconds(),
                                       numActive,
                                       avgCellMCS,
                                       avgCellReps);

    // Update coordination signals from other agents
    for (auto& [otherId, otherAgent] : marlAgents)
    {
        if (otherId != ueId)
        {
            agent->updateCoordination(otherId, 0, currentMCS[otherId], currentRepetitions[otherId]);
        }
    }

    // Congestion estimation
    double congestion = estimateCellCongestion(rsrp,
                                               history.getRsrpVariance(),
                                               history.getRecentPdr(),
                                               numActive,
                                               history.getAvgRachAttempts(),
                                               cellLoad);

    // Energy calculation
    double energy = CalculateNbiotEnergy(currentMCS[ueId],
                                         currentRepetitions[ueId],
                                         currentTP[ueId],
                                         static_cast<ResourceUnit>(currentRU[ueId]),
                                         rachAttempts,
                                         20,
                                         globalAppPeriodSeconds,
                                         CELevel::CE0);

    // Reward
    double reward =
        calculateCongestionAdaptiveReward(ueId,
                                          packetSuccess,
                                          rachSuccess,
                                          rachAttempts,
                                          rsrp,
                                          sinr,
                                          history.getRsrpVariance(),
                                          history.consecutiveLosses,
                                          currentMCS[ueId],
                                          currentTP[ueId],
                                          currentRepetitions[ueId],
                                          currentRU[ueId],
                                          history.getRecentPdr(),
                                          history.getPdrTrend(),
                                          Simulator::Now().GetSeconds() - history.lastSuccessTime,
                                          congestion,
                                          energy);

    // Action selection
    int action = agent->selectAction(state, ueId, history, Simulator::Now().GetSeconds(), cellLoad);

    // Decode action
    auto result = ExtendedActionSpace::decodeAction(action,
                                                    currentMCS[ueId],
                                                    currentRepetitions[ueId],
                                                    static_cast<ResourceUnit>(currentRU[ueId]),
                                                    currentTP[ueId]);

    // MCS ceiling
    uint8_t mcsCeiling = getMandatoryMCSCeiling(sinr, result.newRepetitions);
    if (result.newMCS > mcsCeiling)
    {
        result.newMCS = mcsCeiling;
    }

    // Experience storage
    static std::map<uint32_t, std::vector<double>> prevStates;
    static std::map<uint32_t, int> prevActions;

    if (prevStates.find(ueId) != prevStates.end())
    {
        agent->addExperience(prevStates[ueId], prevActions[ueId], reward, state, false);
        agent->train(32);
    }

    prevStates[ueId] = state;
    prevActions[ueId] = action;

    // Apply
    currentMCS[ueId] = static_cast<uint8_t>(result.newMCS);
    currentRepetitions[ueId] = result.newRepetitions;
    currentRU[ueId] = static_cast<uint8_t>(result.newRU);
    if (result.newTP >= 0.0 && result.newTP <= 23.0)
    {
        currentTP[ueId] = result.newTP;
        nodeTxPowers[ueId] = result.newTP;
    }

    NS_LOG_INFO("MARL LA - UE " << ueId << ": MCS=" << (int)currentMCS[ueId]
                                << " Reps=" << currentRepetitions[ueId]
                                << " RU=" << (int)currentRU[ueId] << " TP=" << currentTP[ueId]
                                << " ε=" << std::fixed << std::setprecision(3)
                                << agent->getEpsilon() << " R=" << std::setprecision(2) << reward);
}

// ============================================================================
// SIMULATION CALLBACKS
// ============================================================================

/**
 * Simulate one NB-IoT uplink transmission cycle for a UE
 * This is the core simulation loop called periodically
 */
void
SimulateUplinkTransmission(uint32_t ueIdx)
{
    uint32_t ueId = ueDevices[ueIdx]->GetUeId();
    double currentTime = Simulator::Now().GetSeconds();

    // Get current parameters
    uint8_t mcs = currentMCS[ueId];
    int reps = currentRepetitions[ueId];
    double tp = currentTP[ueId];
    ResourceUnit ru = static_cast<ResourceUnit>(currentRU[ueId]);

    // 1. Pre-TX RSRP measurement
    RsrpSample preTxMeasurement = SamplePreTxRsrp(ueId, Simulator::Now());

    // 2. RACH procedure
    bool rachSuccess = true;
    int rachAttempts = 1;

    if (globalEnb)
    {
        // Determine CE level from coupling loss
        Vector uePos = nodePositions[ueId];
        Vector enbPos = nodePositions[0];
        double distance =
            std::sqrt(std::pow(uePos.x - enbPos.x, 2) + std::pow(uePos.y - enbPos.y, 2));

        // Simplified coupling loss from RSRP
        double couplingLoss = 15.3 - preTxMeasurement.rsrp_dBm; // eNB RS power - RSRP
        CELevel ceLevel = DetermineCELevel(couplingLoss);

        // Process RACH at eNB
        auto rachResult = globalEnb->ProcessRACH(ueId, tp, ceLevel, Simulator::Now());
        rachSuccess = rachResult.success;
        rachAttempts = rachResult.attemptsUsed;
    }

    // 3. Uplink data transmission
    bool dataSuccess = false;
    double sinr = -20.0;
    double rsrp = preTxMeasurement.rsrp_dBm;
    double rsrq = preTxMeasurement.rsrq_dB;

    if (rachSuccess)
    {
        // Calculate received power at eNB
        Ptr<ConstantPositionMobilityModel> ueMob = CreateObject<ConstantPositionMobilityModel>();
        ueMob->SetPosition(nodePositions[ueId]);
        Ptr<ConstantPositionMobilityModel> enbMob = CreateObject<ConstantPositionMobilityModel>();
        enbMob->SetPosition(nodePositions[0]);

        double rxPower = globalChannel->GetRxPower(tp, ueMob, enbMob);

        // Apply antenna gains
        if (ueAntennaGains.find(ueId) != ueAntennaGains.end())
        {
            rxPower += ueAntennaGains[ueId];
        }

        // Process uplink at eNB
        if (globalEnb)
        {
            auto uplinkResult =
                globalEnb->ProcessUplink(ueId, rxPower, mcs, reps, ru, Simulator::Now());
            dataSuccess = uplinkResult.success;
            sinr = uplinkResult.sinr_dB;
            rsrp = uplinkResult.rsrp_dBm;
            rsrq = uplinkResult.rsrq_dB;
        }
        else
        {
            // Simplified SINR calculation
            double interCellInterf =
                globalChannel->CalculateInterCellInterference(nodePositions[ueId],
                                                              Simulator::Now());
            sinr = globalChannel->CalculateSINR(rxPower, interCellInterf, 5.0);

            // Apply repetition + RU gain
            double effectiveSINR = sinr + GetRepetitionGainDb(reps) +
                                   RU_CONFIGS[static_cast<int>(ru)].processingGainDb;
            dataSuccess = (effectiveSINR >= MCS_TABLE[mcs].requiredSINR_dB);
        }
    }

    bool overallSuccess = rachSuccess && dataSuccess;

    // 4. Calculate energy consumption
    double energy = CalculateNbiotEnergy(mcs,
                                         reps,
                                         tp,
                                         ru,
                                         rachAttempts,
                                         20,
                                         globalAppPeriodSeconds,
                                         CELevel::CE0);

    // 5. Update metrics
    ueMetrics[ueId].packetsSent++;
    ueMetrics[ueId].totalEnergyConsumed_mJ += energy;
    ueMetrics[ueId].totalRachAttempts += rachAttempts;
    if (overallSuccess)
    {
        ueMetrics[ueId].packetsReceived++;
    }
    else if (!rachSuccess)
    {
        ueMetrics[ueId].rachFailures++;
    }
    ueMetrics[ueId].lastPDR =
        ueMetrics[ueId].packetsSent > 0
            ? static_cast<double>(ueMetrics[ueId].packetsReceived) / ueMetrics[ueId].packetsSent
            : 0.0;

    // 6. Record packet
    NbiotPacketRecord record;
    record.timestamp = currentTime;
    record.ueId = ueId;
    record.sequenceNum = ueSequenceCounters[ueId]++;
    record.txPowerDbm = tp;
    record.mcs = mcs;
    record.repetitions = reps;
    record.ru = ru;
    record.rsrp_dBm = rsrp;
    record.rsrq_dB = rsrq;
    record.sinr_dB = sinr;
    record.preTxRssi_dBm = preTxMeasurement.totalRssi_dBm;
    record.rachSuccess = rachSuccess;
    record.rachAttempts = rachAttempts;
    record.dataSuccess = dataSuccess;
    record.airtime_ms = CalculateAirtime_ms(mcs, reps, ru, 20);
    record.energyConsumed_mJ = energy;
    record.ceLevel = DetermineCELevel(15.3 - rsrp);
    record.couplingLoss_dB = 15.3 - rsrp;
    record.payloadBytes = 20;
    record.cellLoad = globalEnb ? globalEnb->GetCellLoad() : 0.0;
    record.collisionFlag = !rachSuccess && rachAttempts > 1;
    completedPackets.push_back(record);

    // 7. Update RL agent
    switch (currentADRMethod)
    {
    case ADRMethod::ON:
        UpdateDefaultLinkAdaptation(ueId, sinr, overallSuccess, rsrp, rachSuccess, rachAttempts);
        break;
    case ADRMethod::DDQN:
        UpdateDDQNADR(ueId, sinr, overallSuccess, rsrp, rachSuccess, rachAttempts);
        break;
    case ADRMethod::PPO:
        UpdatePPOADR(ueId, sinr, overallSuccess, rsrp, rachSuccess, rachAttempts);
        break;
    case ADRMethod::MARL:
        UpdateMARLADR(ueId, sinr, overallSuccess, rsrp, rachSuccess, rachAttempts);
        break;
    case ADRMethod::OFF:
    default:
        // No adaptation, just update history
        deviceHistories[ueId].addRsrp(rsrp);
        deviceHistories[ueId].addSinr(sinr);
        deviceHistories[ueId].addPacketResult(overallSuccess, currentTime);
        deviceHistories[ueId].addRachResult(rachSuccess, rachAttempts);
        break;
    }

    // 8. Schedule next transmission
    Simulator::Schedule(Seconds(globalAppPeriodSeconds), &SimulateUplinkTransmission, ueIdx);
}

// ============================================================================
// CSV OUTPUT
// ============================================================================

/**
 * Write comprehensive CSV output (equivalent to LoRaWAN version)
 */
void
WriteCSVOutput(const std::string& filename)
{
    std::ofstream csv(filename);
    csv << "timestamp,ue_id,sequence_num,tx_power_dbm,mcs,repetitions,ru_type,"
        << "rsrp_dbm,rsrq_db,sinr_db,pre_tx_rssi_dbm,"
        << "rach_success,rach_attempts,data_success,"
        << "airtime_ms,energy_consumed_mj,ce_level,coupling_loss_db,"
        << "payload_bytes,cell_load,collision_flag" << std::endl;

    for (const auto& record : completedPackets)
    {
        csv << std::fixed << std::setprecision(3) << record.timestamp << "," << record.ueId << ","
            << record.sequenceNum << "," << record.txPowerDbm << "," << (int)record.mcs << ","
            << record.repetitions << "," << (int)record.ru << "," << record.rsrp_dBm << ","
            << record.rsrq_dB << "," << record.sinr_dB << "," << record.preTxRssi_dBm << ","
            << record.rachSuccess << "," << record.rachAttempts << "," << record.dataSuccess << ","
            << record.airtime_ms << "," << record.energyConsumed_mJ << "," << (int)record.ceLevel
            << "," << record.couplingLoss_dB << "," << record.payloadBytes << "," << record.cellLoad
            << "," << record.collisionFlag << std::endl;
    }

    csv.close();
    std::cout << "CSV output written: " << filename << " (" << completedPackets.size()
              << " records)" << std::endl;
}

/**
 * Write environment visualization CSV
 */
void
WriteEnvironmentVisualization(const std::string& baseFilename, double radius, uint32_t nUEs)
{
    std::string envFile = baseFilename + "_environment.csv";
    std::ofstream csv(envFile);
    csv << "type,id,x,y,z,details" << std::endl;

    // eNB position
    csv << "eNB,0,0,0,30,serving_cell" << std::endl;

    // UE positions
    for (auto& [id, pos] : nodePositions)
    {
        if (id == 0)
        {
            continue; // Skip eNB
        }
        double distance = std::sqrt(pos.x * pos.x + pos.y * pos.y);
        csv << "UE," << id << "," << pos.x << "," << pos.y << "," << pos.z << ",dist=" << std::fixed
            << std::setprecision(0) << distance << "m" << std::endl;
    }

    // Neighbor cells
    double isd = radius * 2.0 * std::sqrt(3.0);
    for (int i = 0; i < 6; i++)
    {
        double angle = 2.0 * M_PI * i / 6;
        csv << "neighbor_eNB," << i << "," << isd * std::cos(angle) << "," << isd * std::sin(angle)
            << ",30,"
            << "interferer" << std::endl;
    }

    csv.close();
    std::cout << "Environment CSV written: " << envFile << std::endl;
}

// ============================================================================
// INITIAL PARAMETER SELECTION
// ============================================================================

/**
 * Get optimal initial MCS based on distance
 */
uint8_t
getInitialMCSForDistance(double distance)
{
    if (distance < 200)
    {
        return 8; // Close: high MCS
    }
    if (distance < 500)
    {
        return 5; // Medium: mid MCS
    }
    if (distance < 1000)
    {
        return 2; // Far: low MCS
    }
    return 0; // Very far: lowest MCS
}

/**
 * Get optimal initial repetitions based on distance
 */
int
getInitialRepetitionsForDistance(double distance)
{
    if (distance < 200)
    {
        return 1;
    }
    if (distance < 500)
    {
        return 2;
    }
    if (distance < 1000)
    {
        return 8;
    }
    if (distance < 2000)
    {
        return 32;
    }
    return 64;
}

/**
 * Get optimal initial TX power based on distance
 */
double
getInitialTPForDistance(double distance)
{
    if (distance < 100)
    {
        return 14.0;
    }
    if (distance < 500)
    {
        return 20.0;
    }
    return 23.0;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

// Forward declaration
void RunSimulation(uint32_t nUEs,
                   double simulationTime,
                   double appPeriodSeconds,
                   double cellRadius,
                   const std::string& csvFileName,
                   ADRMethod adrMethod,
                   uint32_t nLteInterferers,
                   uint32_t nNeighborCells);

int
main(int argc, char* argv[])
{
    uint32_t nUEs = 10;
    double simulationTime = 300.0;
    double appPeriodSeconds = 30.0;
    double cellRadius = 1000.0;
    std::string csvFileName = "nbiot_dataset.csv";
    std::string adrModeStr = "off";
    bool environmentalModeling = true;
    uint32_t nLteInterferers = 8;
    uint32_t nNeighborCells = 6;

    CommandLine cmd;
    cmd.AddValue("nUEs", "Number of NB-IoT UEs", nUEs);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("appPeriod", "Uplink transmission period in seconds", appPeriodSeconds);
    cmd.AddValue("cellRadius", "Cell radius in meters", cellRadius);
    cmd.AddValue("csvFile", "Output CSV file name", csvFileName);
    cmd.AddValue("adr",
                 "Link adaptation mode: 'off', 'on', 'ddqn', 'ppo', 'marl', or 'all'",
                 adrModeStr);
    cmd.AddValue("environmental", "Enable environmental effects modeling", environmentalModeling);
    cmd.AddValue("lteInterferers", "Number of in-band LTE data interferers", nLteInterferers);
    cmd.AddValue("neighborCells",
                 "Number of neighbor cells for inter-cell interference",
                 nNeighborCells);
    cmd.Parse(argc, argv);

    enableEnvironmentalModeling = environmentalModeling;
    globalAppPeriodSeconds = appPeriodSeconds;

    LogComponentEnable("AdvancedDDQNPERADRNbiotExample", LOG_LEVEL_INFO);

    // Cleanup
    std::cout << "🧹 Cleaning up old nbiot_datasets folder..." << std::endl;
    system("rm -rf nbiot_datasets");
    system("mkdir -p nbiot_datasets");
    std::cout << "✅ Setup complete" << std::endl;

    std::cout << "\n=== NB-IoT RL Link Adaptation Simulation ===" << std::endl;
    std::cout << "Environmental modeling: "
              << (enableEnvironmentalModeling ? "ENABLED" : "DISABLED") << std::endl;
    if (enableEnvironmentalModeling)
    {
        std::cout << "  - Inter-cell interference (" << nNeighborCells << " neighbor cells)"
                  << std::endl;
        std::cout << "  - In-band LTE traffic (" << nLteInterferers << " mobile devices)"
                  << std::endl;
        std::cout << "  - Hardware variability modeling" << std::endl;
    }

    if (adrModeStr == "all")
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "RUNNING ALL LINK ADAPTATION METHODS" << std::endl;
        std::cout << "Simulation Time: " << simulationTime << " seconds" << std::endl;
        std::cout << "Number of UEs: " << nUEs << std::endl;
        std::cout << "App Period: " << appPeriodSeconds << " seconds" << std::endl;
        std::cout << "========================================\n" << std::endl;

        std::cout << "\n[1/5] Running with NO link adaptation..." << std::endl;
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/no_la_" + csvFileName,
                      ADRMethod::OFF,
                      nLteInterferers,
                      nNeighborCells);

        std::cout << "\n[2/5] Running with DEFAULT link adaptation..." << std::endl;
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/default_la_" + csvFileName,
                      ADRMethod::ON,
                      nLteInterferers,
                      nNeighborCells);

        std::cout << "\n[3/5] Running with DDQN-PER link adaptation..." << std::endl;
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/ddqn_la_" + csvFileName,
                      ADRMethod::DDQN,
                      nLteInterferers,
                      nNeighborCells);

        std::cout << "\n[4/5] Running with PPO link adaptation..." << std::endl;
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/ppo_la_" + csvFileName,
                      ADRMethod::PPO,
                      nLteInterferers,
                      nNeighborCells);

        std::cout << "\n[5/5] Running with MARL link adaptation..." << std::endl;
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/marl_la_" + csvFileName,
                      ADRMethod::MARL,
                      nLteInterferers,
                      nNeighborCells);

        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL METHODS COMPARISON COMPLETED" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    else
    {
        ADRMethod adrMethod;
        std::string filePrefix;
        if (adrModeStr == "on")
        {
            adrMethod = ADRMethod::ON;
            filePrefix = "default_la_";
        }
        else if (adrModeStr == "ddqn")
        {
            adrMethod = ADRMethod::DDQN;
            filePrefix = "ddqn_la_";
        }
        else if (adrModeStr == "ppo")
        {
            adrMethod = ADRMethod::PPO;
            filePrefix = "ppo_la_";
        }
        else if (adrModeStr == "marl")
        {
            adrMethod = ADRMethod::MARL;
            filePrefix = "marl_la_";
        }
        else
        {
            adrMethod = ADRMethod::OFF;
            filePrefix = "no_la_";
        }
        RunSimulation(nUEs,
                      simulationTime,
                      appPeriodSeconds,
                      cellRadius,
                      "nbiot_datasets/" + filePrefix + csvFileName,
                      adrMethod,
                      nLteInterferers,
                      nNeighborCells);
    }

    return 0;
}

// ============================================================================
// RUN SIMULATION
// ============================================================================

void
RunSimulation(uint32_t nUEs,
              double simulationTime,
              double appPeriodSeconds,
              double cellRadius,
              const std::string& csvFileName,
              ADRMethod adrMethod,
              uint32_t nLteInterferers,
              uint32_t nNeighborCells)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "Starting NB-IoT simulation:" << std::endl;
    std::cout << "  Duration: " << simulationTime << " seconds" << std::endl;
    std::cout << "  UEs: " << nUEs << std::endl;
    std::cout << "  App Period: " << appPeriodSeconds << " seconds" << std::endl;
    std::cout << "  Cell Radius: " << cellRadius << " m" << std::endl;
    std::string modeLabel = "OFF";
    if (adrMethod == ADRMethod::ON)
    {
        modeLabel = "DEFAULT (3GPP)";
    }
    else if (adrMethod == ADRMethod::DDQN)
    {
        modeLabel = "DDQN-PER";
    }
    else if (adrMethod == ADRMethod::PPO)
    {
        modeLabel = "PPO";
    }
    else if (adrMethod == ADRMethod::MARL)
    {
        modeLabel = "MARL";
    }
    std::cout << "  Link Adaptation: " << modeLabel << std::endl;
    std::cout << "================================================\n" << std::endl;

    // Clear all state
    completedPackets.clear();
    ueSequenceCounters.clear();
    activeTransmitters.clear();
    transmissionEndTimes.clear();
    ueDevices.clear();
    ddqnAgents.clear();
    optimizedAgents.clear();
    ppoAgents.clear();
    marlAgents.clear();
    ueMetrics.clear();
    deviceHistories.clear();
    nodePositions.clear();
    nodeTxPowers.clear();
    currentMCS.clear();
    currentRepetitions.clear();
    currentTP.clear();
    currentRU.clear();
    ueAntennaGains.clear();
    ueNoiseFigures.clear();
    ueHardwareVariance.clear();
    lteDataInterferers = NodeContainer();

    currentADRMethod = adrMethod;

    // Fixed seed
    RngSeedManager::SetSeed(12345);
    RngSeedManager::SetRun(1);

    // ========== Create Channel ==========
    NbiotHelper nbiotHelper;
    nbiotHelper.SetInterCellDistance(cellRadius * 2.0 * std::sqrt(3.0));
    nbiotHelper.SetNumNeighborCells(nNeighborCells);
    globalChannel = nbiotHelper.CreateChannel("urban_macro");
    globalEnergyModel = nbiotHelper.CreateEnergyModel();

    // ========== Create Nodes ==========
    NodeContainer enbNodes;
    enbNodes.Create(1);

    NodeContainer ueNodes;
    ueNodes.Create(nUEs);

    // ========== Position Setup ==========
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();

    // eNB at center, 30m height
    allocator->Add(Vector(0, 0, 30.0));
    nodePositions[enbNodes.Get(0)->GetId()] = Vector(0, 0, 30.0);

    // UE positions (random within cell)
    Ptr<UniformRandomVariable> xRand = CreateObject<UniformRandomVariable>();
    xRand->SetAttribute("Min", DoubleValue(-cellRadius));
    xRand->SetAttribute("Max", DoubleValue(cellRadius));
    xRand->SetStream(4001);

    Ptr<UniformRandomVariable> yRand = CreateObject<UniformRandomVariable>();
    yRand->SetAttribute("Min", DoubleValue(-cellRadius));
    yRand->SetAttribute("Max", DoubleValue(cellRadius));
    yRand->SetStream(4002);

    for (uint32_t i = 0; i < nUEs; i++)
    {
        Vector pos(xRand->GetValue(), yRand->GetValue(), 1.5);
        allocator->Add(pos);
        uint32_t nodeId = ueNodes.Get(i)->GetId();
        nodePositions[nodeId] = pos;
        nodeTxPowers[nodeId] = 23.0; // Default max power
    }

    mobility.SetPositionAllocator(allocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    NodeContainer allNodes = NodeContainer(enbNodes, ueNodes);
    mobility.Install(allNodes);

    // ========== Install Devices ==========
    globalEnb = nbiotHelper.InstallEnbDevice(globalChannel, enbNodes.Get(0));
    ueDevices = nbiotHelper.InstallUeDevices(globalChannel, ueNodes);

    // Register all UEs at eNB
    for (uint32_t i = 0; i < nUEs; i++)
    {
        uint32_t nodeId = ueNodes.Get(i)->GetId();
        globalEnb->RegisterUE(nodeId);
    }

    // ========== Environment Setup ==========
    if (enableEnvironmentalModeling)
    {
        SetupInterCellInterference(cellRadius, nNeighborCells);
        SetupLTEDataInterferers(cellRadius, nLteInterferers);
        InitializeHardwareVariability(nUEs, ueNodes);
    }

    // ========== Initialize UE Parameters ==========
    for (uint32_t i = 0; i < nUEs; i++)
    {
        uint32_t nodeId = ueNodes.Get(i)->GetId();
        Vector pos = nodePositions[nodeId];
        double distance = std::sqrt(pos.x * pos.x + pos.y * pos.y);

        uint8_t initMCS = getInitialMCSForDistance(distance);
        int initReps = getInitialRepetitionsForDistance(distance);
        double initTP = getInitialTPForDistance(distance);

        currentMCS[nodeId] = initMCS;
        currentRepetitions[nodeId] = initReps;
        currentTP[nodeId] = initTP;
        currentRU[nodeId] = 1; // Single-tone 15kHz default
        nodeTxPowers[nodeId] = initTP;

        // Initialize history
        deviceHistories[nodeId] = DeviceHistory();
        deviceHistories[nodeId].currentRU = 1;
        deviceHistories[nodeId].currentRepetitions = initReps;
        deviceHistories[nodeId].batteryLevel = 1.0;

        // Initialize metrics
        ueMetrics[nodeId] = UEMetrics();

        // Apply to device object
        ueDevices[i]->SetMCS(initMCS);
        ueDevices[i]->SetRepetitions(initReps);
        ueDevices[i]->SetTxPower(initTP);
    }

    // ========== Initialize RL Agents ==========
    if (adrMethod == ADRMethod::DDQN || adrMethod == ADRMethod::PPO || adrMethod == ADRMethod::MARL)
    {
        std::string agentType =
            (adrMethod == ADRMethod::DDQN ? "DDQN-PER"
                                          : (adrMethod == ADRMethod::PPO ? "PPO" : "MARL"));
        std::cout << "🎯 Initializing " << agentType << " agents for NB-IoT..." << std::endl;

        for (uint32_t i = 0; i < nUEs; i++)
        {
            uint32_t nodeId = ueNodes.Get(i)->GetId();
            Vector pos = nodePositions[nodeId];
            double distance = std::sqrt(pos.x * pos.x + pos.y * pos.y);

            if (adrMethod == ADRMethod::DDQN)
            {
                optimizedAgents[nodeId] =
                    std::make_unique<OptimizedDDQNAgent>(0.0005, 0.3, 0.999, 0.05, 0.95, 50);
                ddqnAgents[nodeId] =
                    std::make_unique<DDQNPERADRAgent>(0.01, 0.5, 0.998, 0.1, 0.9, 50, 1.0, 0.2);
            }
            else if (adrMethod == ADRMethod::PPO)
            {
                ppoAgents[nodeId] = std::make_unique<PPOAgent>();
            }
            else if (adrMethod == ADRMethod::MARL)
            {
                marlAgents[nodeId] =
                    std::make_unique<MARLAgent>(0.0005, 0.3, 0.999, 0.05, 0.95, 50);
            }

            std::cout << "  UE " << nodeId << ": dist=" << std::fixed << std::setprecision(0)
                      << distance << "m, MCS=" << (int)currentMCS[nodeId]
                      << ", Reps=" << currentRepetitions[nodeId] << ", TP=" << currentTP[nodeId]
                      << "dBm" << std::endl;
        }

        if (adrMethod == ADRMethod::DDQN)
        {
            std::cout << "  ✅ Using Dueling DQN with extended action space (60 actions)"
                      << std::endl;
            std::cout << "  ✅ Actions: MCS delta × Repetition change × RU change" << std::endl;
            std::cout << "  ✅ RACH collision detection heuristics enabled" << std::endl;
        }
        else if (adrMethod == ADRMethod::PPO)
        {
            std::cout << "  ✅ Using PPO with continuous actions (MCS, Reps, TP)" << std::endl;
            std::cout << "  ✅ Actor-Critic architecture with GAE" << std::endl;
        }
        else if (adrMethod == ADRMethod::MARL)
        {
            std::cout << "  ✅ Using MARL with eNB coordination signals" << std::endl;
            std::cout << "  ✅ Cell-wide metrics shared via eNB" << std::endl;
        }
        std::cout << std::endl;
    }

    // ========== Schedule Transmissions ==========
    std::cout << "📡 Scheduling uplink transmissions..." << std::endl;

    for (uint32_t i = 0; i < nUEs; i++)
    {
        // Stagger start times
        Ptr<UniformRandomVariable> startVar = CreateObject<UniformRandomVariable>();
        startVar->SetAttribute("Min", DoubleValue(1.0));
        startVar->SetAttribute("Max", DoubleValue(appPeriodSeconds));
        startVar->SetStream(5000 + i);
        double startTime = startVar->GetValue();

        Simulator::Schedule(Seconds(startTime), &SimulateUplinkTransmission, i);

        std::cout << "  UE " << ueNodes.Get(i)->GetId() << ": first TX at " << std::fixed
                  << std::setprecision(1) << startTime << "s, period=" << appPeriodSeconds << "s"
                  << std::endl;
    }
    std::cout << "✅ All UEs scheduled\n" << std::endl;

    // Schedule CSV output
    Simulator::Schedule(Seconds(simulationTime), [csvFileName]() { WriteCSVOutput(csvFileName); });

    // Write environment visualization
    std::string baseName = csvFileName.substr(0, csvFileName.find_last_of('.'));
    WriteEnvironmentVisualization(baseName, cellRadius, nUEs);

    // ========== Run ==========
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();

    // ========== Statistics ==========
    uint32_t totalSent = 0, totalReceived = 0;
    double totalEnergy = 0;
    for (const auto& record : completedPackets)
    {
        totalSent++;
        if (record.dataSuccess && record.rachSuccess)
        {
            totalReceived++;
        }
        totalEnergy += record.energyConsumed_mJ;
    }

    double pdr = totalSent > 0 ? (100.0 * totalReceived / totalSent) : 0.0;
    double avgEnergy = totalSent > 0 ? (totalEnergy / totalSent) : 0.0;

    std::cout << "\n=== NB-IoT SIMULATION STATISTICS ===" << std::endl;
    std::cout << "Duration: " << simulationTime << "s" << std::endl;
    std::cout << "Total Packets Sent: " << totalSent << std::endl;
    std::cout << "Total Packets Received: " << totalReceived << std::endl;
    std::cout << "PDR: " << std::fixed << std::setprecision(2) << pdr << "%" << std::endl;
    std::cout << "Avg Energy per Packet: " << std::setprecision(2) << avgEnergy << " mJ"
              << std::endl;
    std::cout << "Output: " << csvFileName << std::endl;
    std::cout << "====================================\n" << std::endl;

    Simulator::Destroy();
}
