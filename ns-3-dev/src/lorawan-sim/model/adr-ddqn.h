#ifndef ADR_DDQN_H
#define ADR_DDQN_H

#include <vector>
#include <memory>
#include <random>
#include <map>

namespace ns3 {
namespace lorawan {

/**
 * Experience structure for Prioritized Experience Replay
 */
struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> nextState;
    bool done;
    double priority;
    double tdError;
    
    Experience(const std::vector<double>& s, int a, double r, 
              const std::vector<double>& ns, bool d, double p = 1.0);
};

/**
 * Simple Neural Network implementation for Q-value approximation
 */
class SimpleQNetwork {
private:
    static const int INPUT_SIZE = 6;
    static const int HIDDEN1_SIZE = 128;
    static const int HIDDEN2_SIZE = 64;
    static const int OUTPUT_SIZE = 15;
    
    std::vector<std::vector<double>> weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    std::vector<double> lastInput, lastHidden1, lastHidden2, lastOutput;
    double learningRate;
    
    double relu(double x);
    double reluDerivative(double x);
    
public:
    SimpleQNetwork(double lr = 0.001);
    
    void initializeWeights();
    std::vector<double> forward(const std::vector<double>& input);
    void updateWeights(const std::vector<double>& targetOutput);
    void copyFrom(const SimpleQNetwork& other);
};

/**
 * Prioritized Experience Replay Buffer
 */
class PrioritizedReplayBuffer {
private:
    std::vector<Experience> buffer;
    std::vector<double> priorities;
    size_t maxSize, currentSize, position;
    double alpha, beta, epsilon;
    
public:
    PrioritizedReplayBuffer(size_t maxSize, double alpha = 0.6, double beta = 0.4, double epsilon = 1e-6);
    
    void add(const Experience& experience);
    std::vector<Experience> sample(size_t batchSize, std::vector<size_t>& indices, std::vector<double>& weights);
    void updatePriorities(const std::vector<size_t>& indices, const std::vector<double>& tdErrors);
    size_t size() const;
};

/**
 * DDQN-PER Agent for ADR
 */
class DDQNPERADRAgent {
private:
    SimpleQNetwork mainNetwork, targetNetwork;
    PrioritizedReplayBuffer replayBuffer;
    double epsilon, epsilonDecay, epsilonMin, gamma;
    int targetUpdateFreq, updateCounter;
    std::vector<int> sfActions;
    std::vector<int> tpActions;
    double pdrWeight, energyWeight;
    std::mt19937 rng;
    
    double getRequiredSNR(int sf);
    int getOptimalSFForDistance(double distance);
    
public:
    DDQNPERADRAgent(double lr = 0.001, double eps = 1.0, double epsDecay = 0.995, 
                   double epsMin = 0.01, double gam = 0.95, int targetFreq = 100,
                   double pdrW = 1.0, double energyW = 0.5);
    
    void increaseExploration();
    int selectAction(const std::vector<double>& state);
    void addExperience(const std::vector<double>& state, int action, double reward,
                      const std::vector<double>& nextState, bool done);
    void train(size_t batchSize = 32);
    std::pair<int, int> getParameters(int action);
    double calculateReward(double pdr, double energyConsumption, double prevPdr, double prevEnergy);
    double calculateAdvancedReward(uint32_t nodeId, double pdr, double energyConsumption, 
                              double snr, bool packetSuccess, int currentSF, double currentTP,
                              double distance, uint32_t packetsSent);
    double getEpsilon() const;
};

} // namespace lorawan
} // namespace ns3

#endif // ADR_DDQN_H
