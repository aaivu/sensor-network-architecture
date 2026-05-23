// -*- mode: c++; -*-
#pragma once
// MARL-PPO Agent: MARLPPOActor, MARLPPOCritic, MARLPPOAgent, marlPpoAgents

class MARLPPOActor {
public:
    static const int INPUT_DIM  = 12;
    static const int HIDDEN_DIM = 64;
    static const int NUM_ACTIONS = 48;

private:
    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double>              b1, b2, b3;

    double leakyRelu(double x) const { return x >= 0.0 ? x : 0.01 * x; }

public:
    MARLPPOActor() {
        std::mt19937 gen(std::random_device{}());
        auto heInit = [&](int fan) {
            return std::normal_distribution<double>(0.0, std::sqrt(2.0 / fan));
        };
        w1.assign(INPUT_DIM,  std::vector<double>(HIDDEN_DIM));
        w2.assign(HIDDEN_DIM, std::vector<double>(HIDDEN_DIM));
        w3.assign(HIDDEN_DIM, std::vector<double>(NUM_ACTIONS));
        b1.assign(HIDDEN_DIM, 0.0);
        b2.assign(HIDDEN_DIM, 0.0);
        b3.assign(NUM_ACTIONS, 0.0);
        auto d1 = heInit(INPUT_DIM);
        auto d2 = heInit(HIDDEN_DIM);
        std::normal_distribution<double> d3(0.0, 0.01);
        for (auto& row : w1) for (auto& w : row) w = d1(gen);
        for (auto& row : w2) for (auto& w : row) w = d2(gen);
        for (auto& row : w3) for (auto& w : row) w = d3(gen);
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        int n = std::min((int)input.size(), INPUT_DIM);
        std::vector<double> h1(HIDDEN_DIM, 0.0), h2(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h1[j] = b1[j];
            for (int i = 0; i < n; i++) h1[j] += input[i] * w1[i][j];
            h1[j] = leakyRelu(h1[j]);
        }
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h2[j] = b2[j];
            for (int i = 0; i < HIDDEN_DIM; i++) h2[j] += h1[i] * w2[i][j];
            h2[j] = leakyRelu(h2[j]);
        }
        std::vector<double> logits(NUM_ACTIONS, 0.0);
        for (int j = 0; j < NUM_ACTIONS; j++) {
            logits[j] = b3[j];
            for (int i = 0; i < HIDDEN_DIM; i++) logits[j] += h2[i] * w3[i][j];
        }
        return logits;
    }

    std::vector<double> softmax(const std::vector<double>& logits) const {
        double maxL = *std::max_element(logits.begin(), logits.end());
        std::vector<double> probs(NUM_ACTIONS);
        double sum = 0.0;
        for (int i = 0; i < NUM_ACTIONS; i++) { probs[i] = std::exp(logits[i] - maxL); sum += probs[i]; }
        for (auto& p : probs) p /= sum;
        return probs;
    }

    /** Simplified PPO clipped gradient update for the taken action. */
    void update(const std::vector<double>& input, int action, double advantage,
                double oldLogProb, double lr, double clipEps) {
        int n = std::min((int)input.size(), INPUT_DIM);
        std::vector<double> h1(HIDDEN_DIM, 0.0), h2(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h1[j] = b1[j];
            for (int i = 0; i < n; i++) h1[j] += input[i] * w1[i][j];
            h1[j] = leakyRelu(h1[j]);
        }
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h2[j] = b2[j];
            for (int i = 0; i < HIDDEN_DIM; i++) h2[j] += h1[i] * w2[i][j];
            h2[j] = leakyRelu(h2[j]);
        }
        std::vector<double> logits(NUM_ACTIONS, 0.0);
        for (int j = 0; j < NUM_ACTIONS; j++) {
            logits[j] = b3[j];
            for (int i = 0; i < HIDDEN_DIM; i++) logits[j] += h2[i] * w3[i][j];
        }
        auto probs = softmax(logits);
        double newLogProb = std::log(probs[action] + 1e-8);
        double ratio = std::exp(newLogProb - oldLogProb);
        double clipped = std::max(1.0 - clipEps, std::min(1.0 + clipEps, ratio));
        double ppoGrad = -std::min(ratio, clipped) * advantage;
        std::vector<double> dLogits(NUM_ACTIONS, 0.0);
        for (int j = 0; j < NUM_ACTIONS; j++)
            dLogits[j] = ppoGrad * ((j == action ? 1.0 : 0.0) - probs[j]);
        for (int j = 0; j < NUM_ACTIONS; j++) {
            b3[j] -= lr * dLogits[j];
            for (int i = 0; i < HIDDEN_DIM; i++) w3[i][j] -= lr * dLogits[j] * h2[i];
        }
    }
};

/**
 * Local value network for MARL-PPO critic.
 * Input : 12-dim coordinated state.  Output: scalar V(s).
 */
class MARLPPOCritic {
public:
    static const int INPUT_DIM  = 12;
    static const int HIDDEN_DIM = 64;
private:
    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double>              b1, b2, b3;
    double tanh_(double x) const { return std::tanh(x); }
public:
    MARLPPOCritic() {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dis(0.0, 0.1);
        w1.assign(INPUT_DIM,  std::vector<double>(HIDDEN_DIM));
        w2.assign(HIDDEN_DIM, std::vector<double>(HIDDEN_DIM));
        w3.assign(HIDDEN_DIM, std::vector<double>(1));
        b1.assign(HIDDEN_DIM, 0.0);
        b2.assign(HIDDEN_DIM, 0.0);
        b3.assign(1, 0.0);
        for (auto& row : w1) for (auto& w : row) w = dis(gen);
        for (auto& row : w2) for (auto& w : row) w = dis(gen);
        for (auto& row : w3) for (auto& w : row) w = dis(gen);
    }
    double forward(const std::vector<double>& input) const {
        int n = std::min((int)input.size(), INPUT_DIM);
        std::vector<double> h1(HIDDEN_DIM, 0.0), h2(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h1[j] = b1[j];
            for (int i = 0; i < n; i++) h1[j] += input[i] * w1[i][j];
            h1[j] = tanh_(h1[j]);
        }
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h2[j] = b2[j];
            for (int i = 0; i < HIDDEN_DIM; i++) h2[j] += h1[i] * w2[i][j];
            h2[j] = tanh_(h2[j]);
        }
        double val = b3[0];
        for (int i = 0; i < HIDDEN_DIM; i++) val += h2[i] * w3[i][0];
        return val;
    }
    void update(const std::vector<double>& input, double target, double lr) {
        int n = std::min((int)input.size(), INPUT_DIM);
        std::vector<double> h1(HIDDEN_DIM, 0.0), h2(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h1[j] = b1[j];
            for (int i = 0; i < n; i++) h1[j] += input[i] * w1[i][j];
            h1[j] = tanh_(h1[j]);
        }
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h2[j] = b2[j];
            for (int i = 0; i < HIDDEN_DIM; i++) h2[j] += h1[i] * w2[i][j];
            h2[j] = tanh_(h2[j]);
        }
        double val = b3[0];
        for (int i = 0; i < HIDDEN_DIM; i++) val += h2[i] * w3[i][0];
        double err = target - val;
        b3[0] += lr * err;
        for (int i = 0; i < HIDDEN_DIM; i++) w3[i][0] += lr * err * h2[i];
        std::vector<double> dh2(HIDDEN_DIM, 0.0);
        for (int i = 0; i < HIDDEN_DIM; i++)
            dh2[i] = err * w3[i][0] * (1.0 - h2[i] * h2[i]);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            b2[j] += lr * dh2[j];
            for (int i = 0; i < HIDDEN_DIM; i++) w2[i][j] += lr * dh2[j] * h1[i];
        }
    }
};

/**
 * Per-device MARL-PPO agent.
 * Uses the same 12-dim coordinated state as MARLAgent and the same
 * coordination-aware channel-hop pre-selection before policy sampling.
 */
class MARLPPOAgent {
private:
    MARLPPOActor  actor;
    MARLPPOCritic critic;

    double epsilon, epsilonDecay, epsilonMin;
    double gamma, gaeLambda, clipEpsilon;
    double actorLR, criticLR;

    std::mt19937 rng;

    struct Transition {
        std::vector<double> state;
        int    action;
        double logProb;
        double reward;
        double value;
        bool   done;
    };
    std::vector<Transition> buffer;

public:
    MARLPPOAgent(double lr = 0.0003, double eps = 0.3, double epsDecay = 0.999,
                 double epsMin = 0.05, double g = 0.95,
                 double lam = 0.95,   double clip = 0.2)
        : epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin),
          gamma(g), gaeLambda(lam), clipEpsilon(clip),
          actorLR(lr), criticLR(lr * 3.0),
          rng(std::random_device{}())
    {}

    /** Build the 12-dim coordinated state (mirrors MARLAgent::buildCoordinatedState). */
    std::vector<double> buildCoordinatedState(
        const DeviceHistory& history,
        uint8_t currentSFVal, double currentTPVal, uint8_t currentChannel,
        double currentTime, uint32_t nodeId,
        const std::map<uint32_t, DeviceHistory>& allHistories,
        const std::map<uint32_t, uint8_t>& allSFs,
        const std::map<uint32_t, uint8_t>& allChannels)
    {
        double timeSinceLastSuccess = currentTime - history.lastSuccessTime;
        int sameChannelCount = 0;
        for (const auto& kv : allChannels)
            if (kv.first != nodeId && kv.second == currentChannel) sameChannelCount++;
        return {
            std::max(0.0, std::min(1.0, (history.getLastRssi()  + 120.0) / 40.0)),
            std::max(0.0, std::min(1.0,  history.getRssiVariance() / 20.0)),
            std::max(0.0, std::min(1.0, (history.getLastSnr()   +  20.0) / 50.0)),
            std::max(0.0, std::min(1.0, (history.getSnrTrend()  +   5.0) / 10.0)),
            history.getRecentPdr(),
            std::max(0.0, std::min(1.0, (history.getPdrTrend()  +   0.5) /  1.0)),
            std::max(0.0, std::min(1.0,  history.consecutiveLosses / 5.0)),
            std::max(0.0, std::min(1.0,  timeSinceLastSuccess / 300.0)),
            (currentSFVal  - 7.0) / 5.0,
            (currentTPVal  - 2.0) / 12.0,
            currentChannel / 7.0,
            std::min(1.0, sameChannelCount / g_coordDivisor)  // coordination signal
        };
    }

    /**
     * Select action: coordination-aware channel-hop pre-check, then sample
     * from the softmax policy (same logic as MARLAgent::selectAction).
     */
    int selectAction(const std::vector<double>& state,
                     uint8_t currentChannel,
                     const std::map<uint32_t, uint8_t>& allChannels,
                     uint32_t nodeId,
                     double& logProbOut)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Coordination-aware channel-hop pre-selection
        std::vector<int> channelCounts(8, 0);
        for (const auto& kv : allChannels)
            if (kv.first != nodeId) channelCounts[kv.second]++;

        if (channelCounts[currentChannel] >= 2 && dist(rng) < 0.4) {
            int leastCongested = 0;
            for (int i = 1; i < 8; i++)
                if (channelCounts[i] < channelCounts[leastCongested]) leastCongested = i;
            int action = 12 + leastCongested;
            logProbOut = std::log(1.0 / 48.0);  // Uniform approximation for forced hop
            return action;
        }

        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> uniDist(0, 47);
            int a = uniDist(rng);
            logProbOut = std::log(1.0 / 48.0);
            return a;
        }

        std::vector<double> logits = actor.forward(state);
        std::vector<double> probs  = actor.softmax(logits);
        std::discrete_distribution<int> catDist(probs.begin(), probs.end());
        int a = catDist(rng);
        logProbOut = std::log(probs[a] + 1e-8);
        return a;
    }

    void storeTransition(const std::vector<double>& state, int action,
                         double logProb, double reward, bool done)
    {
        Transition t;
        t.state   = state;
        t.action  = action;
        t.logProb = logProb;
        t.reward  = reward;
        t.value   = critic.forward(state);
        t.done    = done;
        buffer.push_back(t);
    }

    void update() {
        if (buffer.size() < 16) return;
        int T = (int)buffer.size();

        std::vector<double> adv(T, 0.0), ret(T, 0.0);
        double lastGae = 0.0;
        for (int t = T - 1; t >= 0; t--) {
            double nextV = (t == T - 1 || buffer[t].done) ? 0.0 : buffer[t + 1].value;
            double delta = buffer[t].reward + gamma * nextV - buffer[t].value;
            lastGae = delta + gamma * gaeLambda * (buffer[t].done ? 0.0 : lastGae);
            adv[t] = lastGae;
            ret[t] = adv[t] + buffer[t].value;
        }
        double mean = 0.0, std2 = 0.0;
        for (double a : adv) mean += a;
        mean /= T;
        for (double a : adv) std2 += (a - mean) * (a - mean);
        double std_ = std::sqrt(std2 / T + 1e-8);
        for (double& a : adv) a = (a - mean) / std_;

        for (int t = 0; t < T; t++) {
            actor.update(buffer[t].state, buffer[t].action,
                         adv[t], buffer[t].logProb, actorLR, clipEpsilon);
            critic.update(buffer[t].state, ret[t], criticLR);
        }
        buffer.clear();
        if (epsilon > epsilonMin) epsilon *= epsilonDecay;
    }

    double getEpsilon() const { return epsilon; }
    void   increaseExploration() { epsilon = std::min(0.8, epsilon + 0.3); }
};

// Global MARL-PPO agents map
std::map<uint32_t, std::unique_ptr<MARLPPOAgent>> marlPpoAgents;

// ============================================================================
// MAPPO-GAT (Multi-Agent PPO with Graph Attention Network)
// Centralized Training, Decentralized Execution (CTDE)
// Each device runs a local actor enriched with GAT neighbor aggregation.
// A centralized critic sees the global state summary for stable advantage
// estimation during training; at execution time only the actor is needed.
// ============================================================================

/**
 * Graph Attention Network layer (single head, single layer)
 * Each device attends over its neighbours' feature vectors and produces
 * an aggregated representation that captures inter-device interference and
 * channel competition.
 */
