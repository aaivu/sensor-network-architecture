// -*- mode: c++; -*-
#pragma once
// MAPPO-GAT Agent: GATLayer, MAPPOGATActor, MAPPOCentralizedCritic, MAPPOGATAgent, mappoGatAgents

class GATLayer {
public:
    static const int FEAT_DIM   = 12;  // Matches local state dimension
    static const int HIDDEN_DIM = 16;  // Output dimension after linear transform

private:
    // W : FEAT_DIM -> HIDDEN_DIM   (linear projection shared across all nodes)
    std::vector<std::vector<double>> W;
    std::vector<double>              Wbias;
    // Attention scoring vector a : 2*HIDDEN_DIM -> scalar (concatenation form)
    std::vector<double>              attnVec;

    double leakyRelu(double x) const { return x >= 0.0 ? x : 0.2 * x; }

public:
    GATLayer() {
        std::mt19937 gen(1337);
        double scale = std::sqrt(2.0 / FEAT_DIM);
        std::normal_distribution<double> wDis(0.0, scale);
        std::normal_distribution<double> aDis(0.0, 0.02);

        W.assign(FEAT_DIM, std::vector<double>(HIDDEN_DIM));
        Wbias.assign(HIDDEN_DIM, 0.0);
        attnVec.assign(2 * HIDDEN_DIM, 0.0);

        for (auto& row : W)    for (auto& w : row) w = wDis(gen);
        for (auto& a : attnVec)                     a = aDis(gen);
    }

    // Linear projection + LeakyReLU
    std::vector<double> transform(const std::vector<double>& feat) const {
        int n = std::min((int)feat.size(), FEAT_DIM);
        std::vector<double> out(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            out[j] = Wbias[j];
            for (int i = 0; i < n; i++) out[j] += feat[i] * W[i][j];
            out[j] = leakyRelu(out[j]);
        }
        return out;
    }

    // e_ij = LeakyReLU( a^T [ W*h_i || W*h_j ] )
    double attentionScore(const std::vector<double>& hi,
                          const std::vector<double>& hj) const {
        double e = 0.0;
        for (int k = 0; k < HIDDEN_DIM; k++) {
            e += attnVec[k]              * hi[k];
            e += attnVec[HIDDEN_DIM + k] * hj[k];
        }
        return leakyRelu(e);
    }

    /**
     * Aggregate self + neighbours via attention.
     * Returns: aggregated feature vector of size HIDDEN_DIM.
     */
    std::vector<double> forward(
        const std::vector<double>&               selfFeat,
        const std::vector<std::vector<double>>& neighborFeats) const
    {
        std::vector<double> selfH = transform(selfFeat);
        if (neighborFeats.empty()) return selfH;

        // Transform all nodes (self at index 0)
        std::vector<std::vector<double>> allH;
        allH.push_back(selfH);
        for (const auto& nf : neighborFeats) allH.push_back(transform(nf));

        // Compute raw attention scores from self to each node
        std::vector<double> eScores(allH.size());
        for (size_t j = 0; j < allH.size(); j++)
            eScores[j] = attentionScore(selfH, allH[j]);

        // Softmax
        double maxE  = *std::max_element(eScores.begin(), eScores.end());
        double sumEx = 0.0;
        for (double& s : eScores) { s = std::exp(s - maxE); sumEx += s; }
        for (double& s : eScores)   s /= (sumEx + 1e-9);

        // Weighted aggregation
        std::vector<double> agg(HIDDEN_DIM, 0.0);
        for (size_t j = 0; j < allH.size(); j++)
            for (int k = 0; k < HIDDEN_DIM; k++)
                agg[k] += eScores[j] * allH[j][k];

        return agg;
    }
};

// One shared GAT encoder for all MAPPO-GAT agents
static GATLayer sharedGATLayer;

/**
 * MAPPO-GAT Actor network (per device).
 * Input  : local_state (12) concatenated with GAT-aggregated features (16) = 28 dims
 * Output : logits over 48 discrete actions (same action space as MARL / OptimizedDDQN)
 */
class MAPPOGATActor {
public:
    static const int INPUT_DIM  = GATLayer::FEAT_DIM + GATLayer::HIDDEN_DIM; // 28
    static const int HIDDEN_DIM = 64;
    static const int NUM_ACTIONS = 48;

private:
    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double>              b1, b2, b3;

    double leakyRelu(double x) const { return x >= 0.0 ? x : 0.01 * x; }

public:
    MAPPOGATActor() {
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
        std::normal_distribution<double> d3(0.0, 0.01); // small init for output

        for (auto& row : w1) for (auto& w : row) w = d1(gen);
        for (auto& row : w2) for (auto& w : row) w = d2(gen);
        for (auto& row : w3) for (auto& w : row) w = d3(gen);
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        int n = std::min((int)input.size(), INPUT_DIM);
        std::vector<double> h1(HIDDEN_DIM, 0.0);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            h1[j] = b1[j];
            for (int i = 0; i < n; i++) h1[j] += input[i] * w1[i][j];
            h1[j] = leakyRelu(h1[j]);
        }
        std::vector<double> h2(HIDDEN_DIM, 0.0);
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
        for (int i = 0; i < NUM_ACTIONS; i++) {
            probs[i] = std::exp(logits[i] - maxL); sum += probs[i];
        }
        for (auto& p : probs) p /= sum;
        return probs;
    }

    /**
     * Simplified PPO actor update: estimate gradient via policy ratio + clipping.
     * oldLogProb : log π_old(a|s)  (scalar, for the single taken action)
     */
    void update(const std::vector<double>& input, int action, double advantage,
                double oldLogProb, double lr, double clipEps) {
        int n = std::min((int)input.size(), INPUT_DIM);

        // Forward pass (store activations)
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
        std::vector<double> probs = softmax(logits);

        double newLogProb = std::log(probs[action] + 1e-8);
        double ratio      = std::exp(newLogProb - oldLogProb);
        double clipped    = std::max(1.0 - clipEps, std::min(1.0 + clipEps, ratio));
        double ppoGrad    = -std::min(ratio, clipped) * advantage; // negative sign: gradient descent

        // Gradient on output layer (policy gradient for the chosen action)
        std::vector<double> dLogits(NUM_ACTIONS, 0.0);
        for (int j = 0; j < NUM_ACTIONS; j++) {
            // grad log π(a|s) w.r.t. logit_j = (1{j==a} - π(j))
            dLogits[j] = ppoGrad * ((j == action ? 1.0 : 0.0) - probs[j]);
        }
        for (int j = 0; j < NUM_ACTIONS; j++) {
            b3[j] -= lr * dLogits[j];
            for (int i = 0; i < HIDDEN_DIM; i++)
                w3[i][j] -= lr * dLogits[j] * h2[i];
        }
    }
};

/**
 * MAPPO Centralized Critic (shared across all agents of the same run).
 * Input  : local_state (12) + global mean state (12) = 24 dims
 * Output : scalar value V(s_global)
 */
class MAPPOCentralizedCritic {
public:
    static const int LOCAL_DIM   = 12;
    static const int GLOBAL_DIM  = 12;
    static const int INPUT_DIM   = LOCAL_DIM + GLOBAL_DIM; // 24
    static const int HIDDEN_DIM  = 64;

private:
    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double>              b1, b2, b3;

    double tanh_(double x) const { return std::tanh(x); }

public:
    MAPPOCentralizedCritic() {
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

// One shared centralized critic (all MAPPO-GAT agents reference this for CTDE)
static MAPPOCentralizedCritic sharedMAPPOCritic;

/**
 * Per-device MAPPO-GAT agent.
 * Maintains its own actor; uses sharedGATLayer + sharedMAPPOCritic.
 */
class MAPPOGATAgent {
private:
    MAPPOGATActor actor;

    double epsilon, epsilonDecay, epsilonMin;
    double gamma, gaeLambda, clipEpsilon;
    double actorLR, criticLR;

    std::mt19937 rng;

    struct Transition {
        std::vector<double> gatInput;    // actor input: local + GAT (28 dims)
        std::vector<double> criticInput; // critic input: local + global mean (24 dims)
        int    action;
        double logProb;
        double reward;
        double value;
        bool   done;
    };
    std::vector<Transition> buffer;

public:
    MAPPOGATAgent(double lr = 0.0003, double eps = 0.3, double epsDecay = 0.999,
                  double epsMin = 0.05, double g = 0.95,
                  double lam = 0.95,   double clip = 0.2)
        : epsilon(eps), epsilonDecay(epsDecay), epsilonMin(epsMin),
          gamma(g), gaeLambda(lam), clipEpsilon(clip),
          actorLR(lr), criticLR(lr * 3.0),
          rng(std::random_device{}())
    {}

    /**
     * Build actor input: [localState || GAT(neighbours)]  (12 + 16 = 28)
     */
    std::vector<double> buildGATInput(
        const std::vector<double>&               localState,
        const std::vector<std::vector<double>>& neighborStates) const
    {
        std::vector<double> gatFeat = sharedGATLayer.forward(localState, neighborStates);
        std::vector<double> combined = localState;
        combined.insert(combined.end(), gatFeat.begin(), gatFeat.end());
        return combined;
    }

    /**
     * Build critic input: [localState || globalMeanState]  (12 + 12 = 24)
     */
    std::vector<double> buildCriticInput(
        const std::vector<double>&               localState,
        const std::vector<std::vector<double>>& allStates) const
    {
        constexpr int DIM = MAPPOCentralizedCritic::LOCAL_DIM;
        std::vector<double> globalMean(DIM, 0.0);
        if (!allStates.empty()) {
            for (const auto& s : allStates)
                for (int i = 0; i < DIM && i < (int)s.size(); i++)
                    globalMean[i] += s[i];
            for (auto& v : globalMean) v /= (double)allStates.size();
        }
        std::vector<double> criticIn = localState;
        criticIn.insert(criticIn.end(), globalMean.begin(), globalMean.end());
        return criticIn;
    }

    /**
     * Select action; also returns the log-probability of chosen action.
     */
    int selectAction(const std::vector<double>& gatInput, double& logProbOut) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> uniDist(0, 47);
            int a = uniDist(rng);
            logProbOut = std::log(1.0 / 48.0);
            return a;
        }

        std::vector<double> logits = actor.forward(gatInput);
        std::vector<double> probs  = actor.softmax(logits);

        std::discrete_distribution<int> catDist(probs.begin(), probs.end());
        int a = catDist(rng);
        logProbOut = std::log(probs[a] + 1e-8);
        return a;
    }

    void storeTransition(const std::vector<double>& gatInput,
                         const std::vector<double>& criticInput,
                         int action, double logProb,
                         double reward, bool done)
    {
        Transition t;
        t.gatInput    = gatInput;
        t.criticInput = criticInput;
        t.action      = action;
        t.logProb     = logProb;
        t.reward      = reward;
        t.value       = sharedMAPPOCritic.forward(criticInput);
        t.done        = done;
        buffer.push_back(t);
    }

    void update() {
        if (buffer.size() < 16) return;
        int T = (int)buffer.size();

        // GAE advantage computation
        std::vector<double> adv(T, 0.0), ret(T, 0.0);
        double lastGae = 0.0;
        for (int t = T - 1; t >= 0; t--) {
            double nextV = (t == T - 1 || buffer[t].done) ? 0.0 : buffer[t + 1].value;
            double delta = buffer[t].reward + gamma * nextV - buffer[t].value;
            lastGae = delta + gamma * gaeLambda * (buffer[t].done ? 0.0 : lastGae);
            adv[t] = lastGae;
            ret[t] = adv[t] + buffer[t].value;
        }

        // Normalise advantages
        double mean = 0.0, std2 = 0.0;
        for (double a : adv) mean += a;
        mean /= T;
        for (double a : adv) std2 += (a - mean) * (a - mean);
        double std_ = std::sqrt(std2 / T + 1e-8);
        for (double& a : adv) a = (a - mean) / std_;

        // Actor and critic updates
        for (int t = 0; t < T; t++) {
            actor.update(buffer[t].gatInput, buffer[t].action,
                         adv[t], buffer[t].logProb, actorLR, clipEpsilon);
            sharedMAPPOCritic.update(buffer[t].criticInput, ret[t], criticLR);
        }

        buffer.clear();
        if (epsilon > epsilonMin) epsilon *= epsilonDecay;
    }

    double getEpsilon() const { return epsilon; }
    void   increaseExploration() { epsilon = std::min(0.8, epsilon + 0.3); }
};

// Global MAPPO-GAT agents map
std::map<uint32_t, std::unique_ptr<MAPPOGATAgent>> mappoGatAgents;

// ============================================================================
// ENHANCED ENVIRONMENT & SIMULATION LOGIC
// ============================================================================

