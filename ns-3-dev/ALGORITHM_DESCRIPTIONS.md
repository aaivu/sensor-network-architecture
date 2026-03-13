# NB-IoT Link Adaptation Algorithms — Detailed Description

This document describes all five NB-IoT link adaptation policies implemented in
`src/nbiot-adr/examples/advanced-ddqn-per-adr-nbiot.cc`.

---

## Common NB-IoT Parameter Space

Before describing each algorithm, the shared parameter space they all operate on:

### MCS Table (3GPP TS 36.213 Table 16.5.1.2-1)

| MCS | Modulation | Required SINR (dB) | Spectral Efficiency |
| --- | ---------- | ------------------ | ------------------- |
| 0   | QPSK       | −12.6              | 0.15 bps/Hz         |
| 1   | QPSK       | −10.5              | 0.23                |
| 2   | QPSK       | −8.4               | 0.38                |
| 3   | QPSK       | −6.3               | 0.60                |
| 4   | QPSK       | −4.2               | 0.88                |
| 5   | QPSK       | −2.1               | 1.18                |
| 6   | QPSK       | 0.0                | 1.48                |
| 7   | QPSK       | 2.1                | 1.91                |
| 8   | QPSK       | 4.2                | 2.41                |
| 9   | QPSK       | 6.0                | 2.73                |
| 10  | QPSK       | 8.0                | 3.32                |

### Repetition Values

`{1, 2, 4, 8, 16, 32, 64, 128}` — each doubling adds ≈3 dB gain  
(`GetRepetitionGainDb(N) = 10·log₁₀(N)`)

### Resource Unit (RU) Types

| Index | Type              | Subcarriers | Bandwidth | RU Duration | Processing Gain |
| ----- | ----------------- | ----------- | --------- | ----------- | --------------- |
| 0     | 1-tone 3.75 kHz   | 1           | 3.75 kHz  | 32.0 ms     | 16.8 dB         |
| 1     | 1-tone 15 kHz     | 1           | 15 kHz    | 8.0 ms      | 10.8 dB         |
| 2     | 3-tone            | 3           | 45 kHz    | 4.0 ms      | 6.0 dB          |
| 3     | 6-tone            | 6           | 90 kHz    | 2.0 ms      | 3.0 dB          |
| 4     | 12-tone (full BW) | 12          | 180 kHz   | 1.0 ms      | 0.0 dB          |

### TX Power Range

- Min: 14 dBm (Class 6), Max: 23 dBm (Class 3)
- Typical discrete steps: 14, 20, 23 dBm

### Coverage Enhancement Levels

| CE Level | Max Coupling Loss | Max Repetitions | Max RACH Attempts |
| -------- | ----------------- | --------------- | ----------------- |
| CE0      | 144 dB            | 1               | 5                 |
| CE1      | 154 dB            | 16              | 10                |
| CE2      | 164 dB            | 128             | 15                |

---

## Algorithm 1 — SDBP (Static Distance-Based Profile)

**ADRMethod::OFF** — No link adaptation is performed.

### Description

SDBP is the non-adaptive baseline. Each UE is assigned fixed transmission parameters at
initialization based on its distance from the eNB. Parameters never change during the
simulation, regardless of channel conditions or packet outcomes. It represents a typical
static deployment where operators manually configure devices at install time.

### States

None — SDBP does not observe or react to any channel or network state. The device history
(`DeviceHistory`) is still populated for logging, but is never used for parameter updates.

### Actions

None — no decisions are made. The device uses whatever MCS, repetitions, TX power, and RU
were assigned at start-up.

### Variables

| Variable                   | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `currentMCS[ueId]`         | Fixed MCS; set at initialization, never updated |
| `currentRepetitions[ueId]` | Fixed repetitions; never updated                |
| `currentTP[ueId]`          | Fixed TX power (dBm); never updated             |
| `currentRU[ueId]`          | Fixed Resource Unit index; never updated        |

### Constants

None beyond the NB-IoT parameter tables above.

### How It Works

1. At simulation start, each UE is placed at some distance from the eNB.
2. Parameters are assigned (e.g., closer UEs get higher MCS/lower reps; far UEs get
   more repetitions and lower MCS).
3. For every uplink cycle: transmit with fixed parameters → record success/failure →
   schedule next transmission → repeat (no feedback loop).

**Advantage:** Zero overhead, maximum simplicity.  
**Disadvantage:** Cannot adapt to fading, interference, or changing network load.

---

## Algorithm 2 — 3GPP (Standard NB-IoT Link Adaptation)

**ADRMethod::ON** — Classical 3GPP-defined adaptation.

### Description

This implements the standard 3GPP NB-IoT link adaptation procedure from
TS 36.213. It is a rule-based, reactive algorithm that adjusts MCS, repetitions,
and TX power based on measured SINR and recent packet delivery history. It uses
**averaged SINR** over the last 20 transmissions to smooth out transient fading.

The algorithm is equivalent in spirit to LoRaWAN's ADR (Adaptive Data Rate): observe
the channel, pick the highest MCS that is reliably supportable, and add a fixed
safety margin.

### States

| Observable                       | Source                     |
| -------------------------------- | -------------------------- |
| Current SINR                     | eNB uplink processing      |
| SINR history (last 20)           | `sinrHistMap[ueId]`        |
| Current repetitions              | `currentRepetitions[ueId]` |
| Recent PDR (from packet history) | `deviceHistories[ueId]`    |
| Packet success/failure           | RACH + data TX outcome     |

### Actions

The algorithm directly sets three parameters per update step:

| Action Output   | Type       | Range         |
| --------------- | ---------- | ------------- |
| New MCS         | integer    | 0–10          |
| New repetitions | power-of-2 | 1–128         |
| New TX power    | real (dBm) | 14.0–23.0 dBm |

### Constants

| Constant                       | Value    | Purpose                                                    |
| ------------------------------ | -------- | ---------------------------------------------------------- |
| `kMcsSelectionMarginDb`        | 5.0 dB   | Safety margin above required SINR when selecting MCS       |
| `kTargetSinrMarginDb`          | 8.0 dB   | Target excess SINR above MCS requirement for power control |
| `kRepetitionReductionMarginDb` | 12.0 dB  | Excess effective SINR threshold to reduce repetitions      |
| `kMinTxPowerDbm`               | 14.0 dBm | Minimum allowed TX power (Class 6)                         |
| History window                 | 20       | Length of SINR and packet result history                   |

### Variables

| Variable        | Description                                       |
| --------------- | ------------------------------------------------- |
| `avgSINR`       | Running mean of last-20 SINR window               |
| `effectiveSINR` | `avgSINR + GetRepetitionGainDb(currentReps)`      |
| `newMCS`        | Selected MCS index for next transmission          |
| `reps`          | Updated repetitions count                         |
| `newTP`         | Updated TX power (dBm), 0.5× proportional step    |
| `sinrDiff`      | `targetSINR − avgSINR` (for power control)        |
| `recentPdr`     | Fraction of last-N packets delivered successfully |

### How It Works

```
Every uplink feedback event:
  1. Append sinr to history (max 20 entries)
  2. avgSINR = mean(sinrHistory)
  3. effectiveSINR = avgSINR + 10·log10(currentReps)
  4. MCS selection:
       for mcs = 10 downto 0:
         if effectiveSINR >= MCS_TABLE[mcs].requiredSINR + 5.0 dB:
           newMCS = mcs; break
  5. Repetition adjustment:
       if !packetSuccess AND recentPdr < 0.70:
         reps = min(128, reps × 2)        ← increase repetitions on poor PDR
       elif packetSuccess AND effectiveSINR > requiredSINR + 12.0 dB:
         reps = max(1, reps / 2)          ← reduce when margin is large
  6. Power control (proportional):
       targetSINR = MCS_TABLE[newMCS].requiredSINR + 8.0 dB
       newTP = currentTP + 0.5 × (targetSINR − avgSINR)
       newTP = clamp(newTP, 14.0, 23.0)
  7. Apply: currentMCS, currentRepetitions, currentTP ← new values
```

**Advantage:** No learning required; fast convergence to a sensible operating point.  
**Disadvantage:** Fixed margins do not adapt to cell congestion, RACH pressure, or
heterogeneous channel dynamics.

---

## Algorithm 3 — DDQN (Double DQN with Prioritized Experience Replay)

**ADRMethod::DDQN** — Deep Reinforcement Learning with extended action space.

### Description

DDQN is a value-based deep reinforcement learning algorithm. The core idea is to learn,
through trial-and-error, the long-term expected return (Q-value) of each possible action
in each observed state. It uses two improvements over vanilla DQN:

1. **Double DQN**: Decouples action selection (main network) from value estimation (target
   network), reducing overestimation bias.
2. **Prioritized Experience Replay (PER)**: Samples transitions proportional to their
   temporal-difference (TD) error, so surprising/informative experiences are replayed more
   frequently.
3. **Dueling architecture**: Separates the state-value stream V(s) from the
   advantage stream A(s,a), so Q(s,a) = V(s) + A(s,a) − mean(A). This lets the network
   learn which states are valuable independent of any specific action.
4. **Curiosity-driven exploration**: Applies an inverse-visit-count bonus to Q-values
   during greedy selection, encouraging systematic coverage of the state space.

### Neural Network Architecture (`DuelingQNetwork`)

```
Input (14 features)
    ↓
Shared Layer 1: Linear(14→64) + LeakyReLU(α=0.01)
Shared Layer 2: Linear(64→64) + LeakyReLU
        ↙               ↘
Value Stream            Advantage Stream
Linear(64→32)+LReLU     Linear(64→32)+LReLU
Linear(32→1)            Linear(32→60)
        ↘               ↙
    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
Output: 60 Q-values
```

All weights initialized with Xavier normal initialization. Learning rate: 0.0005.

### Replay Buffer (`PrioritizedReplayBuffer`)

- Capacity: 20,000 transitions
- Priority exponent α = 0.6 (priority = |TD error|^α)
- Importance-sampling exponent β = 0.4 (IS weight correction)
- Min priority ε = 1×10⁻⁶ (prevents zero priority)

### State Vector (14 features, `buildRealisticState`)

| Index | Feature                    | Formula / Source                         | Range   |
| ----- | -------------------------- | ---------------------------------------- | ------- |
| 0     | Normalized RSRP            | (RSRP + 140) / 96                        | [0, 1]  |
| 1     | Normalized SINR            | (SINR + 12) / 42                         | [0, 1]  |
| 2     | RSRP variance              | var(RSRP history) / 20                   | [0, 1]  |
| 3     | SINR trend                 | (recent mean − old mean) / 10            | [−1, 1] |
| 4     | Current MCS (normalized)   | currentMCS / 10                          | [0, 1]  |
| 5     | Current repetitions (log2) | log₂(currentReps) / 7                    | [0, 1]  |
| 6     | Current TX power           | currentTP / 23                           | [0, 1]  |
| 7     | Current RU index           | currentRU / 4                            | [0, 1]  |
| 8     | Recent PDR                 | successes / last-20 packets              | [0, 1]  |
| 9     | PDR trend                  | recent PDR − old PDR                     | [−1, 1] |
| 10    | Cell load                  | `globalEnb->GetCellLoad()`               | [0, 1]  |
| 11    | Battery level              | 1 − (energy consumed / battery capacity) | [0, 1]  |
| 12    | RACH success rate          | successful RACHs / last-20 RACHs         | [0, 1]  |
| 13    | Consecutive failures       | min(1, consecutiveLosses / 10)           | [0, 1]  |

### Extended Action Space (60 actions, `ExtendedActionSpace`)

Each action index encodes three simultaneous NB-IoT parameter changes:

```
action = mcsAction × 12 + repAction × 3 + ruAction

mcsAction (0–4) → MCS delta: {−2, −1, 0, +1, +2}
repAction (0–3) → repetition change: {÷4, ÷2, same, ×2}
ruAction  (0–2) → RU change: {narrower, same, wider}
TX power stays unchanged (managed by explicit power control)
```

Results are clamped: MCS ∈ [0,10], reps ∈ [1,128], RU ∈ [0,4].

### Reward Function (`calculateCongestionAdaptiveReward`)

The reward combines six components whose weights adapt to the estimated cell
congestion level:

```
congestion = estimateCellCongestion(rsrp, rsrpVar, recentPDR, numUEs, avgRACH, cellLoad)
           = 0.4·cellLoad + 0.2·(numUEs/50) + 0.2·RACHcongestion + 0.2·(PDR signal)

Component 1 – Packet outcome (weight: 0.35 + 0.25·(1−congestion)):
  +1.0 on success; −1.0 on failure
  +0.3 bonus if PDR ≥ 0.9; −0.5 if 5+ consecutive losses

Component 2 – RACH efficiency (weight: 0.15 + 0.15·congestion):
  −1.0 on RACH failure; +0.3 if first-attempt success; −0.2× per excess attempt

Component 3 – Energy efficiency (weight: 0.15 + 0.2·(1−congestion)):
  Energy bonus: energyWeight × (1 − energyNorm), where energyNorm = energy/200 mJ
  Repetition bonus: energyWeight × 0.5 × (1 − log2(reps)/7)

Component 4 – MCS efficiency (weight: 0.1 + 0.1·(1−congestion)):
  +0.5 if using optimal MCS ceiling; −0.3 if 2+ below optimal

Component 5 – PDR stability (weight: 0.15):
  +0.5 if improving (trend > 0.1); −0.5 if degrading (trend < −0.1)

Component 6 – Coverage class appropriateness:
  −0.1 if over-provisioned (effSINR > 15 dB and reps > 4)
  −0.1 if under-provisioned (effSINR < 0 dB and reps < 8)
```

### Variables

| Variable           | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `epsilon`          | Current exploration rate; starts 0.30                    |
| `epsilonDecay`     | Per-step multiplicative decay = 0.999                    |
| `epsilonMin`       | Floor on epsilon = 0.05                                  |
| `gamma` (γ)        | Discount factor = 0.95                                   |
| `targetUpdateFreq` | Target network sync period = every 50 steps              |
| `updateCounter`    | Running count of training steps                          |
| `stateVisitCounts` | Count of visits to discretized states (curiosity)        |
| `curiosityBonus`   | Exploration bonus per unvisited state = 0.1              |
| `mcsCeiling`       | Hard cap: highest MCS consistent with SINR + 2 dB margin |

### Constants

| Constant          | Value  | Notes                               |
| ----------------- | ------ | ----------------------------------- |
| INPUT_SIZE        | 14     | State vector dimension              |
| HIDDEN_SIZE       | 64     | Shared/value/advantage layer width  |
| VALUE_SIZE        | 32     | Value-stream hidden layer width     |
| ADVANTAGE_SIZE    | 32     | Advantage-stream hidden layer width |
| OUTPUT_SIZE       | 60     | Action space size                   |
| Buffer capacity   | 20,000 | Max experiences stored              |
| Batch size        | 32     | Mini-batch per training step        |
| α (PER)           | 0.6    | Priority exponent                   |
| β (PER)           | 0.4    | IS-weight exponent                  |
| Learning rate     | 0.0005 | Adam-like gradient step             |
| MCS safety margin | 2.0 dB | Margin for `getMandatoryMCSCeiling` |

### How It Works

```
Per uplink feedback event:
  1. Update DeviceHistory (RSRP, SINR, packet result, RACH result)
  2. Build 14-feature state vector s
  3. Estimate congestion level
  4. Calculate reward r using congestion-adaptive function
  5. Select action a:
       with prob ε → smart heuristic (weak signal / congestion) or random
       else         → argmax_a [ Q(s,a) + curiosityBonus / (1 + visitCount(s)) ]
  6. Decode action → (ΔMCS, Δreps, ΔRU)
  7. Apply mandatory MCS ceiling based on current SINR
  8. Store (s_prev, a_prev, r, s) in PER buffer
  9. Sample batch of 32; compute DDQN target:
       best_a' = argmax_a' Q_main(s', a')
       target  = r + γ · Q_target(s', best_a')
  10. Update main network toward target; scale gradients by IS weights
  11. Update TD-error priorities in buffer
  12. Every 50 steps: copy main → target network
  13. ε ← max(εMin, ε × εDecay)
  14. Apply new MCS, reps, RU to UE
```

**Advantage:** Learns complex nonlinear mappings; handles cell congestion via adaptive rewards.  
**Disadvantage:** Requires warm-up period (exploration phase); hyperparameter sensitive.

---

## Algorithm 4 — PPO (Proximal Policy Optimization)

**ADRMethod::PPO** — Actor-Critic RL with continuous action space and clipped objective.

### Description

PPO is an on-policy actor-critic algorithm that directly parameterizes a **policy
distribution** over continuous actions. Unlike DDQN which learns Q-values over discrete
actions, PPO's actor outputs a Gaussian distribution over three continuous dimensions
(MCS, log₂-repetitions, TX power), from which actions are sampled. The **clipped
surrogate objective** prevents large disruptive policy updates, making training stable
without requiring a trust-region constraint.

The critic (value network) estimates the state value V(s), which is used to compute
**Generalized Advantage Estimates (GAE)** — a variance-reduced estimate of how much
better the taken action was compared to the average.

### Neural Network Architecture

**Actor (`PolicyNetwork`):**

```
Input (14 features)
    ↓
Layer 1: Linear(14→64) + tanh
    ↓
Layer 2: Linear(64→64) + tanh
       ↙                ↘
Mean head (3 outputs)   Log-std head (3 outputs)
Linear(64→3)            Linear(64→3), clamped to [−2.0, 0.5]
  + biasMean init:         + biasLogStd init = −1.0
    [5.0, 2.0, 18.0]
         ↓                       ↓
    μ (MCS, log2reps, TP)    σ = exp(logStd)
              ↓
    Sample: a_i ~ N(μ_i, σ_i²), then clamp
```

**Pragmatic bias initialization** ensures the network starts near a sensible policy:

- `biasMean[0] = 5.0` → initial mean MCS ≈ 5
- `biasMean[1] = 2.0` → initial mean log₂(reps) ≈ 2 → 4 repetitions
- `biasMean[2] = 18.0` → initial mean TX power ≈ 18 dBm

**Critic (`ValueNetwork`):**

```
Input (14 features) → Linear(14→64)+tanh → Linear(64→64)+tanh → Linear(64→1) → V(s)
```

All weights: Xavier normal initialization. Actor LR: 0.0003; Critic LR: 0.001.

### State Vector (14 features)

Same 14-feature state as DDQN (see table above), computed identically from
`DeviceHistory` plus current channel and network observables.

### Actions (continuous, then discretized)

| Dimension      | Raw Output                        | Post-processing                   |
| -------------- | --------------------------------- | --------------------------------- |
| MCS            | sample ~ N(μ₀,σ₀²); clamp [0,10]  | round → uint8, clamp [0,10]       |
| log₂(reps)     | sample ~ N(μ₁,σ₁²); clamp [0,7]   | round → index → REPETITION_VALUES |
| TX power (dBm) | sample ~ N(μ₂,σ₂²); clamp [14,23] | round → double, clamp [14.0,23.0] |

Additionally, a hard **MCS ceiling** is always enforced:
`mcsCeiling = largest MCS where effectiveSINR ≥ requiredSINR + 2 dB`

### Log-Probability Computation

For a continuous Gaussian policy, the log-probability of action **a** given state **s**:

$$\log \pi(a|s) = \sum_{i=0}^{2} \left[ -\frac{1}{2}\frac{(a_i - \mu_i)^2}{\sigma_i^2 + \varepsilon} - \frac{1}{2}\log(2\pi(\sigma_i^2 + \varepsilon)) \right]$$

where ε = 1×10⁻⁸ prevents division by zero.

### PPO Clipped Surrogate Update (`updatePolicy`)

```
logRatio      = clamp(currentLogProb − oldLogProb, −20, 20)
ratio         = exp(logRatio)                      ← probability ratio π_new/π_old
clippedRatio  = clamp(ratio, 1−clip, 1+clip)       ← clip = 0.2
unclippedObj  = ratio × advantage
clippedObj    = clippedRatio × advantage

gradientScale = unclippedObj  if unclippedObj ≤ clippedObj
                0.0           otherwise        ← blocked update when outside clip

For each output dimension i:
  gradMean    = (a_i − μ_i) / var_i
  gradLogStd  = (a_i − μ_i)² / var_i − 1

  biasMean[i]   += lr × 0.05 × gradientScale × gradMean
  biasLogStd[i] += lr × 0.02 × (gradientScale × gradLogStd + entropyCoef)
  biasLogStd[i]  = clamp(biasLogStd[i], −2.0, 0.5)
```

Updates are intentionally applied only to the **output bias heads** (`biasMean`,
`biasLogStd`) for stability (lightweight approximation).

### GAE Advantage Computation (`train`)

```
Buffer collects 16 transitions, then:
  lastValue = V_critic(s_last) if not terminal, else 0
  For t from T−1 downto 0:
    delta[t]      = r[t] + γ · V(s[t+1]) − V(s[t])
    gae[t]        = delta[t] + γ · λ · gae[t+1]
    advantage[t]  = gae[t]
    return[t]     = advantage[t] + V(s[t])

Normalize advantages: adv_norm = (adv − mean(adv)) / (std(adv) + 1e−8)
```

### Reward Function

Same `calculateCongestionAdaptiveReward` as DDQN (see Algorithm 3 above), applied
identically to PPO's update loop.

### Variables

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `clipEpsilon`      | PPO clip parameter = 0.2                                  |
| `gamma` (γ)        | Discount factor = 0.95                                    |
| `gaeLambda` (λ)    | GAE smoothing factor = 0.95                               |
| `entropyCoeff`     | Entropy bonus coefficient = 0.01 (encourages exploration) |
| `updateEvery`      | Collect 16 steps between each training call               |
| `stepCount`        | Steps since last training call                            |
| `trajectoryBuffer` | On-policy rollout buffer (cleared after each train call)  |
| `t.logProb`        | Log-probability of the taken action under the old policy  |

### Constants

| Constant          | Value    | Notes                                     |
| ----------------- | -------- | ----------------------------------------- |
| INPUT_SIZE        | 14       | State dimension                           |
| HIDDEN_SIZE       | 64       | Both actor and critic hidden layers       |
| OUTPUT_SIZE       | 3        | Action dimensions (MCS, log2reps, TP)     |
| Actor LR          | 0.0003   | Actor gradient step                       |
| Critic LR         | 0.001    | Critic gradient step                      |
| clip              | 0.2      | PPO clipping radius                       |
| γ                 | 0.95     | Discount factor                           |
| λ                 | 0.95     | GAE decay factor                          |
| updateEvery       | 16       | Steps per training update                 |
| biasMean init     | [5,2,18] | Pragmatic start: MCS≈5, reps=4, TP≈18 dBm |
| biasLogStd init   | −1.0     | Initial std ≈ e^(−1) ≈ 0.37 per dimension |
| MCS safety margin | 2.0 dB   | Hard ceiling margin                       |

### How It Works

```
Per uplink feedback event:
  1. Update DeviceHistory
  2. Build 14-feature state vector s
  3. Compute congestion and reward r
  4. Actor: (μ, logStd) = PolicyNetwork.forward(s)
     Sample action: a_i ~ N(μ_i, σ_i); clamp to valid ranges
  5. Apply MCS ceiling: if action.mcs > mcsCeiling: action.mcs = mcsCeiling
  6. Assemble actionVec = [mcs, log2(reps), txPower, logProb]
  7. Store transition (s_prev, a_prev, r, s) in trajectory buffer
  8. Every 16 steps → train():
       a. Compute returns and GAE advantages (backward pass)
       b. Normalize advantages
       c. For each transition:
            critic.update(s, return)          ← MSE loss on value estimate
            actor.updatePolicy(s, a, oldLogProb, normAdv, clip=0.2, entropy=0.01)
       d. Clear trajectory buffer
  9. Apply new MCS, repetitions, TX power to UE
```

**Advantage:** Stable continuous-action policy; avoids large disruptive updates; entropy
bonus maintains exploration throughout training.  
**Disadvantage:** On-policy → each transition used only once; higher sample complexity
than DDQN in early training.

---

## Algorithm 5 — MARL (Multi-Agent Reinforcement Learning)

**ADRMethod::MARL** — Independent Q-Learning with eNB coordination signals.

### Description

MARL extends DDQN to a **multi-agent setting**. Each UE runs its own independent
`MARLAgent` (structurally identical to `OptimizedDDQNAgent`), but the state vector
includes **cell-wide coordination features** provided by the eNB. These features give
each agent awareness of how all other UEs in the cell are behaving, enabling implicit
coordination without explicit communication between devices.

During exploration, agents can identify when cell congestion is high and deliberately
choose actions that differ from the majority of other agents, reducing resource
contention (de-synchronization strategy).

Unlike CTDE (Centralized Training, Decentralized Execution) frameworks, this is fully
**decentralized** at both training and execution time — each agent trains independently
on its own replay buffer using only its own reward signal.

### Neural Network Architecture

Same Dueling DQN architecture as DDQN:

```
Input (14 features) → Shared FC(14→64)+LReLU → FC(64→64)+LReLU
  ↙ Value stream: FC(64→32)+LReLU → FC(32→1) = V(s)
  ↘ Advantage stream: FC(64→32)+LReLU → FC(32→60) = A(s,a)
Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
```

### Replay Buffer

Same PER buffer as DDQN: capacity 20,000, α=0.6, β=0.4.

### State Vector (14 features, `buildMARLState`)

The MARL state replaces the individual DDQN features 10–13 with cell-wide coordination
features from the eNB:

| Index | Feature                      | Formula / Source                          | Range   |
| ----- | ---------------------------- | ----------------------------------------- | ------- |
| 0     | Normalized RSRP              | Same as DDQN                              | [0, 1]  |
| 1     | Normalized SINR              | Same as DDQN                              | [0, 1]  |
| 2     | RSRP variance                | Same as DDQN                              | [0, 1]  |
| 3     | SINR trend                   | Same as DDQN                              | [−1, 1] |
| 4     | Current MCS (normalized)     | Same as DDQN                              | [0, 1]  |
| 5     | Current repetitions (log2)   | Same as DDQN                              | [0, 1]  |
| 6     | Current TX power             | Same as DDQN                              | [0, 1]  |
| 7     | Recent PDR                   | Same as DDQN (note: MARL uses position 7) | [0, 1]  |
| 8     | RACH success rate            | Same as DDQN                              | [0, 1]  |
| 9     | Consecutive failures         | Same as DDQN                              | [0, 1]  |
| 10    | **Cell load** ← eNB provided | `globalEnb->GetCellLoad()`                | [0, 1]  |
| 11    | **Active UE count** ← eNB    | min(1, numActiveUEs / 50)                 | [0, 1]  |
| 12    | **Average cell MCS** ← eNB   | mean(currentMCS across all UEs) / 10      | [0, 1]  |
| 13    | **Average cell reps** ← eNB  | log₂(mean(currentReps)) / 7               | [0, 1]  |

**Key difference from DDQN:** Features 10–13 are derived from aggregated cell-wide
measurements, not just this UE's own history. The eNB computes and broadcasts these
to each agent before it selects an action.

### Actions

Same 60-action `ExtendedActionSpace` as DDQN:

```
action = mcsAction × 12 + repAction × 3 + ruAction
```

### Coordination Mechanism

The `updateCoordination()` method stores each other agent's most recent action, MCS,
and repetitions:

```cpp
otherAgentActions[otherId] = action;
otherAgentMCS[otherId]     = mcs;
otherAgentReps[otherId]    = reps;
```

During **congested exploration**, instead of a random action, the agent selects the
**least-used action** across all agents to minimize RACH collisions:

```
if congestion detected:
  count how many agents chose each action
  return argmin_a [ actionCounts[a] ]
```

### Reward Function

Same `calculateCongestionAdaptiveReward` as DDQN and PPO. Importantly, because the
MARL agents now observe cell-level co-ordination features (avg MCS/reps, active UEs),
the reward landscape they optimize is effectively shaped by the collective behavior of
the cell — making convergent local optima more likely to be globally cooperative.

### Variables

| Variable            | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `epsilon`           | Per-agent exploration rate; starts 0.30                  |
| `epsilonDecay`      | 0.999 per step                                           |
| `epsilonMin`        | 0.05                                                     |
| `gamma` (γ)         | 0.95                                                     |
| `targetUpdateFreq`  | 50 steps between target network syncs                    |
| `otherAgentActions` | Map of neighbor agent → last action (coordination table) |
| `otherAgentMCS`     | Map of neighbor agent → last MCS                         |
| `otherAgentReps`    | Map of neighbor agent → last repetitions                 |
| `avgCellMCS`        | Mean MCS across all active UEs at current timestep       |
| `avgCellReps`       | Mean repetitions across all active UEs                   |
| `numActive`         | Count of UEs with active entries in `currentMCS`         |

### Constants

Same as DDQN (64 hidden, 60 actions, PER α=0.6, γ=0.95, etc.) except:

| Constant           | Value        | Notes                                |
| ------------------ | ------------ | ------------------------------------ |
| Max UEs for load   | 50           | Normalizes active UE count to [0,1]  |
| Coordination table | per-timestep | Updated before each action selection |

### How It Works

```
Per uplink feedback event (per UE):
  1. Update DeviceHistory (same as DDQN/PPO)
  2. Compute cell-wide averages:
       avgCellMCS  = mean(currentMCS[j]  for all UEs j)
       avgCellReps = mean(currentRepetitions[j] for all UEs j)
       numActive   = count of UEs
  3. Build 14-feature MARL state (includes eNB coordination features)
  4. Update coordination table from all other agents' current parameters
  5. Estimate congestion
  6. Compute reward r
  7. Select action a:
       with prob ε:
         if congested → choose least-used action (anti-collision heuristic)
         else         → random action
       else           → argmax_a Q(s, a)   [greedy from Dueling DQN]
  8. Decode action → (ΔMCS, Δreps, ΔRU)
  9. Apply mandatory MCS ceiling
  10. Store (s_prev, a_prev, r, s) in this UE's PER buffer
  11. Train Dueling DQN (same DDQN procedure: Double Q target, IS weights)
  12. Every 50 steps: copy main → target network
  13. ε ← max(εMin, ε × εDecay)
  14. Apply new MCS, reps, RU to UE
```

**Advantage:** Cell-aware state features allow implicit coordination; anti-collision
exploration avoids RACH pile-ups; each agent still trains fully independently on its own
reward signal (no centralized training required at execution time).  
**Disadvantage:** Coordination features (avg cell MCS/reps) require eNB feedback
channel, which may not always be available; the coordination is implicit and does not
guarantee a globally optimal multi-agent equilibrium.

---

## Side-by-Side Comparison

| Property              | SDBP    | 3GPP                | DDQN                 | PPO                      | MARL                                |
| --------------------- | ------- | ------------------- | -------------------- | ------------------------ | ----------------------------------- |
| **Type**              | Static  | Rule-based          | Value RL             | Policy RL                | Multi-agent RL                      |
| **State size**        | —       | SINR hist           | 14                   | 14                       | 14 (incl. cell-wide)                |
| **Action space**      | None    | Continuous (direct) | 60 discrete          | 3 continuous             | 60 discrete                         |
| **Online learning**   | No      | No (rules)          | Yes                  | Yes                      | Yes                                 |
| **Memory**            | —       | 20-step hist        | 20k PER buf          | 16-step traj             | 20k PER buf                         |
| **Network**           | —       | —                   | Dueling DQN (64/32)  | Actor+Critic (64)        | Dueling DQN (64/32)                 |
| **Exploration**       | —       | —                   | ε-greedy + curiosity | Gaussian noise + entropy | ε-greedy + anti-collision           |
| **RACH awareness**    | No      | Via PDR             | Yes (state+reward)   | Yes (state+reward)       | Yes (state+coord+reward)            |
| **Cell coordination** | No      | No                  | Cell load only       | Cell load only           | Full: avgMCS, avgReps, numUEs, load |
| **Convergence**       | Instant | Fast                | ~100+ steps          | ~16 steps/update         | ~100+ steps                         |
| **Energy modeling**   | No      | No                  | Yes (reward)         | Yes (reward)             | Yes (reward)                        |
