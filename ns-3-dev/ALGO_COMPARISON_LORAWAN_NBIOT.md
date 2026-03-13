# LoRaWAN vs NB-IoT in This Project (DDQN / PPO / MARL)

## 1) How this project works in this area

Your codebase implements the same **policy family** in both stacks:

- Baselines: Static distance-based profile (SDBP) + Classical adaptation
- RL methods: **DDQN**, **PPO**, **MARL**

Then each simulation can run all methods (`--adr=all`) under the same scenario parameters for fair comparison.

- LoRaWAN simulator:
  - `src/lorawan/examples/advanced-ddqn-per-adr-example.cc`
- NB-IoT simulator:
  - `src/nbiot-adr/examples/advanced-ddqn-per-adr-nbiot.cc`
- NB-IoT sweep runner (34 scenarios):
  - `run_sweeps.py`

So your workflow is:

1. Fix scenario conditions (devices, radius/cell radius, interferers, app period, simulation time)
2. Run all 5 methods in identical conditions
3. Compare PDR (and energy where available)

---

## 2) Architecture-level difference in learning environment

| Aspect                     | LoRaWAN implementation                                      | NB-IoT implementation                                    |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| Access model               | LPWAN random access (Aloha-like contention effects)         | Cellular scheduled access + RACH procedure               |
| Main control knobs         | SF, TX power, channel                                       | MCS, repetitions, RU, TX power                           |
| Interference model         | WiFi/LTE + collision-centric behavior                       | Licensed inter-cell + LTE in-band load + RACH congestion |
| Network coordination       | Gateway/server side but less centralized scheduler behavior | eNB-centric coordination and cell-wide metrics           |
| Key reliability bottleneck | Collisions + weak links                                     | RACH failures/attempts + congestion + coverage           |

---

## 3) Algorithm matrix (what changes between stacks)

### DDQN

| Item               | LoRaWAN                                                                                  | NB-IoT                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Core idea          | Q-learning with replay/target network                                                    | Same                                                                                               |
| State flavor       | RF/link + recent packet behavior + channel context                                       | RF/link + repetition/RU + RACH and cell-load context                                               |
| Action space       | Small/basic DDQN (SF/TP), optimized extended version includes channel (up to 48 actions) | Basic DDQN (MCS/TP), optimized extended version includes MCS delta + repetitions + RU (60 actions) |
| Reward emphasis    | PDR-energy balance + collision/congestion handling + SF efficiency                       | PDR-energy + **RACH efficiency** + repetition efficiency + MCS/SINR margin                         |
| Practical behavior | Learns to trade SF/power/channel under collision pressure                                | Learns robust-vs-efficient transport profile (MCS/reps/RU/TP) under RACH + cell load               |

### PPO

| Item                    | LoRaWAN                                   | NB-IoT                                            |
| ----------------------- | ----------------------------------------- | ------------------------------------------------- |
| Core idea               | Actor-Critic PPO with continuous actions  | Same                                              |
| Continuous outputs      | SF and TX power                           | MCS, log2(repetitions), TX power                  |
| Post-processing         | Clamp/map to valid SF/TP range            | Clamp/map to valid MCS/repetition set/TP class    |
| Environment sensitivity | Collision/interference and ADR efficiency | RACH/cell-load and coverage-enhancement tradeoffs |

### MARL

| Item                | LoRaWAN                                        | NB-IoT                                                  |
| ------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Core idea           | Independent learners + coordination signals    | Same                                                    |
| Coordination signal | Channel occupancy / peer behavior              | Cell-wide eNB metrics (active UEs, avg MCS/reps, load)  |
| Conflict target     | Multi-node channel/SF contention               | Multi-UE RACH and shared radio-resource pressure        |
| Practical behavior  | Better de-synchronization/channel distribution | Better cell-level balancing of robustness vs efficiency |

---

## 4) Why results differ even with the same algorithm names

The algorithm name is the same, but the **MDP is different**:

- Different state variables
- Different action semantics
- Different constraints
- Different reward shaping terms

So `DDQN` in LoRaWAN is not directly equivalent to `DDQN` in NB-IoT policy behavior; same for PPO/MARL.

---

## 5) Thesis-ready one-line summary

In this project, DDQN/PPO/MARL are reused across LoRaWAN and NB-IoT as common RL frameworks, but each is re-parameterized to the stack-specific state/action/reward structure (collision-centric SF/channel control in LoRaWAN vs RACH/scheduler-centric MCS-repetition-RU control in NB-IoT), which explains different learning dynamics and performance outcomes.
