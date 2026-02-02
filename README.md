# Safe MPC-Based Reinforcement Learning with Learnable Control Barrier Functions

The code implements an MPC-based RL framework in which **Control Barrier Function (CBF) constraints are enforced inside MPC**, while the **class-K function** in the CBF condition is **parameterized and learned via RL**.


Paper:
**Kerim Dzhumageldyev, Filippo Airaldi, Azita Dabiri**  
*Safe model-based Reinforcement Learning via Model Predictive Control and Control Barrier Functions*
arXiv: https://arxiv.org/abs/2512.04856  
**Submitted to:** IFAC World Congress 2026

---

## Overview

The framework combines:
- **Model Predictive Control (MPC)** as the policy/value-function parameterization (via the MPC optimal control problem),
- **Reinforcement Learning (Q-learning)** to update learnable parameters,
- **CBF constraints** to guarantee safety through forward invariance of the safe set.

Learning updates are performed using sensitivities of the MPC solution.

---

## Implemented Methods

The repository includes the three CBF parameterization strategies introduced in the paper:

1. **Learnable Optimal-Decay CBF (OPTD/LOD-CBF)**  
   Learns optimal-decay parameters in the CBF condition via RL to improve feasibility and performance under input constraints.

2. **Neural Network CBF (NN-CBF)**  
   A feedforward neural network outputs **state-dependent decay parameters** used inside a discrete-time exponential CBF constraint.

3. **Recurrent Neural Network CBF (RNN-CBF)**  
   A recurrent neural network outputs **time-correlated decay parameters** across the MPC horizon, enabling time-varying safety constraints (e.g., moving obstacles).

---

## Running Experiments

Experiments are configured from `main.py` via explicit hyperparameters and environment settings. Typical parameters exposed in `main.py` include:
- exploration/noise scheduling (e.g., `initial_noise_scale`, `noise_variance`, `decay_at_end`)
- RL hyperparameters (e.g., `alpha`, `gamma`, `patience`, `lr_decay`)
- MPC settings (e.g., `episode_duration`, `mpc_horizon`)
- slack/violation penalties (e.g., `slack_penalty_MPC_L1`, `slack_penalty_RL_L1`, `violation_penalty`)
- obstacle configuration (positions, radii, motion modes and parameters)
- NN/RNN architecture (e.g., `hidden_dims`, `layers_list`)
- Initial learnable parameters (ditionary 'params_init')

---

## Note

The directory `old_files/` contains earlier development versions of the code. These are not maintained and very messy.
