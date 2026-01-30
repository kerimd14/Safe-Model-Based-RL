import os
import copy
import numpy as np
import casadi as cs

from config import (
    SAMPLING_TIME,
    SEED,
    NUM_STATES,
    NUM_INPUTS,
    CONSTRAINTS_X,
    CONSTRAINTS_U,
)
from mpc.validation import run_simulation
from rnn.rnn import RNN
from envs.env import env as Env
from viz.viz import generate_experiment_notes

from mpc.validation import run_simulation
from viz.viz import generate_experiment_notes
from viz import viz as viz_mod

from rl.trainer import Trainer
from rl.evaluator import Evaluator

from rl.lrscheduler import LearningRateScheduler  # <- dataclass/state object
from rl.rlagent import RLagent 

def main():
    """
    Main for training a CBF RNN with MPC in a moving-obstacle environment.
    
    
    """
    
    # ─── Experiment variables ────────────────────────────────────────────
    dt = SAMPLING_TIME
    seed = SEED

    # Noise / exploration schedule
    initial_noise_scale = 10
    noise_variance = 5
    decay_at_end = 0.01
    
    num_episodes = 12
    episode_update_freq = 3  # frequency of updates (e.g. update every 10 episodes)
    decay_rate = 1 - np.power(decay_at_end, 1 / (num_episodes / episode_update_freq))
    print(f"Computed noise decay_rate: {decay_rate:.4f}")

    # RL hyper-parameters
    alpha = 7e-5      # initial learning rate
    gamma = 0.99       # discount factor
    slack_penalty_MPC_L1 = 2e7  # penalty on slack variables in CBF constraints for the MPC stage cost
    slack_penalty_MPC_L2 = 0#1e3
    slack_penalty_RL_L1 = 2e7 # penalty on slack variables in CBF constraints for RL stage cost
    slack_penalty_RL_L2 = 0#1e3 # penalty on slack variables in CBF constraints for RL stage cost
    violation_penalty = 0#4e5  # penalty on constraint violation (used in stage cost function)
    # Learning rate scheduler
    # patience = number of epochs with no improvement after which learning rate will be reduced
    patience = 5
    lr_decay = 0.1     # factor to shrink the learning rate with after patience is reached

    # Episode / MPC specs
    episode_duration = 150
    mpc_horizon = 6
    replay_buffer_size = episode_duration * episode_update_freq  # buffer holding number of episodes (e.g. hold 10 episodes)
    
    
    #name of folder where the experiment is saved
    experiment_folder = "SRNNCBF_4"
    
    
    # ──Linear dynamics and MPC parameters───────────────────────────────────
    params_init = {
        "A": cs.DM([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        "B": cs.DM([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt],
        ]),
        "b": cs.DM([0, 0, 0, 0]),
        "V0": cs.DM(0.0),
        "P": 100 * np.eye(NUM_STATES),
        "Q": 10 * np.eye(NUM_STATES),
        "R": np.eye(NUM_INPUTS),
    }
    
     # ─── Obstacle configuration ──────────────────────────────────────────────
     
    # positions = [(-2.0, -1.5), (-3.0, -3.0)]
    # radii     = [0.75, 0.75]
    # modes     = ["step_bounce", "step_bounce"]
    # mode_params = [
    #     {"bounds": (-4.0,  0.0), "speed": 2.3, "dir":  1},
    #     {"bounds": (-4.0,  1.0), "speed": 2.0, "dir": -1},
    # ]
    positions = [(-2.0, -1.5), (-3.0, -3.3), (-2.0, 0.0)]
    radii     = [0.7, 0.7, 1]
    modes     = ["step_bounce", "step_bounce", "static"]
    mode_params = [
    {"bounds": (-4.0,  0), "speed": 2.3, "dir":  1},
    {"bounds": (-4.0,  1.0), "speed": 2.0, "dir": -1},
    {"bounds": (-2.0,  -2.0), "speed": 0.0},
]
    
    # ─── Build & initialize RNN CBF ───────────────────────────────────────────

    input_dim = NUM_STATES + len(positions) + 2*len(positions)+NUM_INPUTS #x+h(x)+ (obs_positions_x + obs_positions_y) + u
    hidden_dims = [32]
    output_dim = len(positions)
    layers_list = [input_dim] + hidden_dims + [output_dim]
    print("RNN layers:", layers_list)

    rnn = RNN(layers_list, positions, radii, mpc_horizon, copy.deepcopy(mode_params), modes)
    flat_rnn_params, _, _, _ = rnn.initialize_parameters()
    params_init["rnn_params"] = flat_rnn_params

    # keep a copy of the original parameters for later logging
    params_before = params_init.copy()
    
    
    # ─── The Learning  ─────────────────────────────────────
    
    # run simulation to get the initial policy before training
    stage_cost_before = run_simulation(
    params_init,
    Env,
    experiment_folder,
    episode_duration,
    layers_list,
    after_updates=False,
    horizon=mpc_horizon,
    positions=positions,
    radii=radii,
    modes=modes,
    mode_params=copy.deepcopy(mode_params),
    slack_penalty_MPC_L1=slack_penalty_MPC_L1,
    slack_penalty_MPC_L2=slack_penalty_MPC_L2,
    )

    # 2) scheduler state object (NOT the update function)
    lr_sched = LearningRateScheduler(
        alpha=alpha,
        patience_threshold=patience,
        lr_decay_factor=lr_decay,
    )

    # 3) agent object (use your refactored RLAgent OR your existing RLclass)
    agent = RLagent(
        params_init=params_init,
        seed=seed,
        alpha=alpha,                    # optional if you set agent.alpha later
        gamma=gamma,
        decay_rate=decay_rate,
        layers_list=layers_list,
        noise_scalingfactor=initial_noise_scale,
        noise_variance=noise_variance,
        patience_threshold=patience,
        lr_decay_factor=lr_decay,
        horizon=mpc_horizon,
        positions=positions,
        radii=radii,
        modes=modes,
        mode_params=copy.deepcopy(mode_params),
        slack_penalty_MPC_L1=slack_penalty_MPC_L1,
        slack_penalty_MPC_L2=slack_penalty_MPC_L2,
        slack_penalty_RL_L1=slack_penalty_RL_L1,
        slack_penalty_RL_L2=slack_penalty_RL_L2,
        violation_penalty=violation_penalty,
    )

    # make sure agent uses scheduler alpha as source of truth
    agent.alpha = lr_sched.alpha

    # 4) evaluator object (separate thing)
    evaluator = Evaluator(
        agent,
        viz_mod
    )

    # 5) trainer
    trainer = Trainer(agent=agent, evaluator=evaluator, viz=viz_mod, lr_scheduler=lr_sched)

    # 6) train
    trained_params = trainer.rl_trainingloop(
        episode_duration=episode_duration,
        num_episodes=num_episodes,
        replay_buffer=replay_buffer_size,
        episode_updatefreq=episode_update_freq,
        experiment_folder=experiment_folder,
    )

    # 7) stage_cost_after (KEEP EXACTLY AS YOU HAVE)
    stage_cost_after = run_simulation(
        trained_params,
        Env,
        experiment_folder,
        episode_duration,
        layers_list,
        after_updates=True,
        horizon=mpc_horizon,
        positions=positions,
        radii=radii,
        modes=modes,
        mode_params=copy.deepcopy(mode_params),
        slack_penalty_MPC_L1=slack_penalty_MPC_L1,
        slack_penalty_MPC_L2=slack_penalty_MPC_L2,
    )

    # 8) notes (KEEP EXACTLY AS YOU HAVE)
    generate_experiment_notes(
        experiment_folder,
        trained_params,
        params_before,
        episode_duration,
        num_episodes,
        seed,
        alpha,
        dt,
        gamma,
        decay_rate,
        decay_at_end,
        initial_noise_scale,
        noise_variance,
        stage_cost_before,
        stage_cost_after,
        layers_list,
        replay_buffer_size,
        episode_update_freq,
        patience,
        lr_decay,
        mpc_horizon,
        modes,
        copy.deepcopy(mode_params),
        positions,
        radii,
        slack_penalty_MPC_L1,
        slack_penalty_MPC_L2,
        slack_penalty_RL_L1,
        slack_penalty_RL_L2,
        violation_penalty
    )
    
if __name__ == "__main__":
    main()

