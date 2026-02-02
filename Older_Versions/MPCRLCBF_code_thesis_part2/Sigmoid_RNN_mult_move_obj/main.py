import multiprocessing
import os
import copy
import optuna
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
from Classes import env, RLclass, RNN
from Functions import (
    run_simulation,
    run_simulation_randomMPC,
    generate_experiment_notes,
    objective,
    save_best_results
)


def main():
    """
    Main for raining a CBF RNN with MPC in a moving-obstacle environment.
    """
    
    # ─── Experiment variables ────────────────────────────────────────────
    dt = SAMPLING_TIME
    seed = SEED

    # Noise / exploration schedule
    initial_noise_scale = 10
    noise_variance = 5
    decay_at_end = 0.01
    
    num_episodes = 3000
    episode_update_freq = 3  # frequency of updates (e.g. update every 10 episodes)
    decay_rate = 1 - np.power(decay_at_end, 1 / (num_episodes / episode_update_freq))
    print(f"Computed noise decay_rate: {decay_rate:.4f}")

    # RL hyper-parameters
    alpha = 7e-5      # initial learning rate
    gamma = 0.95       # discount factor
    slack_penalty_MPC_L1 = 2e7  # penalty on slack variables in CBF constraints for the MPC stage cost
    slack_penalty_MPC_L2 = 0#1e3
    slack_penalty_RL_L1 = 5e3 # penalty on slack variables in CBF constraints for RL stage cost
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
    experiment_folder = "RNN_mult_move_obj_experiment_352"
    
    #check if file exists already, if yes raise an exception
    # if os.path.exists(experiment_folder):
    #     raise FileExistsError(f"Experiment folder '{experiment_folder}' already exists. Please choose a different name.")
    
    
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

    input_dim = NUM_STATES + len(positions) + 2*len(positions) #x+h(x)+ (obs_positions_x + obs_positions_y)
    hidden_dims = [16, 16, 16]#[16, 16, 16]
    output_dim = len(positions)
    layers_list = [input_dim] + hidden_dims + [output_dim]
    print("RNN layers:", layers_list)

    rnn = RNN(layers_list, positions, radii, mpc_horizon, copy.deepcopy(mode_params), modes)
    flat_rnn_params, _, _, _ = rnn.initialize_parameters()
    params_init["rnn_params"] = flat_rnn_params

    # keep a copy of the original parameters for later logging
    params_before = params_init.copy()
    
    
    # ─── The Learning  ─────────────────────────────────────
    
    # run simulation of random MPC to see how the system behaves under initial random noise
    
    # run_simulation_randomMPC(
    #     params_init,
    #     env,
    #     experiment_folder,
    #     episode_duration,
    #     layers_list,
    #     initial_noise_scale,
    #     noise_variance,
    #     mpc_horizon,
    #     positions,
    #     radii,
    #     modes,
    #     copy.deepcopy(mode_params),
    #     slack_penalty_MPC_L1, 
    #     slack_penalty_MPC_L2,
    # )
    
    # run simulation to get the initial policy before training
    stage_cost_before = run_simulation(
        params_init,
        env,
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
    
    # use RL to train the RNN CBF with MPC
    
    rl_agent = RLclass(
        params_init,
        seed,
        alpha,
        gamma,
        decay_rate,
        layers_list,
        initial_noise_scale,
        noise_variance,
        patience,
        lr_decay,
        mpc_horizon,
        positions,
        radii,
        modes,
        copy.deepcopy(mode_params),
        slack_penalty_MPC_L1,
        slack_penalty_MPC_L2,
        slack_penalty_RL_L1,
        slack_penalty_RL_L2,
        violation_penalty
    )
    trained_params = rl_agent.rl_trainingloop(
        episode_duration=episode_duration,
        num_episodes=num_episodes,
        replay_buffer=replay_buffer_size,
        episode_updatefreq=episode_update_freq,
        experiment_folder=experiment_folder,
    )
    
    # evaluate the trained policy
    
    stage_cost_after = run_simulation(
        trained_params,
        env,
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
    
    #save experiment configuration and results
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

    # append final stage-cost to folder name
    suffix = f"_stagecost_{stage_cost_after:.2f}"
    os.rename(experiment_folder, experiment_folder + suffix)
    
    
if __name__ == "__main__":
    main()
# BASE_DIR = "optuna_runs_1"
# if __name__ == "__main__":
    
#     os.makedirs(BASE_DIR, exist_ok=True)

#     storage_path = os.path.join(BASE_DIR, "optuna.db")
#     storage_uri  = f"sqlite:///{storage_path}"
    
#     # optuna.delete_study(study_name="my_rnn_mpc_study", storage="sqlite:///optuna_runs/optuna.db")
#     study = optuna.create_study(
#         storage=storage_uri,  
#         study_name="rnn_mpc_study",
#         direction="minimize",
#         sampler=optuna.samplers.TPESampler(),
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
#         load_if_exists=True,   # <-- resume if the DB file already exists
#     )
#     optuna.logging.set_verbosity(optuna.logging.INFO)
    
#     #n_jobs = 1 to run trials sequentially, running in parallel is not supported in this context
#     # matplotlib and casadi are not thread-safe
#     study.optimize(objective, n_trials=50, n_jobs = 1)

#     print("Best trial:",  study.best_trial.number)
#     print("  Value: ",    study.best_value)
#     print("  Params:",    study.best_params)
    
#     save_best_results(study)

