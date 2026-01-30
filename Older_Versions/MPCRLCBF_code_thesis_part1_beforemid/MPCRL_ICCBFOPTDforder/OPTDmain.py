import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr
from OPTDfunctions import run_simulation, run_simulation_randomMPC, generate_experiment_notes
from OPTDclasses import RLclass, env



######### Main #########
#parameters for running the experiments

dt = 0.2
seed = 69
noise_scalingfactor = 4
noise_variance = 5
alpha = 2e-1
gamma = 0.95
episode_duration= 1000
num_episodes = 3000

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10# updates every 3 episodes


decay_at_end = 0.001
# decay_rate = 1 - np.power(decay_at_end, 1/num_episodes)
decay_rate = 1 - np.power(decay_at_end, 1/(num_episodes/episode_updatefreq))
print(f"decay_rate: {decay_rate}")

params_innit = {
    # State matrices
    "A": cs.DM([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ]) ,
    "B": cs.DM([
            [0.5 * dt**2, 0], 
            [0, 0.5 * dt**2], 
            [dt, 0], 
            [0, dt]
        ]),
    # Learned parameters
    "b":  cs.DM([0, 0, 0, 0]),
    "V0": cs.DM(0.0),
    "P" : 100*np.identity(4),
    "Q" : 10*np.identity(4), 
    "R" : np.identity(2),
    "Pw" : cs.DM(1000),
    "omega0": cs.DM(0.5),
}

#seems params_innit gets overwritten
params_original = params_innit.copy()

experiment_folder_name = "ICCBF_ADAM_1"

# cs.DM([
#              [200, 0, 0, 0], 
#              [0, 240, 0, 0], 
#             [0, 0, 30, 0], 
#              [0, 0, 0, 20]
#         ])

# THE ACTUAL PROGRAM TO RUN

run_simulation_randomMPC(params_innit, env, experiment_folder_name, episode_duration, noise_scalingfactor, noise_variance)

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, False)

rl = RLclass(params_innit, seed, alpha, dt, gamma, decay_rate, noise_scalingfactor, noise_variance)
params = rl.rl_trainingloop(episode_duration = episode_duration, num_episodes = num_episodes, replay_buffer=replay_buffer,  episode_updatefreq = episode_updatefreq, experiment_folder = experiment_folder_name)

stage_cost_sum_after = run_simulation(params, env, experiment_folder_name, episode_duration, True)


generate_experiment_notes(experiment_folder_name, params, params_original, episode_duration, num_episodes, seed, alpha, dt, gamma, decay_rate, 
                          decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, replay_buffer, episode_updatefreq)


suffix = f"_stagecost_{stage_cost_sum_after:.2f}"

new_folder_name = experiment_folder_name + suffix

# Rename the existing folder name
os.rename(experiment_folder_name, new_folder_name)



