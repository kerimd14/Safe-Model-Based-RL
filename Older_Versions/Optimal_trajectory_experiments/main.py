
import numpy as np
import casadi as cs
import os
import copy
from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U
from Classes import env
from Classes import NN
from Functions import run_simulation, generate_experiment_notes





######### Main #########
#parameters for running the experiments

dt = SAMPLING_TIME
seed = SEED
noise_scalingfactor = 10
noise_variance = 5
alpha = 9e-2
gamma = 0.95
episode_duration= 500
num_episodes = 1000

# MPC HORIZON
horizon = 100

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10# updates every 3 episodes

patience_threshold = 10000
lr_decay_factor = 0.1

decay_at_end = 0.01
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
    "R" : np.identity(2)
}


positions = [(-2, -2.25)]


radii     = [1.5]  

modes       = ["static"]  # one obstacle, static

mode_params = [{'bounds': (-5.0, 1.0), 'speed': 0}]


layers_list = [NUM_STATES+len(positions), 14, 14, 14, len(positions)]
print("layers_list: ", layers_list)
#initializing the NN
nn_model = NN(layers_list, positions, radii)

list, _, _ = nn_model.initialize_parameters()

params_innit["nn_params"] = list

#seems params_innit gets overwritten
params_original = params_innit.copy() 

experiment_folder_name = "PLEASSEEE"

#WATCH OUT --> since mode_params was a dict what was happening is that the values of it kept getting changed

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, layers_list, False, horizon, positions, radii, modes, copy.deepcopy(mode_params))


new_folder_name = experiment_folder_name

# Rename the existing folder name
os.rename(experiment_folder_name, new_folder_name)




# PROBLEMS:

# I noticed that i punished cost for using slacks differently in MPC cost func and RL cost func
# MPC cost func: np.sum(2e6 *S)
# RL cost func: 2e6 * np.sum(S)

# SOLUTION: I changed the RL cost func to match the MPC one







