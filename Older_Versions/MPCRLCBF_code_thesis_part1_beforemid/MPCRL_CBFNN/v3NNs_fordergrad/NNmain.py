
import numpy as np
import casadi as cs
import os
from NNclasses import env
from NNclasses import RLclass
from NNclasses import NN
from NNfunctions import run_simulation, run_simulation_randomMPC, generate_experiment_notes




######### Main #########
#parameters for running the experiments

dt = 0.2
seed = 69
noise_scalingfactor = 4
noise_variance = 5
alpha = 5e-1
gamma = 0.95
episode_duration= 3000
num_episodes = 3000

layers_list = [5, 8, 8, 8, 1]

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10 #updates every 3 episodes

patience_threshold = 500000
lr_decay_factor = 0.1

decay_at_end = 0.001
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

#initializing the NN
nn_model = NN(layers_list)

list, _, _ = nn_model.initialize_parameters()

# params_innit["nn_params"] = [8.68988, 0.113226, 0.94921, 0.606646, -0.019237, 0.714464, 0.0135782, -0.145787, -0.985668, 0.55075, 0.73875, 0.613452, 1.03654, 0.52131, -1.17257, 0.962697, 0.73258, 0.394336, 0.38787, -0.817604, -1.05808, 0.194632, 0.80375, 0.510693, 0.781365, -0.411421, -0.178558, -0.350625, 0.112494, 0.7624, -0.00629666, 0.624698, 1.00585, -0.0235127, 0.142683, 0.241291, -0.0427109, 0.780393, -0.108307, -0.518297, -0.0396565, -0.535967, 0.0880232, 0.037587, 0.606849, -0.321823, 0.588068, -0.640506, -0.037634, 0.537233, -0.33394, -0.107361, 0.366882, 0.357039, -0.717554, -0.676827, -0.68908, 0.264525, -0.0107465, 0.626145, -0.624284, 0.648139, 0.56465, 0.136729, -0.313771, 0.158473, 0.331849, 0.0619711, -0.625714, 0.225135, 0.512166, -0.365256, 0.00899511, 0.776962, 0.0668224, -0.540994, -0.482998, -0.620313, 0.43786, 0.553294, 0.606352, -0.540575, 0.0637099, 0.162135, -0.763058, -0.54968, 0.00216372, -0.631179, 0.33471, -0.0725024, -0.167258, 0.731214, -0.885691, 0.663786, -0.763905, -0.267674, -0.298441, 0.754004, 0.527154, -0.154886, -0.782285, 0.674843, 0.455167, -0.221651, -0.729416, -0.0563607, 0.59696, -0.352609, -0.33474, 0.564971, -0.700854, 0.520901, 0.549813, -0.118321, 0.150477, 0.31034, 0.0657322, 0.267532, 0.331889, -0.139547, 0.372086, -0.147198, 0.523142, -0.306808, -0.402896, -0.285125, 0.215587, -0.133174, -0.245199, -0.763946, -0.696176, -0.0636758, -0.810889, 0.079286, 0.0697291, -0.136758, 0.45202, -0.673474, -0.398644, 0.0468211, -0.0166751, -0.938132, -0.70155, 0.0866476, 0.556725, -0.901438, 0.362193, 0.0219599, 0.536376, -0.623307, -0.779672, 0.672602, -0.59251, -0.372438, -0.818159, 0.366653, -0.778137, -0.325257, -0.57001, -0.516975, 0.922701, 0.560242, 0.570845, 0.0522808, 0.798454, 0.864793, -0.0278299, -0.190858, 0.027215, 0.500691, 0.187527, -0.307794, 0.39467, -0.384384, 0.762793, -0.161723, -0.0956068, -0.0883591, -0.11264, -0.13001, 0.00395559, -0.0259626, 0.13453, -0.0670972, 0.266076, -0.200598, 0.0171076, 0.11844, 0.0123189, -0.0731845, 0.0491683, 0.010987, -0.175011, -0.199556, -0.0794244, 0.0457445, -0.155767, 0.118186, -0.139033, 0.214829, -0.245529]
params_innit["nn_params"] = list
#seems params_innit gets overwritten
params_original = params_innit.copy() 

experiment_folder_name = "NNforderADAM_21"



# run_simulation_randomMPC(params_innit, env, experiment_folder_name, episode_duration, layers_list, noise_scalingfactor, noise_variance)

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, layers_list, False)

rl = RLclass(params_innit, seed, alpha, dt, gamma, decay_rate, layers_list, noise_scalingfactor, noise_variance, patience_threshold, lr_decay_factor)
params = rl.rl_trainingloop(episode_duration = episode_duration, num_episodes = num_episodes, replay_buffer=replay_buffer, episode_updatefreq = episode_updatefreq, experiment_folder = experiment_folder_name)

stage_cost_sum_after = run_simulation(params, env, experiment_folder_name, episode_duration, layers_list, True)


generate_experiment_notes(experiment_folder_name, params, params_original, episode_duration, num_episodes, seed, alpha, dt, 
                          gamma, decay_rate, decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, 
                          stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq, patience_threshold, lr_decay_factor)


suffix = f"_stagecost_{stage_cost_sum_after:.2f}"

new_folder_name = experiment_folder_name + suffix

# Rename the existing folder name
os.rename(experiment_folder_name, new_folder_name)







