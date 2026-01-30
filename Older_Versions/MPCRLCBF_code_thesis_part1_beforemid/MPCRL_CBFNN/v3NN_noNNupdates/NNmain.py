
import numpy as np
import casadi as cs
from NNclasses import env
from NNclasses import RLclass
from NNclasses import NN
from NNfunctions import run_simulation, run_simulation_randomMPC, generate_experiment_notes




######### Main #########
#parameters for running the experiments

dt = 0.2
seed = 69
noise_scalingfactor = 7
noise_variance = 7
alpha = 5e-2
gamma = 0.95
episode_duration= 3000
num_episodes = 300

layers_list = [5, 7, 7,1]

decay_at_end = 0.0001
decay_rate = 1 - np.power(decay_at_end, 1/num_episodes)
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

params_innit["nn_params"] = list

#seems params_innit gets overwritten
params_original = params_innit.copy()

experiment_folder_name = "NN_trial_4"



run_simulation_randomMPC(params_innit, env, experiment_folder_name, episode_duration, layers_list, noise_scalingfactor, noise_variance)

stage_cost_sum_before = run_simulation(params_innit, env, experiment_folder_name, episode_duration, layers_list, False)

rl = RLclass(params_innit, seed, alpha, dt, gamma, decay_rate, layers_list, noise_scalingfactor, noise_variance)
params = rl.rl_trainingloop(episode_duration = episode_duration, num_episodes = num_episodes, experiment_folder = experiment_folder_name)

stage_cost_sum_after = run_simulation(params, env, experiment_folder_name, episode_duration, layers_list, True)


generate_experiment_notes(experiment_folder_name, params, params_original, episode_duration, num_episodes, seed, alpha, dt, 
                          gamma, decay_rate, decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, layers_list)








