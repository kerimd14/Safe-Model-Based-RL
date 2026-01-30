
import numpy as np
import casadi as cs
import os
from NNclasses import env
from NNclasses import RLclass
from NNclasses import NN
from NNfunctions import run_simulation, run_simulation_randomMPC, generate_experiment_notes, plot_kappa_vs_h_for_states#, #plot_kappa_with_linear_baseline




######### Main #########
#parameters for running the experiments

dt = 0.2
seed = 69
noise_scalingfactor = 4
noise_variance = 5
alpha = 1e-1
gamma = 0.95
episode_duration= 3000
num_episodes = 3000

layers_list = [5, 8, 8, 1]

replay_buffer= 10*episode_duration #buffer is 5 episodes long
episode_updatefreq = 10# updates every 3 episodes

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

params_innit["nn_params"] = list

#seems params_innit gets overwritten
params_original = params_innit.copy() 

experiment_folder_name = "FIXEDNN_2"

# nn_model.plot_kappa(h_vals= None, params = params_innit, experiment_folder=experiment_folder_name)

states_to_test = [
    np.array([-5, -5, 0, 0]),
    np.array([-4, -3, 0.2, -0.1]),
    np.array([-3.5, -2.2, 0.0, 0.0]),
    np.array([-1.0, -4.0, 0.3, 0.1]),
]

# if you already have learned params:
# flat_params = learned_params  # cs.DM column

# otherwise let the function init its own
plot_kappa_vs_h_for_states(
    nn=nn_model,
    states_list=states_to_test,
    flat_params=None,      # or your learned params
    h_range=(-20, 40),
    n_points=400,
    show_raw=True,         # also draw raw NN(x,h) (dashed)
    title="κ(x,h) vs h across several fixed states",
)

# plot_kappa_with_linear_baseline(
#     nn=nn_model,
#     states_list=states_to_test,
#     flat_params=None,
#     h_range=(-1.0, 2.0),
#     n_points=500,
#     show_raw=False,         # paper only shows κ curves
#     linear_slope=None,      # auto-fit
#     fit_window=0.2,
#     title="learned κ vs fitted linear κ"
# )

# plot_kappa_with_linear_baseline(
#     nn=nn_model,
#     states_list=states_to_test,
#     flat_params=None,
#     h_range=(-1.0, 2.0),
#     n_points=500,
#     show_raw=False,
#     linear_slope=5.0,
#     title="learned κ vs linear κ (c=5)"
# )

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







