
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import matplotlib.pyplot as plt
from alphaclasses import MPC



def stage_cost_func(action, x):
            """Computes the stage cost :math:`L(s,a)`.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])

            state = x
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action
            )
                

def MPC_func(x, mpc, params, solver_inst):
        
        # bounds
        X_lower_bound = -5 * np.array([1, 1, 1, 1])#-1e6 * 5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.array([1, 1, 1, 1])#1e6 * 5 * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound, np.zeros(mpc.horizon)])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound, np.inf*np.ones(mpc.horizon)])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])  
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        print(f"lbx: {lbx}")

        #params of MPC
        P = params["P"]
        Q = params["Q"]
        R = params["R"]
        V = params["V0"]

    
        #flatten
        A_flat = cs.reshape(params["A"] , -1, 1)
        B_flat = cs.reshape(params["B"] , -1, 1)
        P_diag = cs.diag(P) #cs.reshape(P , -1, 1)
        Q_flat = cs.reshape(Q , -1, 1)
        R_flat = cs.reshape(R , -1, 1)

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["alpha"]),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        # print(f"omega parameter: {solution['x'][-1]}")
        # print(f"the whole solution: {solution['x']}")

        return u_opt, solution["f"]

# def save_figure(fig, filename, experiment_folder):
     
#     save_choice = input("Save the figure? (y/n): ")
#     if save_choice == "y":
#         os.makedirs(experiment_folder, exist_ok=True) # make directory ( exist_ok makes sure it doenst throw exception when it alreadt exists)
#         file_path = os.path.join(experiment_folder, filename) # add the file to directory
#         fig.savefig(file_path)
#         print(f"Figure saved as: {file_path}")
#     else:
#          print("Figure not saved")

def save_figures(figures, experiment_folder):
        save_choice = True#input("Save the figure? (y/n): ")
        if save_choice == True:#"y":
            os.makedirs(experiment_folder, exist_ok=True) # make directory ( exist_ok makes sure it doenst throw exception when it alreadt exists)
            for fig, filename in figures: 
                file_path = os.path.join(experiment_folder, filename) # add the file to directory
                fig.savefig(file_path)
                print(f"Figure saved as: {file_path}")
        else:
            print("Figure not saved")

def save_notes(experiment_folder, notes, filename="notes.txt"):
    os.makedirs(experiment_folder, exist_ok=True)
    notes_path = os.path.join(experiment_folder, filename)
    with open(notes_path, "w") as file:
        file.write(notes)


def calculate_trajectory_length(states):
    # compute pairwise Euclidean distances and sum everything
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)



def run_simulation(params, env, experiment_folder, episode_duration, after_updates):

    env = env(sampling_time=0.2)
    
    state, _ = env.reset(seed=69, options={})
    states = [state]
    actions = []
    stage_cost = []
    mpc = MPC(0.2)

    solver_inst = mpc.MPC_solver() 

    for i in range(episode_duration):
        action, _ = MPC_func(state, mpc, params, solver_inst)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)

        stage_cost.append(stage_cost_func(action, state))

        print(i)

        if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
            break

    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    stage_cost = stage_cost.reshape(-1) 

    figstates=plt.figure()
    plt.plot(
        states[:, 0], states[:, 1],
        "o-"
    )

    # Plot the obstacle
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])

    # Set labels and title
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title("Trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid()

    figactions=plt.figure()
    plt.plot(actions[:, 0], "o-", label="Action 1")
    plt.plot(actions[:, 1], "o-", label="Action 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Action")
    plt.title("Actions")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    figstagecost=plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("Stage Cost")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    
    if after_updates == False:
        figs = [
                    (figstates, "states_MPCregular_beforeupdates"),
                    (figactions, "actions_MPCregular_beforeupdates"),
                    (figstagecost, "stagecost_MPCregular_beforeupdates"),
                    (figsvelocity, "velocity_MPCregular_beforeupdates")
                ]
    else:
         figs = [
                    (figstates, "states_MPCregular_afterupdates"),
                    (figactions, "actions_MPCregular_afterupdates"),
                    (figstagecost, "stagecost_MPCregular_afterupdates"),
                    (figsvelocity, "velocity_MPCregular_afterupdates")
                ]

    save_figures(figs,  experiment_folder)

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

    return sum(stage_cost)

def MPC_func_random(x, mpc, params, solver_inst, rand_noise):
        dt = 0.2
        
        # bounds
        X_lower_bound = -5 * np.array([1, 1, 1, 1])#-1e6 * 5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.array([1, 1, 1, 1])#1e6 * 5 * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound, np.zeros(mpc.horizon)])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound, np.inf*np.ones(mpc.horizon)])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])  
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        #params of MPC
        P = params["P"]
        Q = params["Q"]
        R = params["R"]
        V = params["V0"]

    
        #flatten
        A_flat = cs.reshape(params["A"] , -1, 1)
        B_flat = cs.reshape(params["B"] , -1, 1)
        P_diag = cs.diag(P) #cs.reshape(P , -1, 1)
        Q_flat = cs.reshape(Q , -1, 1)
        R_flat = cs.reshape(R , -1, 1)


        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["alpha"], rand_noise),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
  


        return u_opt, solution["f"]

def run_simulation_randomMPC(params, env, experiment_folder, episode_duration, noise_scalingfactor, noise_variance):

    env = env(sampling_time=0.2)


    np_random = np.random.default_rng(seed=69)
    state, _ = env.reset(seed=69, options={})
    states = [state]
    actions = []
    stage_cost = []
    mpc = MPC(0.2)

    solver_inst = mpc.MPC_solver_rand() 

    for i in range(episode_duration):
        rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        action, _ = MPC_func_random(state, mpc, params, solver_inst, rand_noise=rand_noise)

        # if i<(0.65*2000):
        # else:f
        #     action, _ = MPC_func(state, mpc, params)
        # action, _ = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)

        stage_cost.append(stage_cost_func(action, state))

        print(i)

        # if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
        #     break

    
    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    stage_cost = stage_cost.reshape(-1) 


    figstates=plt.figure()
    plt.plot(
        states[:, 0], states[:, 1],
        "o-"
    )

    # Plot the obstacle
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])

    # Set labels and title
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title("Trajectories"
    "")
    plt.legend()
    plt.axis("equal")
    plt.grid()

    figactions=plt.figure()
    plt.plot(actions[:, 0], "o-", label="Action 1")
    plt.plot(actions[:, 1], "o-", label="Action 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Action")
    plt.title("Actions")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # 

    figstagecost=plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("Stage Cost")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    

    figs = [
                (figstates, "states_MPCnoise"),
                (figactions, "actions_MPCnoise"),
                (figstagecost, "stagecost_MPCrandom"),
                (figsvelocity, "velocity_MPCnoise")
            ]

    save_figures(figs,  experiment_folder)

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

def generate_experiment_notes(experiment_folder, params, params_innit, episode_duration, num_episodes, seed, alpha, sampling_time, gamma, decay_rate, decay_at_end, 
                              noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, replay_buffer, episode_updatefreq):
    # used to save the parameters automatically

    notes = f"""
    Experiment Settings:
    --------------------
    Episode Duration: {episode_duration}
    Number of Episodes: {num_episodes}
    Sampling time: {sampling_time}

    Learning Parameters:
    --------------------
    Seed: {seed}
    Alpha (Learning Rate): {alpha}
    Decay Rate of Noise: {decay_rate}
    Decay At end of Noise: {decay_at_end}
    Initial Noise scaling factor: {noise_scalingfactor}
    Moise variance: {noise_variance}
    Gamma: {gamma}
    Replay Buffer: {replay_buffer}
    Episode Update Frequency: {episode_updatefreq} # for example performs updates every 3 episodes

    MPC Parameters Before Learning:
    --------------
    P Matrix: {params_innit['P']}
    V : {params_innit['V0']}

    MPC Parameters After Learning:
    ---------------
    P Matrix: {params['P']}
    V : {params['V0']}

    Stage Cost:
    ---------------
    Summed Stage Cost of simulation before update: {stage_cost_sum_before}
    Summed Stage Cost of simulation after updates: {stage_cost_sum_after}

    Neural Network params:
    ---------------
    Initialized alpha: {params_innit['alpha']}
    Learned alpha: {params['alpha']}



    Additional Notes:
    -----------------
    - Off-policy training with initial parameters
    - Noise scaling based on distance to target
    - Decay rate applied to noise over iterations
    - scaling adjused

    """
    save_notes(experiment_folder, notes)
