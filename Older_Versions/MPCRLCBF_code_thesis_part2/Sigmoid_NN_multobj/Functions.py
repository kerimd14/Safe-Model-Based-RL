
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import matplotlib.pyplot as plt
from Classes import MPC
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U

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
        
        fwd_func = mpc.nn.numerical_forward()
        
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x)]
        alpha.append(cs.DM(fwd_func(x, h_func_list,  params["nn_params"])))

        # bounds
        X_lower_bound = -CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(mpc.nn.obst.obstacle_num * (mpc.horizon))
        cbf_const_ubg = np.zeros(mpc.nn.obst.obstacle_num * (mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound])

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

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["nn_params"]),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )

        g_resid = solution["g"][4:]        # vector of all g(x)

        print(f"g reid: {g_resid}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"], alpha, g_resid

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



def run_simulation(params, env, experiment_folder, episode_duration, layers_list, after_updates, horizon, positions, radii):

    env = env()
    mpc = MPC(layers_list, horizon, positions, radii)

   
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    g_resid_lst = []    
    
    # extract list of h functions
    h_func_list = mpc.nn.obst.make_h_functions()

    alphas = []
    #cycle through to plot different h functions later
    hx = [ np.array([ float(hf(cs.DM(state))) for hf in h_func_list ]) ]
    # hx = []

    solver_inst = mpc.MPC_solver_noslack() 


    for i in range(episode_duration):


        action, _, alpha, g_resid = MPC_func(state, mpc, params, solver_inst)

        alphas.append(alpha)


        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        print(f"state from env: {state}")
        states.append(state)
        actions.append(action)
        g_resid_lst.append(-g_resid)

        hx.append(np.array([ float(hf(cs.DM(state))) for hf in h_func_list ]))

        stage_cost.append(stage_cost_func(action, state))

        print(i)

        if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
            break


    print(f"alphas: {alphas}")
    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    g_resid_lst = np.array(g_resid_lst)
    hx = np.vstack(hx)
    alphas = np.array(alphas)
    print(f"alphas shape: {alphas.shape}")
    alphas = np.squeeze(alphas)  # remove single-dimensional entries from the shape
    print(f"alphas shape: {alphas.shape}")

    stage_cost = stage_cost.reshape(-1) 

    fig_states = plt.figure()
    plt.plot(states[:, 0], states[:, 1], "o-", label="trajectory")
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
        plt.gca().add_patch(circle)
    plt.xlim([-CONSTRAINTS_X, CONSTRAINTS_X])
    plt.ylim([-CONSTRAINTS_X, CONSTRAINTS_X])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("State Trajectory")
    plt.axis("equal")
    plt.grid()
    plt.legend()


    fig_actions = plt.figure()
    plt.plot(actions[:, 0], "o-", label="Action 1")
    plt.plot(actions[:, 1], "o-", label="Action 2")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Action")
    plt.title("Actions Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    fig_stagecost = plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Stage Cost")
    plt.title("Stage Cost Over Time")
    plt.grid()
    plt.tight_layout()

    fig_velocity = plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity $v_x$")
    plt.plot(states[:, 3], "o-", label="Velocity $v_y$")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Velocity Value")
    plt.title("Velocities Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    
    m = mpc.nn.obst.obstacle_num  # number of obstacles
    print(f"shape of alphas: {alphas.shape}")
    

    fig_alpha = plt.figure()
    if m == 1:
        plt.plot(alphas, "o-", label="$\\alpha(x_k)$")
    else:
        for i in range(m):
            plt.plot(alphas[:, i], "o-", label=f"$\\alpha_{{{i+1}}}(x_k)$")
    plt.xlabel("Iteration $k$")
    plt.ylabel("$\\alpha_i(x_k)$")
    plt.title("Neural‐Network Outputs $\\alpha_i$ (one per obstacle)")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid()
    plt.tight_layout()

    
    hx_figs = []
    for i in range(m):
        fig_hi = plt.figure()
        plt.plot(hx[:, i], "o-", label=f"$h_{i+1}(x_k)$")
        plt.xlabel("Iteration $k$")
        plt.ylabel(f"$h_{i+1}(x_k)$")
        plt.title(f"Obstacle {i+1}: $h_{i+1}(x_k)$ Over Time")
        plt.grid()
        plt.tight_layout()
        hx_figs.append((fig_hi, f"hx_obstacle_{i+1}.png"))

    #    margin_i[k] = h_i(x_{k+1}) − (1 − α_k)·h_i(x_k)

    # Only compute if you want margins. If not, skip this block.
    # T = hx.shape[0] - 1
    # margin = np.zeros((T, m))
    # for k in range(T):
    #     for i in range(m):
    #         margin[k, i] = hx[k+1, i] - (1 - alphas[k]) * hx[k, i]

    # margin_figs = []
    # for i in range(m):
    #     fig_mi = plt.figure()
    #     plt.plot(margin[:, i], "o-", label=f"Margin $i={i+1}$")
    #     plt.axhline(0, color="r", linestyle="--", label="Safety Threshold")
    #     plt.xlabel("Iteration $k$")
    #     plt.ylabel(fr"$h_{i+1}(x_{{k+1}}) \;-\;(1-\alpha_k)\,h_{i+1}(x_k)$")
    #     plt.title(f"Obstacle {i+1}: Safety Margin Over Time")
    #     plt.legend(loc="lower left")
    #     plt.grid()
    #     plt.tight_layout()
    #     margin_figs.append((fig_mi, f"margin_obstacle_{i+1}.png"))

    N = hx.shape[0]
    iters = np.arange(N)
    cmap = cm.get_cmap("nipy_spectral", N)
    norm = Normalize(vmin=0, vmax=N - 1)

    hx_col_figs = []
    for i in range(m):
        fig_hi_col = plt.figure()
        plt.scatter(iters, hx[:, i],
                    c=iters, cmap=cmap, norm=norm, s=20)
        plt.xlabel("Iteration $k$")
        plt.ylabel(f"$h_{i+1}(x_k)$")
        plt.title(f"Obstacle {i+1}: $h_{i+1}(x_k)$ Colored by Iteration")
        plt.colorbar(label="Iteration $k$")
        plt.grid()
        plt.tight_layout()
        hx_col_figs.append((fig_hi_col, f"hx_colored_obstacle_{i+1}.png"))


    # Save Figures

    figs_to_save = [
        (fig_states,    "states_trajectory.png"),
        (fig_actions,   "actions.png"),
        (fig_stagecost, "stagecost.png"),
        (fig_alpha,     "alpha.png"),
        (fig_velocity,  "velocity.png"),
    ]
    # add each h_i plot:
    figs_to_save += hx_figs
    # add each margin_i plot:
    # figs_to_save += margin_figs
    # add each colored‐by‐iteration h_i plot:
    figs_to_save += hx_col_figs

    save_figures(figs_to_save, experiment_folder)
    plt.show()
    plt.close("all")  # Close all figures to free memory


    # ─── 6) Final summary ───

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {stage_cost.sum():.3f}")

    return stage_cost.sum()

def MPC_func_random(x, mpc, params, solver_inst, rand_noise):
        
        # bounds
        X_lower_bound = -CONSTRAINTS_X *np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(mpc.nn.obst.obstacle_num * (mpc.horizon))
        cbf_const_ubg = np.zeros(mpc.nn.obst.obstacle_num * (mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound, np.zeros(mpc.nn.obst.obstacle_num *mpc.horizon)])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound, np.inf*np.ones(mpc.nn.obst.obstacle_num *mpc.horizon)])

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


        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["nn_params"], rand_noise),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        fwd_func = mpc.nn.numerical_forward()
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x)]
        alpha.append(cs.DM(fwd_func(x, h_func_list,  params["nn_params"])))
        

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"], alpha

def run_simulation_randomMPC(params, env, experiment_folder, episode_duration, layers_list, noise_scalingfactor, 
                             noise_variance, horizon, positions, radii):

    env = env()


    np_random = np.random.default_rng(seed=SEED)
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    alphas = []
    mpc = MPC(layers_list, horizon, positions, radii)

    solver_inst = mpc.MPC_solver_rand() 

    for i in range(episode_duration):
        rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        action, _, alpha = MPC_func_random(state, mpc, params, solver_inst, rand_noise=rand_noise)

        # if i<(0.65*2000):
        # else:f
        #     action, _ = MPC_func(state, mpc, params)
        # action, _ = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)
        alphas.append(alpha)

        stage_cost.append(stage_cost_func(action, state))

        print(i)

        # if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
        #     break

    
    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    stage_cost = stage_cost.reshape(-1) 
    alphas = np.array(alphas)
    alphas = np.squeeze(alphas)


    figstates=plt.figure()
    plt.plot(
        states[:, 0], states[:, 1],
        "o-"
    )

    # Plot the obstacle
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
        plt.gca().add_patch(circle)
    plt.xlim([-CONSTRAINTS_X, 0])
    plt.ylim([-CONSTRAINTS_X, 0])

    # Set labels and title
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid()

    figactions=plt.figure()
    plt.plot(actions[:, 0], "o-", label="Action 1")
    plt.plot(actions[:, 1], "o-", label="Action 2")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Action")
    plt.title("Actions")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()

    figstagecost=plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Iteration $k$")
    plt.ylabel("Cost")
    plt.title("Stage Cost")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    m = mpc.nn.obst.obstacle_num
    print(f"shape of alphas: {alphas.shape}")
    
    figsalpha=plt.figure()
    
    if m == 1:
        plt.plot(alphas, "o-", label="$\\alpha(x_k)$")
    else:
        for i in range(m):
            plt.plot(alphas[:, i], "o-", label=f"$\\alpha_{{{i+1}}}(x_k)$")
    plt.xlabel("Iteration $k$")
    plt.ylabel("$alpha$ Value")
    plt.title("$alpha$")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Iteration $k$")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.show()

    figs = [
                (figstates, "states_MPCnoise"),
                (figactions, "actions_MPCnoise"),
                (figstagecost, "stagecost_MPCrandom"),
                (figsalpha, "alpha_MPCnoise"),
                (figsvelocity, "velocity_MPCnoise")
            ]

    save_figures(figs,  experiment_folder)

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

def generate_experiment_notes(experiment_folder, params, params_innit, episode_duration, num_episodes, seed, alpha, sampling_time, gamma, decay_rate, decay_at_end, 
                              noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq,
                              patience_threshold, lr_decay_factor, horizon):
    # used to save the parameters automatically

    notes = f"""
    Experiment Settings:
    --------------------
    Episode Duration: {episode_duration}
    Number of Episodes: {num_episodes}
    Sampling time: {sampling_time}
    Layers List: {layers_list}
    Patience Threshold: {patience_threshold}
    Learing Rate Decay Factor: {lr_decay_factor}

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
    horizon: {horizon}

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
    Initialized params: {params_innit['nn_params']}
    Learned NN params: {params['nn_params']}



    Additional Notes:
    -----------------
    - Off-policy training with initial parameters
    - Noise scaling based on distance to target
    - Decay rate applied to noise over iterations
    - scaling adjused

    """
    save_notes(experiment_folder, notes)
