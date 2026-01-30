
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr
from OPTDclasses import  MPC
from matplotlib.colors import Normalize
import matplotlib.cm as cm




def noise_scale_by_distance(x, y, max_radius=0.5):
    dist = np.sqrt(x**2 + y**2)
    if dist >= max_radius:
        return 1.0
    else:
        return (dist / max_radius)

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
                

def MPC_func(x, mpc, params):

        solver_inst = mpc.MPC_solver() 
        
        # bounds
        X_lower_bound = -5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound,  np.array([0]), np.array([1e-6])])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound,  np.array([np.inf]), np.array([1])])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])  
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        #params of MPC
        P = params["P"]
        Q = params["Q"]
        R = params["R"]
        Pw = params["Pw"]
        omega0 = params["omega0"]
        V = params["V0"]

    
        #flatten
        A_flat = cs.reshape(params["A"] , -1, 1)
        B_flat = cs.reshape(params["B"] , -1, 1)
        P_diag = cs.diag(P) #cs.reshape(P , -1, 1)
        Q_flat = cs.reshape(Q , -1, 1)
        R_flat = cs.reshape(R , -1, 1)

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, Pw, omega0, params["w_pen"]),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )

        g_resid = solution["g"][4]
        print(f"g reid: {g_resid}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        # print(f"omega parameter: {solution['x'][-1]}")
        # print(f"the whole solution: {solution['x']}")
        omega = solution['x'][-1]

        return u_opt, solution["f"], omega, g_resid

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
    mpc = MPC(0.2)


   
    state, _ = env.reset(seed=69, options={})
    states = [state]
    actions = []
    stage_cost = []
    omegas = []
    hx = [mpc.h_func(cs.DM(state))]
    g_resid_lst = []    

    for i in range(episode_duration):
        action, _, omega, g_resid = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)
        omegas.append(omega)
        g_resid_lst.append(-g_resid)

        stage_cost.append(stage_cost_func(action, state))
        hx.append(mpc.h_func(cs.DM(state)))

        print(i)

        if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
            break

    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    hx = np.array(hx)
    g_resid_lst = np.array(g_resid_lst)

    stage_cost = stage_cost.reshape(-1) 
    omegas = np.array(omegas)
    omegas = omegas.reshape(-1) 
    hx = hx.reshape(-1)
    g_resid = g_resid_lst.reshape(-1)

    figstates=plt.figure()
    plt.plot(
        states[:, 0], states[:, 1],
        "o-", label=f"Trajectory (Pw={params['Pw']}, ω₀={params['omega0']})"
    )

    # Plot the obstacle
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])

    # Set labels and title
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title("Trajectories for different combinations of $P_w$ and $\omega_0$")
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
    

    figsomega=plt.figure()
    plt.plot(omegas, "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("$omega$ Value")
    plt.title("$omega$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    ##plt.show()


    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    ##plt.show()

    figshx =plt.figure()
    plt.plot(hx, "o-", label="$h(x_k)$")
    plt.xlabel("Iteration")
    plt.ylabel("$h(x_k)$ Value")
    plt.title("$h(x_k)$ Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    figshxmarg =plt.figure()
    margin = hx[1:] - (np.ones(omegas.shape[0])- omegas) * hx[:-1]
    plt.plot(margin, marker='o', linestyle='-')
    plt.axhline(0, color='r', linestyle='--', label='safety threshold')
    plt.xlabel(r'step $k$')
    plt.ylabel(r'$h(x_{k+1}) - (1-\alpha) \cdot h(x_k)$')
    plt.title('CBF Safety Margin over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figshxmargact =plt.figure()
    plt.plot(g_resid, marker='o', linestyle='-', label = "$g(x_k)$")
    plt.plot(margin, marker='o', linestyle='-', label = "$margin oth$")
    plt.axhline(0, color='r', linestyle='--', label='safety threshold')
    plt.xlabel(r'step $k$')
    plt.ylabel(r'$h(x_{k+1}) - (1-\alpha) \cdot h(x_k)$')
    plt.title('CBF Safety Margin over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    figs = []

    # colormap and norm
    N = len(hx)
    iters = np.arange(N)
    #cmap = cm.get_cmap('viridis')
    cmap = cm.get_cmap('nipy_spectral', N)  
    norm = Normalize(vmin=0, vmax=N-1)

    fig8 = plt.figure()
    plt.plot(states[:,0], states[:,1], color='gray', alpha=0.5)
    sc1 = plt.scatter(states[:,0], states[:,1], c=iters, cmap=cmap, norm=norm, s=40)
    plt.colorbar(sc1, label='Iteration $k$')
    circle = plt.Circle((-2, -2.25), 1.5, color='k', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5,0]); plt.ylim([-5,0])
    plt.xlabel('$x$ (m)'); plt.ylabel('$y$ (m)')
    plt.title('Trajectory Colored by Iteration')
    plt.axis('equal'); plt.grid(); plt.tight_layout()
    figs.append((fig8, f"states_colored_MPCregular_{'afterupdates' if after_updates else 'beforeupdates'}"))

    # Omegas colored
    fig9 = plt.figure()
    plt.plot(omegas, color='gray', alpha=0.5)
    sc2 = plt.scatter(iters[:-1], omegas, c=iters[:-1], cmap=cmap, norm=norm, s=40)
    plt.colorbar(sc2, label='Iteration $k$')
    plt.xlabel('Time (s)'); plt.ylabel('$\omega$ Value')
    plt.title('Omega Colored by Iteration'); plt.grid(); plt.tight_layout()
    figs.append((fig9, f"omega_colored_MPCregular_{'afterupdates' if after_updates else 'beforeupdates'}"))

    # h(x) colored
    fig10 = plt.figure()
    plt.plot(hx, color='gray', alpha=0.5)
    sc3 = plt.scatter(iters, hx, c=iters, cmap=cmap, norm=norm, s=40)
    plt.colorbar(sc3, label='Iteration $k$')
    plt.xlabel('Iteration'); plt.ylabel('$h(x_k)$')
    plt.title('h(x_k) Colored by Iteration'); plt.grid(); plt.tight_layout()
    figs.append((fig10, f"hx_colored_MPCregular_{'afterupdates' if after_updates else 'beforeupdates'}"))

    # Margin colored
    fig11 = plt.figure()
    plt.plot(margin, color='gray', alpha=0.5)
    sc4 = plt.scatter(iters[:-1], margin, c=iters[:-1], cmap=cmap, norm=norm, s=40)
    plt.colorbar(sc4, label='Iteration $k$')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('step $k$'); plt.ylabel(r'$h(x_{k+1}) - (1-\alpha)\,h(x_k)$')
    plt.title('Margin Colored by Iteration'); plt.grid(); plt.tight_layout()
    figs.append((fig11, f"marghx_colored_MPCregular_{'afterupdates' if after_updates else 'beforeupdates'}"))

    plt.show()
    
    if after_updates == False:
        figs = [
                    (figstates, "states_MPCregular_beforeupdates"),
                    (figactions, "actions_MPCregular_beforeupdates"),
                    (figstagecost, "stagecost_MPCregular_beforeupdates"),
                    (figsomega, "omega_MPCregular_beforeupdates"),
                    (figsvelocity, "velocity_MPCregular_beforeupdates"),
                    (figshx, "hx_MPCregular_beforeupdates"),
                    (figshxmarg, "marghx_MPCregular_beforeupdates")
                ]
    else:
         figs = [
                    (figstates, "states_MPCregular_afterupdates"),
                    (figactions, "actions_MPCregular_afterupdates"),
                    (figstagecost, "stagecost_MPCregular_afterupdates"),
                    (figsomega, "omega_MPCregular_afterupdates"),
                    (figsvelocity, "velocity_MPCregular_afterupdates"),
                    (figshx, "hx_MPCregular_afterupdates"),
                    (figshxmarg, "marghx_MPCregular_afterupdates")
                ]

    save_figures(figs,  experiment_folder)
    plt.close('all')

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

    return sum(stage_cost)

def MPC_func_random(x, mpc, params, solver_inst, rand_noise):
        dt = 0.2
        
        # bounds
        X_lower_bound = -5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound,  np.array([0]), np.array([1e-6])])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound,  np.array([np.inf]), np.array([1])])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])  
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        #params of MPC
        P = params["P"]
        Q = params["Q"]
        R = params["R"]
        Pw = params["Pw"]
        omega0 = params["omega0"]
        V = params["V0"]

    
        #flatten
        A_flat = cs.reshape(params["A"] , -1, 1)
        B_flat = cs.reshape(params["B"] , -1, 1)
        P_diag = cs.diag(P) #cs.reshape(P , -1, 1)
        Q_flat = cs.reshape(Q , -1, 1)
        R_flat = cs.reshape(R , -1, 1)


        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, Pw, omega0, rand_noise, params["w_pen"]),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        print(solution["x"][-1])

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
        #rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))

        # if (np_random.random() < 0.5):
        #         rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        # else:
        #         rand_noise = np.zeros((2,1))

        rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1)) #*noise_scale_by_distance(state[0], state[1])

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
        "o-", label=f"Trajectory (Pw={params['Pw']}, ω₀={params['omega0']})"
    )

    # Plot the obstacle
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])

    # Set labels and title
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title("Trajectories for different combinations of $P_w$ and $\omega_0$")
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
    # #plt.show()

    figstagecost=plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("Stage Cost")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    ##plt.show()

    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    ##plt.show()

    figs = [
                (figstates, "states_MPCnoise"),
                (figactions, "actions_MPCnoise"),
                (figstagecost, "stagecost_MPCrandom"),
                (figsvelocity, "velocity_MPCrandom")
            ]

    save_figures(figs,  experiment_folder)
    plt.close('all')

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

def generate_experiment_notes(experiment_folder, params, params_innit, episode_duration, num_episodes, seed, alpha, sampling_time, 
                              gamma, decay_rate, decay_at_end, noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, 
                              replay_buffer, episode_updatefreq):
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
    ---------------
    Pw: {params_innit['Pw']}
    Omega0: {params_innit['omega0']}
    P Matrix: {params_innit["P"]}
    V : {params_innit['V0']}

    MPC Parameters After Learning:
    ---------------
    Pw: {params['Pw']}
    Omega0: {params['omega0']}
    P Matrix: {params["P"]}
    V : {params['V0']}

    Stage Cost:
    ---------------
    Summed Stage Cost of simulation before update: {stage_cost_sum_before}
    Summed Stage Cost of simulation after updates: {stage_cost_sum_after}



    Additional Notes:
    -----------------
    - Off-policy training with initial parameters
    - Noise scaling based on distance to target
    - No time penalty applied
    - I changed alpha_vec
    - Decay rate applied to noise over iterations
    - Trying to decay noise variance too over iterations
    - Added L2 regulization for the QP

    """
    save_notes(experiment_folder, notes)


