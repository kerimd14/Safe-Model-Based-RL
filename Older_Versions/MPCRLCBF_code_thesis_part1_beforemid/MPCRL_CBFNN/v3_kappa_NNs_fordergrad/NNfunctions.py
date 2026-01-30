
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import matplotlib.pyplot as plt
from NNclasses import MPC
from matplotlib.colors import Normalize
import matplotlib.cm as cm



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
                

def MPC_func(x, mpc, params, solver_inst,  x_prev, lam_x_prev, lam_g_prev):
        
        fwd_func = mpc.nn.numerical_forw_kappa_fn()

        hx = mpc.h_func(cs.DM(x)) #h(x_k)

        alpha = float(fwd_func(x, hx, params["nn_params"]))
        
        print(f"alpha from nn: {alpha}")
        

        # bounds
        X_lower_bound = -5 * np.array([1, 1, 1, 1])#-1e6 * 5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.array([1, 1, 1, 1])#1e6 * 5 * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

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
            x0    = x_prev,
            lam_x0 = lam_x_prev,
            lam_g0 = lam_g_prev,                   
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )

        print(f"solution: {solution['f']}")

        g_resid = solution["g"][4]        # vector of all g(x)

        print(f"g reid: {g_resid}")

        print(f"MPC lagrange mult g: {solution['lam_g']}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]
 
        return u_opt, solution["f"], alpha, g_resid, x_prev, lam_x_prev, lam_g_prev

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



def run_simulation(params, env, experiment_folder, episode_duration, layers_list, after_updates):

    env = env(sampling_time=0.2)
    mpc = MPC(0.2, layers_list)

   
    state, _ = env.reset(seed=69, options={})
    states = [state]
    actions = []
    stage_cost = []
    g_resid_lst = []  

    alphas = []
    hx = [float(mpc.h_func(cs.DM(state)))]
    # hx = []

    solver_inst = mpc.MPC_solver_noslack() 

    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()

    for i in range(episode_duration):


        action, _, alpha, g_resid, x_prev, lam_x_prev, lam_g_prev = MPC_func(state, mpc, params, solver_inst, 
                                                                             x_prev, lam_x_prev, lam_g_prev)

        alphas.append(alpha)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        print(f"state from env: {state}")
        states.append(state)
        actions.append(action)
        g_resid_lst.append(-g_resid)

        hx.append(float(mpc.h_func((state))))

        stage_cost.append(stage_cost_func(action, state))

        print(i)

        if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
            break

    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    g_resid_lst = np.array(g_resid_lst)
    hx = np.array(hx)
    alphas = np.array(alphas)

    stage_cost = stage_cost.reshape(-1) 
    hx = hx.reshape(-1)
    g_resid = g_resid_lst.reshape(-1)
    alphas = alphas.reshape(-1) 

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
    plt.xlabel("Iteration")
    plt.ylabel("Action")
    plt.title("Actions")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    figstagecost=plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Stage Cost")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    figsalpha=plt.figure()
    plt.plot(alphas, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("$alpha$ Value")
    plt.title("$alpha$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
  
    figsvelocity=plt.figure()
    plt.plot(states[:, 2], "o-", label="Velocity x")
    plt.plot(states[:, 3], "o-", label="Velocity y")    
    plt.xlabel("Iteration")
    plt.ylabel("Velocity Value")
    plt.title("Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.show()

    figshx =plt.figure()
    plt.plot(hx, "o-", label="$h(x_k)$")
    plt.xlabel("Iteration")
    plt.ylabel("$h(x_k)$ Value")
    plt.title("$h(x_k)$ Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    figshxmarg =plt.figure()
    margin = hx[1:] - hx[:-1] + alphas
    plt.plot(margin, marker='o', linestyle='-', label = 'wrong margin')
    plt.plot(g_resid, marker='o', linestyle='-', label = 'correct margin')
    plt.axhline(0, color='r', linestyle='--', label='safety threshold')
    plt.xlabel(r'step $k$')
    plt.ylabel(r'$h(x_{k+1}) - (1-\alpha) \cdot h(x_k)$')
    plt.title('CBF Safety Margin over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figshxmargact =plt.figure()
    plt.plot(g_resid, marker='o', linestyle='-')
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
    plt.plot(alphas, color='gray', alpha=0.5)
    sc2 = plt.scatter(iters[:-1], alphas, c=iters[:-1], cmap=cmap, norm=norm, s=40)
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

    plt.show()
    
    if after_updates == False:
        figs = [
                    (figstates, "states_MPCregular_beforeupdates"),
                    (figactions, "actions_MPCregular_beforeupdates"),
                    (figstagecost, "stagecost_MPCregular_beforeupdates"),
                    (figsvelocity, "velocity_MPCregular_beforeupdates"),
                    (figshx, "hx_MPCregular_beforeupdates"),
                ]
    else:
         figs = [
                    (figstates, "states_MPCregular_afterupdates"),
                    (figactions, "actions_MPCregular_afterupdates"),
                    (figstagecost, "stagecost_MPCregular_afterupdates"),
                    (figsvelocity, "velocity_MPCregular_afterupdates"),
                    (figshx, "hx_MPCregular_afterupdates"),
                ]

    save_figures(figs,  experiment_folder)

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {sum(stage_cost)}")

    return sum(stage_cost)

def MPC_func_random(x, mpc, params, solver_inst, rand_noise,  x_prev, lam_x_prev, lam_g_prev):
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
        


        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["nn_params"], rand_noise),
            x0    = x_prev,
            lam_x0 = lam_x_prev,
            lam_g0 = lam_g_prev,
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"], x_prev, lam_x_prev, lam_g_prev

def run_simulation_randomMPC(params, env, experiment_folder, episode_duration, layers_list, noise_scalingfactor, noise_variance):

    env = env(sampling_time=0.2)


    np_random = np.random.default_rng(seed=69)
    state, _ = env.reset(seed=69, options={})
    states = [state]
    actions = []
    stage_cost = []
    mpc = MPC(0.2, layers_list)

    solver_inst = mpc.MPC_solver_rand() 
    
    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()

    for i in range(episode_duration):
        rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        action, _, x_prev, lam_x_prev, lam_g_prev = MPC_func_random(state, mpc, params, solver_inst, rand_noise=rand_noise,
                                                                    x_prev=x_prev, lam_x_prev=lam_x_prev, lam_g_prev=lam_g_prev)

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
    # plt.show()

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
    #plt.show()

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
                              noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq,
                              patience_threshold, lr_decay_factor):
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


def build_raw_nn_numeric(nn):
    """Return f_raw(x,h,flat) = NN([x;h]) using current nn.forward."""
    x = cs.MX.sym('x', 4, 1)
    h = cs.MX.sym('h', 1, 1)
    y = nn.forward(cs.vertcat(x, h))
    return cs.Function('rawNN', [x, h, nn.get_flat_parameters()], [y])

def plot_kappa_vs_h_for_states(
    nn,
    states_list,
    flat_params=None,
    h_range=(-20.0, 20.0),
    n_points=300,
    show_raw=False,
    title="κ(x,h) vs h for multiple fixed states",
):
    """
    Sweeps h over a range, for multiple fixed x, and plots:
      κ(x,h) = NN([x;h]) - NN([x;0])
    Optionally also plots raw NN([x;h]).

    Args:
      nn            : your NN instance
      states_list   : iterable of states; each x is (4,) or (4,1)
      flat_params   : cs.DM flat params; if None uses nn.initialize_parameters()
      h_range       : (h_min, h_max)
      n_points      : number of samples in the sweep
      show_raw      : if True, also plot raw NN(x,h) curves faintly
      title         : figure title
    """
    # params
    if flat_params is None:
        flat_params, _, _ = nn.initialize_parameters()

    # casadi functions
    kappa_fn = nn.numerical_forw_kappa_fn()   # kappa(x,h,flat)
    raw_fn   = build_raw_nn_numeric(nn)      # rawNN(x,h,flat)

    # sweep grid
    h_vals = np.linspace(h_range[0], h_range[1], n_points)

    # plotting setup
    plt.figure(figsize=(7.5, 5.0))
    cmap = plt.get_cmap('tab10')

    # monotonicity summary
    print("Monotonicity check (finite differences of κ wrt h):")
    for idx, x in enumerate(states_list):
        x_dm = cs.DM(x).reshape((4,1))
        kappa_curve = []
        raw_curve   = []

        # evaluate across h
        for h_val in h_vals:
            # κ(x,h)
            kv = float(kappa_fn(x_dm, cs.DM([[h_val]]), flat_params))
            kappa_curve.append(kv)
            if show_raw:
                rv = float(raw_fn(x_dm, cs.DM([[h_val]]), flat_params))
                raw_curve.append(rv)

        kappa_curve = np.asarray(kappa_curve)

        # finite-diff slope
        diffs = np.diff(kappa_curve) / np.diff(h_vals)
        num_neg = np.sum(diffs < -1e-8)
        min_slope = np.min(diffs)
        print(f"  x[{idx}]={np.array(x).ravel()} -> neg slopes: {int(num_neg)}, min slope: {min_slope:.3e}")

        # plot κ
        color = cmap(idx % 10)
        plt.plot(h_vals, kappa_curve, label=f'κ(x[{idx}],h)', color=color, linewidth=2)

        # optional raw NN
        if show_raw:
            plt.plot(h_vals, raw_curve, linestyle='--', color=color, alpha=0.35, label=f'NN(x[{idx}],h)')

        # mark the anchor at h=0 (κ should be 0 there)
        # find index closest to h=0
        j0 = np.argmin(np.abs(h_vals))
        plt.scatter([h_vals[j0]], [kappa_curve[j0]], color=color, marker='o', s=25, zorder=3)

    plt.axhline(0.0, linestyle=':', linewidth=1)
    plt.xlabel('h')
    plt.ylabel('κ(x,h)  (and NN(x,h) if shown)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_kappa_with_linear_baseline(
#     nn,
#     states_list,
#     flat_params=None,
#     h_range=(-1.0, 2.0),
#     n_points=400,
#     show_raw=False,
#     linear_slope=None,     # if None, auto-fit slope near h=0 for each x
#     fit_window=0.2,        # half-width around 0 for slope fit
#     title="K-function: learned κ vs linear baseline"
# ):
#     """
#     For each fixed x in states_list:
#       - sweep h, compute κ(x,h) = NN([x;h]) - NN([x;0])
#       - optionally compute raw NN(x,h)
#       - plot κ and linear baseline c*h (c fixed or estimated by LS near h=0)
#     """
#     if flat_params is None:
#         flat_params, _, _ = nn.initialize_parameters()

#     kappa_fn = nn.numerical_forw_kappa_fn()
#     raw_fn   = build_raw_nn_numeric(nn)

#     h_vals = np.linspace(h_range[0], h_range[1], n_points)
#     j0 = np.argmin(np.abs(h_vals))  # index closest to 0

#     plt.figure(figsize=(6.5, 6))
#     cmap = plt.get_cmap('tab10')

#     for i, x0 in enumerate(states_list):
#         x_dm = cs.DM(x0).reshape((4,1))
#         kappa_curve = np.array([
#             float(kappa_fn(x_dm, cs.DM([[hv]]), flat_params)) for hv in h_vals
#         ])

#         # pick slope c
#         if linear_slope is None:
#             # fit y = c*h around h=0
#             mask = np.abs(h_vals) <= fit_window
#             h_fit = h_vals[mask].reshape(-1,1)
#             y_fit = kappa_curve[mask].reshape(-1,1)
#             # least squares: c = (h^T h)^{-1} h^T y
#             denom = float(h_fit.T @ h_fit)
#             c = float((h_fit.T @ y_fit)/denom) if denom > 1e-12 else 1.0
#         else:
#             c = linear_slope

#         kappa_lin = c * h_vals

#         # plot learned κ
#         color = cmap(i % 10)
#         plt.plot(h_vals, kappa_curve, color=color, linewidth=2, label=f"learned κ (x[{i}])")
#         # (optional) raw NN
#         if show_raw:
#             raw_curve = np.array([
#                 float(raw_fn(x_dm, cs.DM([[hv]]), flat_params)) for hv in h_vals
#             ])
#             plt.plot(h_vals, raw_curve, "--", color=color, alpha=0.35, label=f"raw NN (x[{i}])")
#         # baseline
#         plt.plot(h_vals, kappa_lin, color=color, alpha=0.8, linestyle=":", linewidth=2,
#                  label=f"linear κ (c={c:.2f})")

#         # mark κ(x,0)
#         plt.scatter([h_vals[j0]], [kappa_curve[j0]], s=25, color=color, zorder=3)

#         # print slope near 0
#         if 1 < j0 < len(h_vals)-2:
#             slope_num = (kappa_curve[j0+1]-kappa_curve[j0-1])/(h_vals[j0+1]-h_vals[j0-1])
#             print(f"x[{i}]={np.array(x0)}  κ(x,0)={kappa_curve[j0]:.3e}   dκ/dh|0≈{slope_num:.3f}   baseline c={c:.3f}")

#     plt.axhline(0, color='k', lw=0.8, ls='-')
#     plt.axvline(0, color='k', lw=0.8, ls='-')
#     plt.xlabel("h")
#     plt.ylabel("kappa")
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()