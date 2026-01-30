
import numpy as np
import os # to communicate with the operating system
import optuna
import copy
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
from Classes import MPC, ObstacleMotion, NN, env, RLclass
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.animation as animation
from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U


def stage_cost_func(action, x, S, slack_penalty):
            """Computes the stage cost :math: L(s,a).
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])

            state = x
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + np.sum(slack_penalty * S)  # slack penalty
            )
                

def MPC_func(x, mpc, params, solver_inst, xpred_list, ypred_list, x_prev, lam_x_prev, lam_g_prev):
        

        # bounds
        # X_lower_bound = -CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        # X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))

        X_lower_bound = -np.tile(CONSTRAINTS_X, mpc.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, mpc.horizon)

        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))
        
        print(f"state_const_lbg: {state_const_lbg.shape}, state_const_ubg: {state_const_ubg.shape}")

        cbf_const_lbg = -np.inf * np.ones(mpc.nn.obst.obstacle_num * (mpc.horizon))
        cbf_const_ubg = np.zeros(mpc.nn.obst.obstacle_num * (mpc.horizon))
        
        print(f"cbf_const_lbg: {cbf_const_lbg.shape}, cbf_const_ubg: {cbf_const_ubg.shape}")

        # lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound])  
        # ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound])
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

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, 
                                              P_diag, Q_flat, R_flat, params["nn_params"], 
                                              xpred_list, ypred_list),
            x0    = x_prev,
            lam_x0 = lam_x_prev,
            lam_g0 = lam_g_prev,
            ubx = ubx,  
            lbx = lbx,
            ubg = ubg,
            lbg = lbg,
        )

        g_resid = solution["g"][mpc.ns*mpc.horizon:]        # vector of all g(x)

        print(f"g reid: {g_resid}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        
        # NN output
        fwd_func = mpc.nn.numerical_forward()
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
        alpha.append(cs.DM(fwd_func(x, h_func_list, xpred_list[:(1)*mpc.nn.obst.obstacle_num], 
                                    ypred_list[(1)*mpc.nn.obst.obstacle_num], params["nn_params"])))
        
        #warm start variables
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]
        
        S = solution["x"][mpc.na * (mpc.horizon) + mpc.ns * (mpc.horizon+1):]

        return u_opt, solution["f"], alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, S

def save_figures(figures, experiment_folder):
        save_choice = True#input("Save the figure? (y/n): ")
        if save_choice == True:#"y":
            os.makedirs(experiment_folder, exist_ok=True) # make directory ( exist_ok makes sure it doenst throw exception when it alreadt exists)
            for fig, filename in figures: 
                file_path = os.path.join(experiment_folder, filename) # add the file to directory
                fig.savefig(file_path)
                plt.close(fig)
                print(f"Figure saved as: {file_path}")
        else:
            print("Figure not saved")

def save_notes(experiment_folder, notes, filename="notes.txt"):
    os.makedirs(experiment_folder, exist_ok=True)
    notes_path = os.path.join(experiment_folder, filename)
    with open(notes_path, "w") as file:
        file.write(notes)
        
# def save_animations(animations, experiment_folder):
#     """
#     animations : list of (matplotlib.animation.Animation, filename)
#     experiment_folder : base directory where to drop them
#     """
#     os.makedirs(experiment_folder, exist_ok=True)
#     for ani, filename in animations:
#         out_path = os.path.join(experiment_folder, filename)
#         ani.save(out_path, writer="pillow", fps=3, dpi=90)
#         print(f"Animation saved as: {out_path}")
        
def make_system_obstacle_animation(
    states: np.ndarray,
    obs_positions: np.ndarray,
    radii: list,
    constraints_x: float,
    out_path: str,
):
    """
    states        : (T,4) array of system [x,y,vx,vy]
    obs_positions : (T, m, 2) array of obstacle centers
    radii         : list of length m
    constraints_x : scalar for plotting window
    out_path      : path to save the .gif
    """
    T, m = obs_positions.shape[:2]
    system_xy = states[:, :2]

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"System + Moving Obstacles")

    # fixed zoom
    span = constraints_x
    ax.set_xlim(-1.1*span, +0.1*span)
    ax.set_ylim(-1.1*span, +0.1*span)

    # system artists
    line, = ax.plot([], [], "o-", lw=2, label="system path")
    dot,  = ax.plot([], [], "ro", ms=6,    label="system")

    # grab a real list of colors from the “tab10” qualitative map
    cmap   = plt.get_cmap("tab10")
    colors = cmap.colors  # this is a tuple-list of length 10
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # one circle per obstacle
    circles = []
    for i, r in enumerate(radii):
        c = plt.Circle(
            (0, 0), r,
            fill=False,
            color=colors[i % len(colors)],
            lw=2,
            label=f"obstacle {i+1}"
        )
        ax.add_patch(c)
        circles.append(c)

    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        for c in circles:
            c.center = (0, 0)
        return [line, dot] + circles

    def update(k):
        # system
        xk, yk = system_xy[k]
        line.set_data(system_xy[:k+1, 0], system_xy[:k+1, 1])
        dot.set_data(xk, yk)
        # obstacles
        for i, c in enumerate(circles):
            cx, cy = obs_positions[k, i]
            c.center = (cx, cy)
        return [line, dot] + circles

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        blit=True, interval=100
    )
    ani.save(out_path, writer="pillow", fps=3, dpi=90)
    plt.close(fig)


def calculate_trajectory_length(states):
    # compute pairwise Euclidean distances and sum everything
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)



def run_simulation(params, env, experiment_folder, episode_duration, 
                   layers_list, after_updates, horizon, positions,
                   radii, modes, mode_params, slack_penalty_eval):
    """
    USE the after_updates flag to determine if the simulation is run after the updates or not!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    env = env()
    mpc = MPC(layers_list, horizon, positions, radii, slack_penalty_eval, mode_params)
    obst_motion = ObstacleMotion(positions, modes, mode_params)

   
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    g_resid_lst = [] 
    lam_g_hist = []   
    
    # extract list of h functions
    h_func_list = mpc.nn.obst.make_h_functions()

    alphas = []
    
    # xpred_list = np.zeros((mpc.nn.obst.obstacle_num, 1))
    # ypred_list = np.zeros((mpc.nn.obst.obstacle_num, 1))
    xpred_list, ypred_list = obst_motion.predict_states(horizon)
    
    print(f"xpred_list: {xpred_list}, ypred_list: {ypred_list}")
    #cycle through to plot different h functions later
    hx = [ np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]) ]
    # hx = []

    solver_inst = mpc.MPC_solver() 
    
    #for plotting the moving obstacle
    obs_positions = [obst_motion.current_positions()]
    
    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()  # initialize warm start variables

    for i in range(episode_duration):

        action, _, alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, S = MPC_func(state, 
                                                                                mpc, 
                                                                                params,
                                                                                solver_inst, 
                                                                                xpred_list, 
                                                                                ypred_list,
                                                                                x_prev, 
                                                                                lam_x_prev, 
                                                                                lam_g_prev)

        alphas.append(alpha)


        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        print(f"state from env: {state}")
        states.append(state)
        actions.append(action)
        g_resid_lst.append(-g_resid)
        
        arr = np.array(lam_g_prev).flatten()
    
        lam_g_hist.append(arr)

        hx.append(np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]))

        stage_cost.append(stage_cost_func(action, state, S, slack_penalty_eval))

        print(i)
        
        #object moves
        _ = obst_motion.step()
        
        xpred_list, ypred_list = obst_motion.predict_states(horizon)
        
        obs_positions.append(obst_motion.current_positions())

        print(f"positons: {obst_motion.current_positions()}")
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
    obs_positions = np.array(obs_positions)
    lam_g_hist = np.vstack(lam_g_hist)

    stage_cost = stage_cost.reshape(-1) 
    
    obs_positions = np.array(obs_positions)   # shape (T, m, 2)
    out_gif = os.path.join(experiment_folder, f"system_and_obstacle_{'after' if after_updates else 'before'}.gif")
    make_system_obstacle_animation(
    states,
    obs_positions,
    radii,
    CONSTRAINTS_X[0],
    out_gif,
)
    
    suffix = 'after' if after_updates else 'before'
    cols = [f"lam_g_{i}" for i in range(lam_g_hist.shape[1])]
    df = pd.DataFrame(lam_g_hist, columns=cols)


    df = df.round(3)


    table_str = df.to_string(index=False)

    txt_path = os.path.join(experiment_folder, f"lam_g_prev_{suffix}.txt")
    with open(txt_path, 'w') as f:
        f.write(table_str)

    fig_states = plt.figure()
    plt.plot(states[:,0], states[:,1], "o-", label=r"trajectory")
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, fill=False, linewidth=2, edgecolor="k")
        plt.gca().add_patch(circle)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"State Trajectory")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    save_figures([(fig_states,
                   f"states_trajectory_{'after' if after_updates else 'before'}.svg")],
                 experiment_folder)


    fig_actions = plt.figure()
    plt.plot(actions[:,0], "o-", label=r"Action 1")
    plt.plot(actions[:,1], "o-", label=r"Action 2")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Action")
    plt.title(r"Actions Over Time")
    plt.grid()
    plt.legend()
    save_figures([(fig_actions,
                   f"actions_{'after' if after_updates else 'before'}.svg")],
                 experiment_folder)

    fig_stagecost = plt.figure()
    plt.plot(stage_cost, "o-")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Stage Cost")
    plt.title(r"Stage Cost Over Time")
    plt.grid()
    save_figures([(fig_stagecost,
                   f"stagecost_{'after' if after_updates else 'before'}.svg")],
                 experiment_folder)

    fig_velocity = plt.figure()
    plt.plot(states[:,2], "o-", label=r"$v_{x}$")
    plt.plot(states[:,3], "o-", label=r"$v_{y}$")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Velocity")
    plt.title(r"Velocities Over Time")
    plt.grid()
    plt.legend()
    save_figures([(fig_velocity,
                   f"velocity_{'after' if after_updates else 'before'}.svg")],
                 experiment_folder)


    
    m = mpc.nn.obst.obstacle_num  # number of obstacles
    print(f"shape of alphas: {alphas.shape}")
    

    fig_alpha = plt.figure()
    if alphas.ndim == 1:
        plt.plot(alphas, "o-", label=r"$\alpha(x_k)$")
    else:
        for i in range(alphas.shape[1]):
            plt.plot(alphas[:,i], "o-", label=rf"$\alpha_{{{i+1}}}(x_k)$")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\alpha_i(x_k)$")
    plt.title(r"Neural-Network Outputs $\alpha_i(x_k)$")
    plt.grid()
    plt.legend(loc="upper right", fontsize="small")
    save_figures([(fig_alpha,
                   f"alpha_{'after' if after_updates else 'before'}.svg")],
                 experiment_folder)

    
    # h(x) plots
    for i in range(hx.shape[1]):
        fig_hi = plt.figure()
        plt.plot(hx[:,i], "o-", label=rf"$h_{{{i+1}}}(x_k)$")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
        plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
        plt.grid()
        save_figures([(fig_hi,
                       f"hx_obstacle_{i+1}_{'after' if after_updates else 'before'}.svg")],
                     experiment_folder)

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

    # Colored-by-iteration h(x)
    hx_colored = []
    N = hx.shape[0]
    cmap = cm.get_cmap("nipy_spectral", N)
    norm = Normalize(vmin=0, vmax=N-1)
    for i in range(hx.shape[1]):
        fig_hi_col = plt.figure()
        plt.scatter(np.arange(N), hx[:,i],
                    c=np.arange(N), cmap=cmap, norm=norm, s=20)
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
        plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Colored by Iteration")
        plt.colorbar(label=r"Iteration $k$")
        plt.grid()
        save_figures([(fig_hi_col,
                       f"hx_colored_obstacle_{i+1}.svg")],
                     experiment_folder)

    print(f"Saved all figures for {'after' if after_updates else 'before'} run.")

    return stage_cost.sum()


def MPC_func_random(x, mpc, params, solver_inst, rand_noise,  xpred_list, ypred_list, x_prev, lam_x_prev, lam_g_prev):
        
        alpha = []
        # bounds
        # X_lower_bound = -CONSTRAINTS_X *np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        # X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))
        
        X_lower_bound = -np.tile(CONSTRAINTS_X, mpc.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, mpc.horizon)
        
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

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat,
                                              params["nn_params"], rand_noise, xpred_list, ypred_list),
            x0    = x_prev,
            lam_x0 = lam_x_prev,
            lam_g0 = lam_g_prev,
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        fwd_func = mpc.nn.numerical_forward()
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
        print(f"h_func_list: {h_func_list}")
        alpha.append(cs.DM(fwd_func(x, h_func_list, xpred_list[:(1)*mpc.nn.obst.obstacle_num], 
                                    ypred_list[(1)*mpc.nn.obst.obstacle_num],  params["nn_params"])))
        

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        
        #warm start variables
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]
        
        S = solution["x"][mpc.na * (mpc.horizon) + mpc.ns * (mpc.horizon+1):]

        return u_opt, solution["f"], alpha, x_prev, lam_x_prev, lam_g_prev, S
    
def noise_scale_by_distance(x, y, max_radius=3):
            # i might remove this because it doesnt allow for exploration of the last states which is important
            
            
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)**2
    

def run_simulation_randomMPC(params, env, experiment_folder, episode_duration, layers_list, noise_scalingfactor, 
                             noise_variance, horizon, positions, radii, modes, mode_params, slack_penalty_eval):

    env = env()
    obst_motion = ObstacleMotion(positions, modes, mode_params)

    np_random = np.random.default_rng(seed=SEED)
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    alphas = []
    mpc = MPC(layers_list, horizon, positions, radii, slack_penalty_eval, mode_params)
    
    xpred_list, ypred_list = obst_motion.predict_states(horizon)
    
    print(f"xpred_list rand: {xpred_list}, ypred_list rand: {ypred_list}")

    solver_inst = mpc.MPC_solver_rand()
    
    obs_positions = [obst_motion.current_positions()] 
    
    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()  # initialize warm start variables

    for i in range(episode_duration):
        rand_noise = noise_scale_by_distance(state[0], state[1])*noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        action, _, alpha,  x_prev, lam_x_prev, lam_g_prev, S = MPC_func_random(state, 
                                                                            mpc, 
                                                                            params, 
                                                                            solver_inst, 
                                                                            rand_noise, 
                                                                            xpred_list, 
                                                                            ypred_list, 
                                                                            x_prev, 
                                                                            lam_x_prev, 
                                                                            lam_g_prev)

        # if i<(0.65*2000):
        # else:f
        #     action, _ = MPC_func(state, mpc, params)
        # action, _ = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)
        alphas.append(alpha)

        stage_cost.append(stage_cost_func(action, state, S, slack_penalty_eval))
        
        #object moves
        _ = obst_motion.step()
        
        xpred_list, ypred_list = obst_motion.predict_states(horizon)
        
        obs_positions.append(obst_motion.current_positions())

        print(i)

        # if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
        #     break

    
    states = np.array(states)
    actions = np.array(actions)
    stage_cost = np.array(stage_cost)
    stage_cost = stage_cost.reshape(-1) 
    alphas = np.array(alphas)
    alphas = np.squeeze(alphas)
    
    
    obs_positions = np.array(obs_positions)   # shape (T, m, 2)
    out_gif = os.path.join(experiment_folder, "system_and_obstacle_random.gif")
    make_system_obstacle_animation(
    states,
    obs_positions,
    radii,
    CONSTRAINTS_X[0],
    out_gif,
)

    # State Trajectory with obstacles
    fig_states = plt.figure()
    plt.plot(states[:, 0], states[:, 1], "o-", label=r"trajectory")
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, fill=False, linewidth=2, edgecolor="k")
        plt.gca().add_patch(circle)
    plt.xlim([-CONSTRAINTS_X[0], 0])
    plt.ylim([-CONSTRAINTS_X[1], 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"Trajectories")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    save_figures([(fig_states, "states_MPCnoise.svg")], experiment_folder)

    # Actions over time
    fig_actions = plt.figure()
    plt.plot(actions[:, 0], "o-", label=r"Action 1")
    plt.plot(actions[:, 1], "o-", label=r"Action 2")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Action")
    plt.title(r"Actions")
    plt.grid()
    plt.legend()
    save_figures([(fig_actions, "actions_MPCnoise.svg")], experiment_folder)

    # Stage Cost
    fig_stagecost = plt.figure()
    plt.plot(stage_cost, "o-", label=r"Stage Cost")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Cost")
    plt.title(r"Stage Cost")
    plt.grid()
    plt.legend()
    save_figures([(fig_stagecost, "stagecost_MPCrandom.svg")], experiment_folder)
    
    # Alpha values from nn
    m = mpc.nn.obst.obstacle_num
    fig_alpha = plt.figure()
    if m == 1:
        plt.plot(alphas, "o-", label=r"$\alpha(x_k)$")
    else:
        for i in range(m):
            plt.plot(alphas[:, i], "o-", label=rf"$\alpha_{{{i+1}}}(x_k)$")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\alpha_i(x_k)$")
    plt.title(r"Neural‐Network Outputs $\alpha_i(x_k)$")
    plt.grid()
    plt.legend(loc="upper right", fontsize="small")
    save_figures([(fig_alpha, "alpha_MPCnoise.svg")], experiment_folder)

    # Velocities
    fig_velocity = plt.figure()
    plt.plot(states[:, 2], "o-", label=r"$v_x$")
    plt.plot(states[:, 3], "o-", label=r"$v_y$")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Velocity")
    plt.title(r"Velocity Plot")
    plt.grid()
    plt.legend()
    save_figures([(fig_velocity, "velocity_MPCnoise.svg")], experiment_folder)

    print(f"Stage Cost: {sum(stage_cost)}")

def generate_experiment_notes(experiment_folder, params, params_innit, episode_duration, num_episodes, seed, alpha, sampling_time, gamma, decay_rate, decay_at_end, 
                              noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq,
                              patience_threshold, lr_decay_factor, horizon, modes, mode_params, positions, radii, slack_penalty):
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
    Modes: {modes}
    Mode Parameters: {mode_params}
    Positions: {positions}
    Radii: {radii}
    Slack Penalty: {slack_penalty}

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
    

def run_experiment(exp_config):
    # ─── unpack everything from config ───────────────────────
    dt                   = SAMPLING_TIME
    seed                 = SEED

    # noise schedule
    initial_noise_scale  = exp_config["initial_noise_scale"]
    noise_variance       = exp_config["noise_variance"]
    decay_at_end         = exp_config["decay_at_end"]

    # compute decay rate
    num_episodes         = exp_config["num_episodes"]
    episode_update_freq  = 10
    decay_rate           = 1 - np.power(
                              decay_at_end,
                              1 / (num_episodes / episode_update_freq)
                          )

    # RL hyper-params
    alpha                = exp_config["alpha"]
    gamma                = 0.95
    patience             = exp_config["patience"]
    lr_decay             = exp_config["lr_decay"]
    slack_penalty        = exp_config["slack_penalty"]
    slack_penalty_eval  = 2e5

    # episode/MPC specs
    episode_duration     = exp_config["episode_duration"]
    mpc_horizon          = 5
    replay_buffer_size   = episode_duration * 10  # buffer holding number of episodes (e.g. hold 10 episodes)

    # experiment folder
    experiment_folder    = exp_config["experiment_folder"]
    
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
    positions = [(-2.0, -1.5), (-3.0, -3.0)]
    radii     = [0.75, 0.75]
    modes     = ["step_bounce", "step_bounce"]
    mode_params = [
        {"bounds": (-4.0,  0.0), "speed": 2.3, "dir":  1},
        {"bounds": (-4.0,  1.0), "speed": 2.0, "dir": -1},
    ]
    

    input_dim = NUM_STATES + len(positions) + 2*len(positions) #x+h(x)+ (obs_positions_x + obs_positions_y)
    hidden_dims = [16, 16]
    output_dim = len(positions)
    layers_list = [input_dim] + hidden_dims + [output_dim]
    print("NN layers:", layers_list)

    nn = NN(layers_list, positions, radii, copy.deepcopy(mode_params))
    params_init["nn_params"], _, _ = nn.initialize_parameters()
    # keep a copy of the original parameters for later logging
    params_before = params_init.copy()
    

    # ─── The Learning  ─────────────────────────────────────
    
    # run simulation of random MPC to see how the system behaves under initial random noise
    run_simulation_randomMPC(
        params_init,
        env,
        experiment_folder,
        episode_duration,
        layers_list,
        initial_noise_scale,
        noise_variance,
        mpc_horizon,
        positions,
        radii,
        modes,
        copy.deepcopy(mode_params),
        slack_penalty,
    )

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
        slack_penalty_eval=slack_penalty,
    )
    # use RL to train the nn CBF with MPC
    
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
        slack_penalty,
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
        slack_penalty_eval=slack_penalty,
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
        slack_penalty,
    )


    return stage_cost_after
BASE_DIR = "optuna_runs_1"
def objective(trial):
    #trial is an instance of optuna.Trial
    os.makedirs(BASE_DIR, exist_ok=True)
    
    
    exp_config = {
        "initial_noise_scale": trial.suggest_int("init_noise", 5, 20),
        "noise_variance":      trial.suggest_int("noise_var", 3, 15),
        "decay_at_end":        trial.suggest_float("decay_end", 0.01, 0.1),
        "alpha":               trial.suggest_loguniform("alpha", 1e-4, 5e-1),
        "patience":            trial.suggest_int("patience", 100, 1_000),
        "lr_decay":            trial.suggest_float("lr_decay", 0.1, 0.9),
        "num_episodes":        trial.suggest_int("num_episodes", 1_000, 3_000),
        "episode_duration":    trial.suggest_int("ep_duration",  150, 300),
        "slack_penalty":       trial.suggest_int("slack_penalty",  5e2, 1e6),
    }
    
    
    folder = os.path.join(BASE_DIR, f"trial_{trial.number}")
    os.makedirs(folder, exist_ok=True)
    exp_config["experiment_folder"] = folder
    
    
    return run_experiment(exp_config)


def save_best_results(study: optuna.Study,
                      base_dir: str = BASE_DIR,
                      filename: str = "best_results.txt"):
    """Dump the best trial’s info into a text file under BASE_DIR."""
    best = study.best_trial
    notes = f"""
Best Optuna Trial
-----------------
Trial Number: {best.number}
Value: {best.value}

Parameters:
{os.linesep.join(f'  • {k}: {v}' for k, v in best.params.items())}
"""
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)
    with open(path, "w") as f:
        f.write(notes.strip() + "\n")
    print(f"[Info] Saved best trial to {path}")