
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import matplotlib.pyplot as plt
from Classes import MPC, ObstacleMotion
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.animation as animation
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
                

def MPC_func(x, mpc, params, solver_inst, xpred_list, ypred_list):
        
        fwd_func = mpc.nn.numerical_forward()
        
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
        alpha.append(cs.DM(fwd_func(x, h_func_list,  params["nn_params"])))

        # bounds
        X_lower_bound = -CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))


        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))
        
        print(f"state_const_lbg: {state_const_lbg.shape}, state_const_ubg: {state_const_ubg.shape}")

        cbf_const_lbg = -np.inf * np.ones(mpc.nn.obst.obstacle_num * (mpc.horizon))
        cbf_const_ubg = np.zeros(mpc.nn.obst.obstacle_num * (mpc.horizon))
        
        print(f"cbf_const_lbg: {cbf_const_lbg.shape}, cbf_const_ubg: {cbf_const_ubg.shape}")

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

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["nn_params"], xpred_list, ypred_list),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg= lbg,
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
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("system + Moving Obstacles")

    # fixed zoom
    span = constraints_x
    ax.set_xlim(-1.1*span, +1.1*span)
    ax.set_ylim(-1.1*span, +1.1*span)

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



def run_simulation(params, env, experiment_folder, episode_duration, layers_list, after_updates, horizon, positions, radii, modes, mode_params):
    """
    USE the after_updates flag to determine if the simulation is run after the updates or not!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    env = env()
    mpc = MPC(layers_list, horizon, positions, radii)
    obst_motion = ObstacleMotion(positions, modes, mode_params)

   
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    g_resid_lst = []    
    
    # extract list of h functions
    h_func_list = mpc.nn.obst.make_h_functions()

    alphas = []
    
    # xpred_list = np.zeros((mpc.nn.obst.obstacle_num, 1))
    # ypred_list = np.zeros((mpc.nn.obst.obstacle_num, 1))
    xpred_list, ypred_list = obst_motion.predict_states(horizon)
    
    print(f"xpred_list: {xpred_list.shape}, ypred_list: {ypred_list.shape}")
    #cycle through to plot different h functions later
    hx = [ np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]) ]
    # hx = []

    solver_inst = mpc.MPC_solver_noslack() 
    
    #for plotting the moving obstacle
    obs_positions = [obst_motion.current_positions()]

    for i in range(episode_duration):


        action, _, alpha, g_resid = MPC_func(state, mpc, params, solver_inst, xpred_list, ypred_list)

        alphas.append(alpha)


        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        print(f"state from env: {state}")
        states.append(state)
        actions.append(action)
        g_resid_lst.append(-g_resid)

        hx.append(np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]))

        stage_cost.append(stage_cost_func(action, state))

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

    stage_cost = stage_cost.reshape(-1) 
    
    obs_positions = np.array(obs_positions)   # shape (T, m, 2)
    out_gif = os.path.join(experiment_folder, "system_and_obstacle.gif")
    make_system_obstacle_animation(
    states,
    obs_positions,
    radii,
    CONSTRAINTS_X,
    out_gif,
)
    # T = len(states)
    # system_xy = states[:, :2]

    # fig_anim, ax = plt.subplots()
    # ax.set_autoscale_on(False)
    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$y$")
    # ax.set_title("system + Moving Obstacle")
    # ax.grid(True)
    # ax.axis("equal")

    # # system path line + current position dot
    # line, = ax.plot([], [], "o-", lw=2, label="system path")
    # dot,  = ax.plot([], [], "ro", ms=6,    label="system")

    # # obstacle circle
    # circle = plt.Circle((0,0), radii[0], fill=False, color="k", lw=2, label="obstacle")
    # ax.add_patch(circle)
    # ax.legend(loc="upper right")

    # def init():
    #     ax.set_xlim(-1.5*CONSTRAINTS_X, CONSTRAINTS_X)
    #     ax.set_ylim(-1.5*CONSTRAINTS_X, CONSTRAINTS_X)
    #     line.set_data([], [])
    #     dot.set_data([], [])
    #     circle.center = (obs_positions[0,0,0], obs_positions[0,0,1])
    #     return line, dot, circle

    # def update(k):
    #     # system
    #     xk, yk = system_xy[k]
    #     line.set_data(system_xy[:k+1,0], system_xy[:k+1,1])
    #     dot.set_data(xk, yk)
    #     # obstacle
    #     cx, cy = obs_positions[k,0]
    #     circle.center = (cx, cy)
    #     return line, dot, circle

    # ani = animation.FuncAnimation(
    #     fig_anim, update, frames=T, init_func=init,
    #     blit=True, interval=100
    # )
    # # ani.save(out_path, fps=10, dpi=150)
    # ani.save(os.path.join(experiment_folder, "system_and_obstacle.gif"), fps=10, dpi=150)
    # plt.show()
    # plt.close(fig_anim)


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
    #plt.show()
    plt.close("all")  # Close all figures to free memory

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {stage_cost.sum():.3f}")

    return stage_cost.sum()

def MPC_func_random(x, mpc, params, solver_inst, rand_noise,  xpred_list, ypred_list):
        
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


        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, params["nn_params"], rand_noise, xpred_list, ypred_list),
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
        alpha.append(cs.DM(fwd_func(x, h_func_list,  params["nn_params"])))
        

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"], alpha

def run_simulation_randomMPC(params, env, experiment_folder, episode_duration, layers_list, noise_scalingfactor, 
                             noise_variance, horizon, positions, radii, modes, mode_params):

    env = env()
    obst_motion = ObstacleMotion(positions, modes, mode_params)

    np_random = np.random.default_rng(seed=SEED)
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    alphas = []
    mpc = MPC(layers_list, horizon, positions, radii)
    
    xpred_list, ypred_list = obst_motion.predict_states(horizon)

    solver_inst = mpc.MPC_solver_rand()
    
    obs_positions = [obst_motion.current_positions()] 

    for i in range(episode_duration):
        rand_noise = noise_scalingfactor*np_random.normal(loc=0, scale=noise_variance, size = (2,1))
        action, _, alpha = MPC_func_random(state, mpc, params, solver_inst, rand_noise, xpred_list, ypred_list)

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
    CONSTRAINTS_X,
    out_gif,
)

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
                              patience_threshold, lr_decay_factor, horizon, modes, mode_params, positions, radii):
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
