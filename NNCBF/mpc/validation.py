import numpy as np
import os # to communicate with the operating system
import optuna
import copy
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt

from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U

# Refactored-style imports (match RNN refactor layout) while keeping the rest intact
from viz.viz import save_figures
from mpc.mpc import MPC
from obstacles.obstacles_motion import ObstacleMotion
from viz.animation import (
    make_system_obstacle_animation_v2,
    make_system_obstacle_animation_v3,
    make_system_obstacle_svg_frames_v3,
    make_system_obstacle_montage_v1,
)

from matplotlib.colors import Normalize, ListedColormap
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib import rcParams
import matplotlib.image as mpimg
import matplotlib.lines as mlines

from npz_builder import NPZBuilder


def stage_cost_func(action, x, S, slack_penalty):
            """Computes the stage cost :math: L(s,a).
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])

            state = x
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + slack_penalty*np.sum(S)  # slack penalty
            )


def MPC_func(x, mpc, params, solver_inst, xpred_list, ypred_list, x_prev, lam_x_prev, lam_g_prev):


        # bounds
        # X_lower_bound = -CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        # X_upper_bound = CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))

        X_lower_bound = -np.tile(CONSTRAINTS_X, mpc.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, mpc.horizon)

        # keep consistent with constraints
        U_lower_bound = -CONSTRAINTS_U*np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = CONSTRAINTS_U*np.ones(mpc.na * (mpc.horizon))

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

        # keep consistent with the RNN refactor packing style
        Q_flat = cs.diag(Q)  # rather than reshape
        R_flat = cs.diag(R)  # rather than reshape

        solution = solver_inst(
            p=cs.vertcat(
                A_flat, B_flat, params["b"], V,
                P_diag, Q_flat, R_flat,
                params["nn_params"],
                xpred_list, ypred_list
            ),
            x0=x_prev,
            lam_x0=lam_x_prev,
            lam_g0=lam_g_prev,
            ubx=ubx,
            lbx=lbx,
            ubg=ubg,
            lbg=lbg,
        )

        g_resid = solution["g"][mpc.ns*mpc.horizon:]        # vector of all g(x)

        print(f"g reid: {g_resid}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        # NN output
        fwd_func = mpc.nn.numerical_forward()
        alpha = []

        # be explicit about obstacle slicing + consistent DM types
        m = mpc.nn.obst.obstacle_num
        x_dm = cs.DM(x)

        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x_dm, xpred_list[:m], ypred_list[:m])]
        alpha.append(
            cs.DM(
                fwd_func(
                    x_dm,
                    h_func_list,
                    xpred_list[:m],
                    ypred_list[:m],
                    params["nn_params"]
                )
            )
        )

        #warm start variables
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]

        S = solution["x"][mpc.na * (mpc.horizon) + mpc.ns * (mpc.horizon+1):]
        X_plan = cs.reshape(solution["x"][:mpc.ns * (mpc.horizon+1)], mpc.ns, mpc.horizon + 1)
        plan_xy = np.array(X_plan[:2, :]).T

        return u_opt, solution["f"], alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, S, plan_xy


def run_simulation(params, env, experiment_folder, episode_duration,
                   layers_list, after_updates, horizon, positions,
                   radii, modes, mode_params, slack_penalty_eval):
    """
    USE the after_updates flag to determine if the simulation is run after the updates or not!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    slack_penalty_eval = slack_penalty_eval/(horizon+len(positions))  # normalize slack penalty by horizon + number of obstacles
    env = env()
    mpc = MPC(layers_list, horizon, positions, radii, slack_penalty_eval, mode_params, modes)
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


    plans = []

    slacks_eval = []

    m = mpc.nn.obst.obstacle_num
    N = horizon

    def unflatten_slack(S_raw, m, N):
                """
                Turn S_flat (m*N,) or (m*N,1) from CasADi into (m, N).
                CasADi reshape is column-major (Fortran order): columns are horizon j.
                """
                S_raw = np.array(cs.DM(S_raw).full())  # to dense numpy
                # If already 2D (m, N), accept it.
                if S_raw.shape == (m, N):
                    return S_raw
                # If (N, m), transpose.
                if S_raw.shape == (N, m):
                    return S_raw.T
                # If flat of length m*N → reshape column-major
                flat = S_raw.reshape(-1)
                if flat.size == m * N:
                    return flat.reshape(m, N, order="F")
                raise ValueError(f"Unexpected slack shape {S_raw.shape}, cannot make (m={m}, N={N}).")

    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()  # initialize warm start variables

    for i in range(episode_duration):

        action, _, alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, S, plan_xy = MPC_func(state,
                                                                                mpc,
                                                                                params,
                                                                                solver_inst,
                                                                                xpred_list,
                                                                                ypred_list,
                                                                                x_prev,
                                                                                lam_x_prev,
                                                                                lam_g_prev)

        alphas.append(alpha)
        plans.append(plan_xy)
        S_now_mN = unflatten_slack(S, m, N)   # shape (m, N)
        slacks_eval.append(S_now_mN)


        action = cs.fmin(cs.fmax(cs.DM(action), -CONSTRAINTS_U), CONSTRAINTS_U)
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
    plans = np.array(plans)
    slacks_eval = np.stack(slacks_eval, axis=0)

    T, m, N = slacks_eval.shape
    t_eval = np.arange(T)
    for oi in range(m):
                fig_slack_i = plt.figure(figsize=(10, 4))
                for j in range(N):
                    plt.plot(t_eval, slacks_eval[:, oi, j], label=rf"horizon $j={j+1}$", marker="o", linewidth=1.2)
                plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
                plt.xlabel(r"Iteration $k$")
                plt.ylabel(rf"Slack $S_{{{oi+1},j}}(k)$")
                plt.title(rf"Obstacle {oi+1}: slacks across prediction horizon")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=min(4, N), fontsize="small")
                plt.tight_layout()

                save_figures(
                    [(fig_slack_i, f"slack_obs{oi+1}_{'after' if after_updates else 'before'}.svg")],
                    experiment_folder)

    stage_cost = stage_cost.reshape(-1)

    obs_positions = np.array(obs_positions)   # shape (T, m, 2)
    out_gif = os.path.join(experiment_folder, f"system_and_obstacle_{'after' if after_updates else 'before'}.gif")

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

    T_pred = plans.shape[0]
    print(f"plans.shape", plans.shape)
    make_system_obstacle_animation_v2(
        states[:T_pred],
        plans,
        obs_positions[:T_pred],
        radii,
        CONSTRAINTS_X[0],
        out_gif,
        trail_len=mpc.horizon      # fade the tail to last H
    )
    out_gif = os.path.join(experiment_folder, f"system_and_obstaclewithobstpred_{'after' if after_updates else 'before'}.gif")
    make_system_obstacle_animation_v3(
        states[:T_pred],
        plans,
        obs_positions[:T_pred],
        radii,
        CONSTRAINTS_X[0],
        out_gif,
        trail_len=mpc.horizon,      # fade the tail to last H
        camera="follow",
        follow_width=1.0,             # view width around agent when following
        follow_height=1.0,
    )


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

    suffix   = "after" if after_updates else "before"
    svg_dir  = os.path.join(experiment_folder, f"snapshots_{suffix}")
    os.makedirs(svg_dir, exist_ok=True)

    # keep lengths consistent with your GIFs
    T_pred = plans.shape[0]

    make_system_obstacle_svg_frames_v3(
        states_eval=states[:T_pred],
        pred_paths=plans,                      # (T_pred, N+1, 2)
        obs_positions=obs_positions[:T_pred], # (T_pred, m, 2)
        radii=radii,
        constraints_x=CONSTRAINTS_X[0],

        svg_dir=svg_dir,
        svg_prefix=f"system_{suffix}",        # files like system_before_0000.svg

        start=0, stop=T_pred, stride=1,       # every frame
        camera="follow",                      # match your GIF if you want
        follow_width=1.0,
        follow_height=1.0,
        legend_outside=True,
        keep_text_as_text=True,               # selectable text in SVG
        pad_inches=0.05,
    )

    fig, axes = make_system_obstacle_montage_v1(
    states[:T_pred], plans, obs_positions[:T_pred], radii, CONSTRAINTS_X[0],
    frame_indices=[6, 11, 14, 16, 17, 18, 20, 26],
    grid=(3, 3),
    use_empty_cell_for_legend=True,
    label_outer_only=True,          # <- only borders labeled
    ticklabels_outer_only=True,
    legend_auto_scale=True,
    legend_scale_factor=0.8,
    axis_labelsize=30, tick_fontsize=20, axis_labelpad_xy=(24,24),
    k_fontsize=20,
    figsize_per_ax=(5.0, 5.0),
    auto_enlarge_when_outer_only=2,   # make panels bigger
    gaps_outer_only=(0.01, 0.01),        # tighter gaps
    k_annotation="inside", k_loc="upper left", k_fmt="k={k}",
    figscale=0.5,
    out_path=os.path.join(svg_dir, f"NN_snapshots_{'after' if after_updates else 'before'}.pdf")

    )


    #save in an npz file

    suffix   = "after" if after_updates else "before"
    data_dir = os.path.join(experiment_folder, "thesis_data_nn")

    # (Optional) guard against weird dtypes if anything came from CasADi
    states        = np.asarray(states,        dtype=np.float64)
    actions       = np.asarray(actions,       dtype=np.float64)
    stage_cost    = np.asarray(stage_cost,    dtype=np.float64).reshape(-1)
    g_resid_lst   = np.asarray(g_resid_lst,   dtype=np.float64)
    hx            = np.asarray(hx,            dtype=np.float64)
    alphas        = np.asarray(alphas,        dtype=np.float64)
    obs_positions = np.asarray(obs_positions, dtype=np.float64)
    lam_g_hist    = np.asarray(lam_g_hist,    dtype=np.float64)
    plans         = np.asarray(plans,         dtype=np.float64)
    slacks_eval  = np.asarray(slacks_eval,  dtype=np.float64)

    sim_data = NPZBuilder(data_dir, "simulation", float_dtype="float32")
    sim_data.add(
        states=states,
        actions=actions,
        stage_cost=stage_cost,
        g_resid=g_resid_lst,
        hx=hx,
        alphas=alphas,
        obs_positions=obs_positions,
        lam_g_hist=lam_g_hist,
        plans=plans,
        slacks_eval=slacks_eval
    )

    # include useful constants so plotting scripts are totally standalone
    sim_data.meta(
        radii=np.asarray(radii, dtype=np.float64),
        constraints_x=float(CONSTRAINTS_X[0]),
        horizon=int(mpc.horizon),
        dt=float(getattr(env, "dt", 0.0)),
        run_tag=suffix
    )

    npz_path = sim_data.finalize(suffix=suffix)
    print(f"[saved] {npz_path}")

    return stage_cost.sum()
