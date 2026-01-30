
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
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.patches import Patch
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
                                    ypred_list[(1)*mpc.nn.obst.obstacle_num], u_opt, params["nn_params"])))
        
        #warm start variables
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]
        
        S = solution["x"][mpc.na * (mpc.horizon) + mpc.ns * (mpc.horizon+1):]
        X_plan = cs.reshape(solution["x"][:mpc.ns * (mpc.horizon+1)], mpc.ns, mpc.horizon + 1)
        plan_xy = np.array(X_plan[:2, :]).T

        return u_opt, solution["f"], alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, S, plan_xy

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
    ax.set_xlim(-1.1*span, +0.5*span)
    ax.set_ylim(-1.1*span, +0.5*span)

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
        dot.set_data([xk], [yk])
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
    
    
def make_system_obstacle_animation_v2(
            states_eval: np.ndarray,      # (T,4) or (T,2)
            pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
            obs_positions: np.ndarray,    # (T, m, 2)
            radii: list,                  # (m,)
            constraints_x: float,         # used for static window
            out_path: str,                # output GIF path

            # display controls
            figsize=(6.5, 6),
            dpi=140,
            legend_outside=True,
            legend_loc="upper left",

            # zoom / camera
            camera="static",              # "static" or "follow"
            follow_width=4.0,             # view width around agent when following
            follow_height=4.0,

            # timing & colors
            trail_len: int | None = None, # if None → horizon length
            fps: int = 12,                # save speed (lower = slower)
            interval_ms: int = 300,       # live/preview speed (higher = slower)
            system_color: str = "C0",     # trail color
            pred_color: str = "orange",   # prediction color (line + markers)

            # output/interaction
            show: bool = False,           # open interactive window (zoom/pan)
            save_gif: bool = True,
            save_mp4: bool = False,
            mp4_path: str | None = None,
        ):
            """Animated plot of system, moving obstacles, trailing path, and predicted horizon."""

            # ---- harmonize lengths (avoid off-by-one) ----
            T_state = states_eval.shape[0]
            T_pred  = pred_paths.shape[0]
            T_obs   = obs_positions.shape[0]
            T = min(T_state, T_pred, T_obs)  # clamp to shortest
            system_xy    = states_eval[:T, :2]
            obs_positions = obs_positions[:T]
            pred_paths    = pred_paths[:T]

            # shapes
            Np1 = pred_paths.shape[1]
            N   = max(0, Np1 - 1)
            if trail_len is None:
                trail_len = N
            m = obs_positions.shape[1]

            # colors
            sys_rgb  = mcolors.to_rgb(system_color)
            pred_rgb = mcolors.to_rgb(pred_color)

            # ---- figure/axes ----
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.35)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(r"System + Moving Obstacles + Horizon")

            # initial static window (camera="follow" will override per-frame)
            span = constraints_x
            ax.set_xlim(-1.1*span, +0.5*span)
            ax.set_ylim(-1.1*span, +0.5*span)

            # ---- artists ----
            # trail: solid line + fading dots (dots exclude current)
            trail_ln, = ax.plot([], [], "-", lw=2, color=sys_rgb, zorder=2.0, label=fr"last {trail_len} steps")
            trail_pts  = ax.scatter([], [], s=26, zorder=2.1)

            # system dot (topmost)
            agent_pt,  = ax.plot([], [], "o", ms=7, color="red", zorder=5.0, label="system")

            # prediction: fading line (LineCollection) + markers (all orange)
            pred_lc = LineCollection([], linewidths=2, zorder=2.2)
            ax.add_collection(pred_lc)
            horizon_markers = [ax.plot([], [], "o", ms=5, color=pred_rgb, zorder=2.3)[0] for _ in range(N)]
            # proxy line so it appears in legend
            ax.plot([], [], "-", lw=2, color=pred_rgb, label="predicted horizon", zorder=2.2)

            # obstacles
            cmap   = plt.get_cmap("tab10")
            colors = cmap.colors
            circles = []
            for i, r in enumerate(radii):
                c = plt.Circle((0, 0), r, fill=False, color=colors[i % len(colors)], lw=2, label=f"obstacle {i+1}", zorder=1.0)
                ax.add_patch(c)
                circles.append(c)

            # legend placement
            if legend_outside:
                # leave room on the right
                fig.subplots_adjust(right=0.68)
                ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
            else:
                ax.legend(loc="upper right", framealpha=0.9)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # ---- helpers ----
            def _trail_window(k):
                start = max(0, k - trail_len)
                return start, k + 1

            def _set_follow_view(xc, yc):
                half_w = follow_width  / 2.0
                half_h = follow_height / 2.0
                ax.set_xlim(xc - half_w, xc + half_w)
                ax.set_ylim(yc - half_h, yc + half_h)

            # ---- init & update ----
            def init():
                trail_ln.set_data([], [])
                trail_pts.set_offsets(np.empty((0, 2)))
                agent_pt.set_data([], [])
                pred_lc.set_segments([])
                for mkr in horizon_markers:
                    mkr.set_data([], [])
                for c in circles:
                    c.center = (0, 0)
                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles]

            def update(k):
                xk, yk = system_xy[k]
                agent_pt.set_data([xk], [yk])

                if camera == "follow":
                    _set_follow_view(xk, yk)

                # trail: line + fading dots (exclude current)
                s, e = _trail_window(k)
                tail_xy = system_xy[s:e]
                trail_ln.set_data(tail_xy[:, 0], tail_xy[:, 1])

                pts_xy = tail_xy[:-1]
                if len(pts_xy) > 0:
                    trail_pts.set_offsets(pts_xy)
                    n = len(pts_xy)
                    alphas = np.linspace(0.3, 1.0, n)  # old→light, new→solid
                    cols = np.tile((*sys_rgb, 1.0), (n, 1))
                    cols[:, 3] = alphas
                    trail_pts.set_facecolors(cols)
                    trail_pts.set_edgecolors('none')
                else:
                    trail_pts.set_offsets(np.empty((0, 2)))

                # prediction: fading line + markers
                ph = pred_paths[k]                  # (N+1, 2)
                if N > 0:
                    future = ph[1:, :]              # (N, 2)
                    pred_poly = np.vstack((ph[0:1, :], future))  # include current for first segment
                    segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)  # (N, 2, 2)
                    pred_lc.set_segments(segs)

                    seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
                    seg_cols[:, 3] = np.linspace(1.0, 0.35, N)  # near→far fade
                    pred_lc.set_colors(seg_cols)

                    for j in range(N):
                        horizon_markers[j].set_data([future[j, 0]], [future[j, 1]])
                else:
                    pred_lc.set_segments([])
                    for mkr in horizon_markers:
                        mkr.set_data([], [])

                # obstacles
                for i, c in enumerate(circles):
                    cx, cy = obs_positions[k, i]
                    c.center = (cx, cy)

                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles]

            # blit=False if camera follows (limits change each frame)
            blit_flag = (camera != "follow")
            ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                        blit=blit_flag, interval=interval_ms)

            # ---- save / show ----
            if save_gif:
                ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
            if save_mp4:
                try:
                    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
                    ani.save(mp4_path or out_path.replace(".gif", ".mp4"), writer=writer, dpi=dpi)
                except Exception as e:
                    print("MP4 save failed. Install ffmpeg or add it to PATH. Error:", e)

            if show:
                plt.show()   # interactive zoom/pan
            else:
                plt.close(fig)
                    

def make_system_obstacle_animation_v3(
        states_eval: np.ndarray,      # (T,4) or (T,2)
        pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
        obs_positions: np.ndarray,    # (T, m, 2)
        radii: list,                  # (m,)
        constraints_x: float,         # used for static window
        out_path: str,                # output GIF path

        # display controls
        figsize=(6.5, 6),
        dpi=140,
        legend_outside=True,
        legend_loc="upper left",

        # zoom / camera
        camera="static",              # "static" or "follow"
        follow_width=4.0,             # view width around agent when following
        follow_height=4.0,

        # timing & colors
        trail_len: int | None = None, # if None → horizon length
        fps: int = 12,                # save speed (lower = slower)
        interval_ms: int = 300,       # live/preview speed (higher = slower)
        system_color: str = "C0",     # trail color
        pred_color: str = "orange",   # prediction color (line + markers)

        # output/interaction
        show: bool = False,           # open interactive window (zoom/pan)
        save_gif: bool = True,
        save_mp4: bool = False,
        mp4_path: str | None = None,
    ):
    """Animated plot of system, moving obstacles, trailing path, and predicted horizon.
       Now also draws faded, dashed obstacle outlines at the next N predicted steps.
    """

    # ---- harmonize lengths (avoid off-by-one) ----
    T_state = states_eval.shape[0]
    T_pred  = pred_paths.shape[0]
    T_obs   = obs_positions.shape[0]
    T = min(T_state, T_pred, T_obs)  # clamp to shortest
    system_xy     = states_eval[:T, :2]
    obs_positions = obs_positions[:T]
    pred_paths    = pred_paths[:T]

    # shapes
    Np1 = pred_paths.shape[1]
    N   = max(0, Np1 - 1)
    if trail_len is None:
        trail_len = N
    m = obs_positions.shape[1]

    # colors
    sys_rgb  = mcolors.to_rgb(system_color)
    pred_rgb = mcolors.to_rgb(pred_color)

    # ---- figure/axes ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"System + Moving Obstacles + Horizon")

    # initial static window (camera="follow" will override per-frame)
    span = constraints_x
    ax.set_xlim(-1.1*span, +0.2*span)   # widened so circles aren’t clipped
    ax.set_ylim(-1.1*span, +0.2*span)

    # ---- artists ----
    # trail: solid line + fading dots (dots exclude current)
    trail_ln, = ax.plot([], [], "-", lw=2, color=sys_rgb, zorder=2.0, label=fr"last {trail_len} steps")
    trail_pts  = ax.scatter([], [], s=26, zorder=2.1)

    # system dot (topmost)
    agent_pt,  = ax.plot([], [], "o", ms=7, color="red", zorder=5.0, label="system")

    # prediction: fading line (LineCollection) + markers (all orange)
    pred_lc = LineCollection([], linewidths=2, zorder=2.2)
    ax.add_collection(pred_lc)
    horizon_markers = [ax.plot([], [], "o", ms=5, color=pred_rgb, zorder=2.3)[0] for _ in range(N)]
    # proxy line so it appears in legend
    ax.plot([], [], "-", lw=2, color=pred_rgb, label="predicted horizon", zorder=2.2)

    # obstacles (current time k)
    cmap   = plt.get_cmap("tab10")
    colors = cmap.colors
    circles = []
    for i, r in enumerate(radii):
        c = plt.Circle((0, 0), r, fill=False, color=colors[i % len(colors)],
                       lw=2, label=f"obstacle {i+1}", zorder=1.0)
        ax.add_patch(c)
        circles.append(c)

    # --- NEW: predicted obstacle outlines for the next N steps (ghosted) ---
    # one dashed circle per (future step h=1..N, obstacle i=1..m)
    pred_alpha_seq = np.linspace(0.35, 0.3, max(N, 1))  # nearer -> darker, farther -> lighter
    pred_circles_layers = []  # list of lists: [layer_h][i] -> patch
    for h in range(1, N+1):
        layer = []
        a = float(pred_alpha_seq[h-1])
        for i, r in enumerate(radii):
            pc = plt.Circle((0, 0), r, fill=False,
                            color=colors[i % len(colors)],
                            lw=1.2, linestyle="--", alpha=a,
                            zorder=0.8)  # behind current circles
            ax.add_patch(pc)
            layer.append(pc)
        pred_circles_layers.append(layer)
    if N > 0:
        # legend proxy for predicted obstacle outlines
        ax.plot([], [], linestyle="--", lw=1.2, color=colors[0],
                alpha=0.3, label="obstacle (predicted)")

    # legend placement
    if legend_outside:
        fig.subplots_adjust(right=0.68)
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, framealpha=0.9)
    else:
        ax.legend(loc="upper right", framealpha=0.9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- helpers ----
    def _trail_window(k):
        start = max(0, k - trail_len)
        return start, k + 1

    def _set_follow_view(xc, yc):
        half_w = follow_width  / 2.0 + max(radii)
        half_h = follow_height / 2.0 + max(radii)
        ax.set_xlim(xc - half_w, xc + half_w)
        ax.set_ylim(yc - half_h, yc + half_h)

    # ---- init & update ----
    def init():
        trail_ln.set_data([], [])
        trail_pts.set_offsets(np.empty((0, 2)))
        agent_pt.set_data([], [])
        pred_lc.set_segments([])
        for mkr in horizon_markers:
            mkr.set_data([], [])
        for c in circles:
            c.center = (0, 0)
        for layer in pred_circles_layers:
            for pc in layer:
                pc.center = (0, 0)
                pc.set_visible(False)
        return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles,
                *[pc for layer in pred_circles_layers for pc in layer]]

    def update(k):
        xk, yk = system_xy[k]
        agent_pt.set_data([xk], [yk])

        if camera == "follow":
            _set_follow_view(xk, yk)

        # trail: line + fading dots (exclude current)
        s, e = _trail_window(k)
        tail_xy = system_xy[s:e]
        trail_ln.set_data(tail_xy[:, 0], tail_xy[:, 1])

        pts_xy = tail_xy[:-1]
        if len(pts_xy) > 0:
            trail_pts.set_offsets(pts_xy)
            n = len(pts_xy)
            alphas = np.linspace(0.3, 1.0, n)  # old→light, new→solid
            cols = np.tile((*sys_rgb, 1.0), (n, 1))
            cols[:, 3] = alphas
            trail_pts.set_facecolors(cols)
            trail_pts.set_edgecolors('none')
        else:
            trail_pts.set_offsets(np.empty((0, 2)))

        # prediction: fading line + markers
        ph = pred_paths[k]                  # (N+1, 2)
        if N > 0:
            future = ph[1:, :]              # (N, 2)
            pred_poly = np.vstack((ph[0:1, :], future))   # include current for first segment
            segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)  # (N, 2, 2)
            pred_lc.set_segments(segs)

            seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
            seg_cols[:, 3] = np.linspace(1.0, 0.35, N)  # near→far fade
            pred_lc.set_colors(seg_cols)

            for j in range(N):
                horizon_markers[j].set_data([future[j, 0]], [future[j, 1]])
        else:
            pred_lc.set_segments([])
            for mkr in horizon_markers:
                mkr.set_data([], [])

        # obstacles (current time k)
        for i, c in enumerate(circles):
            cx, cy = obs_positions[k, i]
            c.center = (cx, cy)

        # --- predicted obstacle outlines at k+1..k+N ---
        if N > 0:
            for h, layer in enumerate(pred_circles_layers, start=1):
                t = min(k + h, T - 1)  # clamp to last available pose
                for i, pc in enumerate(layer):
                    cx, cy = obs_positions[t, i]
                    pc.center = (cx, cy)
                    pc.set_visible(True)

        return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles,
                *[pc for layer in pred_circles_layers for pc in layer]]

    # blit=False if camera follows (limits change each frame)
    blit_flag = (camera != "follow")
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                  blit=blit_flag, interval=interval_ms)

    # ---- save / show ----
    if save_gif:
        ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    if save_mp4:
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
            ani.save(mp4_path or out_path.replace(".gif", ".mp4"),
                     writer=writer, dpi=dpi)
        except Exception as e:
            print("MP4 save failed. Install ffmpeg or add it to PATH. Error:", e)

    if show:
        plt.show()
    else:
        plt.close(fig)      
    
def make_system_obstacle_svg_frames_v3(
        states_eval: np.ndarray,      # (T,4) or (T,2)
        pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
        obs_positions: np.ndarray,    # (T, m, 2)
        radii: list,                  # (m,)
        constraints_x: float,         # used for static window

        # where to save SVGs
        svg_dir: str,                 # directory to write frames
        svg_prefix: str = "frame",    # filename prefix -> frame_0000.svg

        # which frames to export
        start: int = 0,
        stop: int | None = None,      # exclusive; None -> T
        stride: int = 1,

        # display controls (match your v3 defaults)
        figsize=(6.5, 6),
        legend_outside=True,
        legend_loc="upper left",
        camera="static",              # "static" or "follow"
        follow_width=4.0,
        follow_height=4.0,
        system_color: str = "C0",
        pred_color: str = "orange",

        # SVG options
        keep_text_as_text: bool = True,   # True -> selectable/editable text in SVG
        pad_inches: float = 0.05,         # outer padding when saving
    ):
    """
    Save per-frame SVG snapshots of the same scene as make_system_obstacle_animation_v3.
    Produces svg_dir/svg_prefix_0000.svg, svg_dir/svg_prefix_0001.svg, ...
    """

    # --- SVG config ---
    if keep_text_as_text:
        rcParams['svg.fonttype'] = 'none'   # keep text as text (not paths)

    # ---- harmonize lengths ----
    T = min(states_eval.shape[0], pred_paths.shape[0], obs_positions.shape[0])
    system_xy     = states_eval[:T, :2]
    pred_paths    = pred_paths[:T]
    obs_positions = obs_positions[:T]

    Np1 = pred_paths.shape[1]
    N   = max(0, Np1 - 1)
    m   = obs_positions.shape[1]

    # colors
    sys_rgb  = mcolors.to_rgb(system_color)
    pred_rgb = mcolors.to_rgb(pred_color)

    # ---- figure/axes ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"System + Moving Obstacles + Horizon")

    span = constraints_x
    ax.set_xlim(-1.1*span, +0.2*span)
    ax.set_ylim(-1.1*span, +0.2*span)

    # ---- artists (same as v3) ----
    trail_ln, = ax.plot([], [], "-", lw=2, color=sys_rgb, zorder=2.0, label="last steps")
    trail_pts  = ax.scatter([], [], s=26, zorder=2.1)

    agent_pt,  = ax.plot([], [], "o", ms=7, color="red", zorder=5.0, label="system")

    pred_lc = LineCollection([], linewidths=2, zorder=2.2)
    ax.add_collection(pred_lc)
    horizon_markers = [ax.plot([], [], "o", ms=5, color=pred_rgb, zorder=2.3)[0] for _ in range(N)]
    ax.plot([], [], "-", lw=2, color=pred_rgb, label="predicted horizon", zorder=2.2)

    cmap   = plt.get_cmap("tab10")
    colors = cmap.colors
    circles = []
    for i, r in enumerate(radii):
        c = plt.Circle((0, 0), r, fill=False, color=colors[i % len(colors)],
                       lw=2, label=f"obstacle {i+1}", zorder=1.0)
        ax.add_patch(c)
        circles.append(c)

    # ghosted predicted obstacle outlines
    pred_alpha_seq = np.linspace(0.35, 0.30, max(N, 1))
    pred_circles_layers = []
    for h in range(1, N+1):
        layer = []
        a = float(pred_alpha_seq[h-1])
        for i, r in enumerate(radii):
            pc = plt.Circle((0, 0), r, fill=False,
                            color=colors[i % len(colors)],
                            lw=1.2, linestyle="--", alpha=a,
                            zorder=0.8)
            ax.add_patch(pc)
            layer.append(pc)
        pred_circles_layers.append(layer)
    if N > 0:
        ax.plot([], [], linestyle="--", lw=1.2, color=colors[0], alpha=0.3, label="obstacle (predicted)")

    # legend
    if legend_outside:
        fig.subplots_adjust(right=0.70)
        leg = ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0, framealpha=0.9)
    else:
        leg = ax.legend(loc="upper right", framealpha=0.9)

    # helpers
    def _trail_window(k, trail_len=N):
        s = max(0, k - trail_len)
        return s, k + 1

    def _set_follow_view(xc, yc):
        half_w = follow_width  / 2.0 + max(radii)
        half_h = follow_height / 2.0 + max(radii)
        ax.set_xlim(xc - half_w, xc + half_w)
        ax.set_ylim(yc - half_h, yc + half_h)

    # init
    def _init():
        trail_ln.set_data([], [])
        trail_pts.set_offsets(np.empty((0, 2)))
        agent_pt.set_data([], [])
        pred_lc.set_segments([])
        for mkr in horizon_markers:
            mkr.set_data([], [])
        for c in circles:
            c.center = (0, 0)
        for layer in pred_circles_layers:
            for pc in layer:
                pc.center = (0, 0)
                pc.set_visible(False)

    # update one frame (returns artist list if you need it)
    def _update(k):
        xk, yk = system_xy[k]
        agent_pt.set_data([xk], [yk])

        if camera == "follow":
            _set_follow_view(xk, yk)

        # trail
        s, e = _trail_window(k)
        tail_xy = system_xy[s:e]
        trail_ln.set_data(tail_xy[:, 0], tail_xy[:, 1])

        pts_xy = tail_xy[:-1]
        if len(pts_xy) > 0:
            trail_pts.set_offsets(pts_xy)
            n = len(pts_xy)
            alphas = np.linspace(0.3, 1.0, n)
            cols = np.tile((*sys_rgb, 1.0), (n, 1))
            cols[:, 3] = alphas
            trail_pts.set_facecolors(cols)
            trail_pts.set_edgecolors('none')
        else:
            trail_pts.set_offsets(np.empty((0, 2)))

        # prediction path
        ph = pred_paths[k]
        if N > 0:
            future = ph[1:, :]
            pred_poly = np.vstack((ph[0:1, :], future))
            segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)
            pred_lc.set_segments(segs)
            seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
            seg_cols[:, 3] = np.linspace(1.0, 0.35, N)
            pred_lc.set_colors(seg_cols)
            for j in range(N):
                horizon_markers[j].set_data([future[j, 0]], [future[j, 1]])
        else:
            pred_lc.set_segments([])
            for mkr in horizon_markers:
                mkr.set_data([], [])

        # current obstacles
        for i, c in enumerate(circles):
            cx, cy = obs_positions[k, i]
            c.center = (cx, cy)

        # ghosted future obstacle outlines
        if N > 0:
            for h, layer in enumerate(pred_circles_layers, start=1):
                t = min(k + h, T - 1)
                for i, pc in enumerate(layer):
                    cx, cy = obs_positions[t, i]
                    pc.center = (cx, cy)
                    pc.set_visible(True)

    # ---- export loop ----
    os.makedirs(svg_dir, exist_ok=True)
    _init()

    if stop is None:
        stop = T
    frames = range(start, min(stop, T), stride)

    for k in frames:
        _update(k)
        fig.canvas.draw_idle()
        # include legend in the tight bounding box
        fig.savefig(
            os.path.join(svg_dir, f"{svg_prefix}_{k:04d}.svg"),
            format="svg",
            bbox_inches="tight",
            pad_inches=pad_inches,
            bbox_extra_artists=[leg],
        )

    plt.close(fig)
    
    
def draw_snapshot_on_axes(
    ax: plt.Axes,
    k: int,
    system_xy: np.ndarray,      # (T,2)
    pred_paths: np.ndarray,     # (T, N+1, 2)
    obs_positions: np.ndarray,  # (T, m, 2)
    radii: list,                # (m,)
    *,
    constraints_x: float,
    camera: str = "static",     # "static" | "follow"
    follow_width: float = 4.0,
    follow_height: float = 4.0,
    system_color: str = "C0",
    pred_color: str = "orange",
    legend_colors=None,         # optional (tab10 etc.), else inferred
    show_legend=False           # we’ll use a single legend at figure-level
):
    """
    Render a single frame k onto the provided axes (no title).
    Matches the look of your v3 animation/snapshots.
    """
    T = min(system_xy.shape[0], pred_paths.shape[0], obs_positions.shape[0])
    k = int(np.clip(k, 0, T - 1))
    ph = pred_paths
    Np1 = ph.shape[1]
    N = max(0, Np1 - 1)
    m = obs_positions.shape[1]

    sys_rgb  = mcolors.to_rgb(system_color)
    pred_rgb = mcolors.to_rgb(pred_color)

    # axes styling
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    # window
    span = constraints_x
    if camera == "static":
        ax.set_xlim(-1.1*span, +0.2*span)
        ax.set_ylim(-1.1*span, +0.2*span)
    # colors
    if legend_colors is None:
        cmap = plt.get_cmap("tab10")
        legend_colors = list(cmap.colors)

    # trail (use last N steps to match animation vibe)
    def _trail_window(idx, trail_len=N):
        s = max(0, idx - trail_len)
        return s, idx + 1

    s, e = _trail_window(k)
    tail_xy = system_xy[s:e]
    ax.plot(tail_xy[:, 0], tail_xy[:, 1], "-", lw=2, color=sys_rgb, zorder=2.0)
    pts_xy = tail_xy[:-1]
    if len(pts_xy) > 0:
        n = len(pts_xy)
        alphas = np.linspace(0.3, 1.0, n)
        cols = np.tile((*sys_rgb, 1.0), (n, 1))
        cols[:, 3] = alphas
        sc = ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=26, zorder=2.1)
        sc.set_facecolors(cols)
        sc.set_edgecolors('none')

    # agent
    xk, yk = system_xy[k]
    ax.plot([xk], [yk], "o", ms=7, color="red", zorder=5.0)

    if camera == "follow":
        half_w = follow_width  / 2.0 + max(radii)
        half_h = follow_height / 2.0 + max(radii)
        ax.set_xlim(xk - half_w, xk + half_w)
        ax.set_ylim(yk - half_h, yk + half_h)

    # prediction path + horizon markers
    if N > 0:
        future = ph[k, 1:, :]
        pred_poly = np.vstack((ph[k, 0:1, :], future))
        segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)
        lc = LineCollection(segs, linewidths=2, zorder=2.2)
        seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
        seg_cols[:, 3] = np.linspace(1.0, 0.35, N)
        lc.set_colors(seg_cols)
        ax.add_collection(lc)
        # markers at horizon points
        for j in range(N):
            ax.plot([future[j, 0]], [future[j, 1]], "o", ms=5, color=pred_rgb, zorder=2.3)

    # current obstacles
    circles = []
    for i, r in enumerate(radii):
        cx, cy = obs_positions[k, i]
        c = plt.Circle((cx, cy), r, fill=False, color=legend_colors[i % len(legend_colors)],
                       lw=2, zorder=1.0)
        ax.add_patch(c)
        circles.append(c)

    # ghosted predicted obstacle outlines
    if N > 0:
        pred_alpha_seq = np.linspace(0.35, 0.30, max(N, 1))
        for h in range(1, N+1):
            a = float(pred_alpha_seq[h-1])
            t = min(k + h, T - 1)
            for i, r in enumerate(radii):
                cx, cy = obs_positions[t, i]
                pc = plt.Circle((cx, cy), r, fill=False,
                                color=legend_colors[i % len(legend_colors)],
                                lw=1.2, linestyle="--", alpha=a, zorder=0.8)
                ax.add_patch(pc)

    # no titles per subplot
    ax.set_title("")


def make_system_obstacle_montage_v1(
    states_eval: np.ndarray, pred_paths: np.ndarray, obs_positions: np.ndarray,
    radii: list, constraints_x: float, frame_indices: list[int],

    # ---- layout ----
    grid: tuple[int, int] | None = None,
    figsize_per_ax: tuple[float, float] = (3.6, 3.4),
    hspace: float = 0.45, wspace: float = 0.45,
    figscale: float = 1.0,  # global figure scale

    # ---- camera ----
    camera: str = "static", follow_width: float = 4.0, follow_height: float = 4.0,

    # ---- styling ----
    system_color: str = "C0", pred_color: str = "orange",
    tick_fontsize: int = 16, axis_labelsize: int = 22,
    axis_labelpad_xy: tuple[int, int] = (24, 24),
    spine_width: float = 1.25, tick_width: float = 1.25,

    # ---- k label ----
    k_annotation: str = "inside", k_loc: str = "upper left",
    k_box: bool = True, k_fontsize: int = 14, k_fmt: str = "k={k}",

    # ---- legend (explicit, centered in empty cell) ----
    legend_fontsize: int = 16,
    legend_auto_scale: bool = True,        # scales with figure size
    legend_scale_factor: float = 0.92,     # gentle reduction so it doesn't dominate
    use_empty_cell_for_legend: bool = True,# use first empty cell if available
    legend_borderaxespad: float = 0.6,
    legend_borderpad: float = 0.6,

    # ---- label/tick policies ----
    label_outer_only: bool = True,          # Y on first col, X on last row
    ticklabels_outer_only: bool = True,     # tick labels only on outer panels

    # Poster look (hide everything globally)
    hide_axis_labels: bool = False,
    hide_ticks: bool = False,

    # Auto-enlarge when axes fully hidden (poster)
    auto_enlarge_when_no_axes: float | None = 1.25,
    gaps_no_axes: tuple[float, float] = (0.12, 0.12),

    # Auto-enlarge when using outer-only (not fully hidden)
    auto_enlarge_when_outer_only: float | None = 1.25,
    gaps_outer_only: tuple[float, float] = (0.10, 0.10),

    # Optional: trail length override (default uses horizon N)
    trail_len: int | None = None,

    # ---- output ----
    out_path: str | None = None, dpi: int = 200,
):
    """
    Multi-frame montage with:
      • outer-only or hidden axes options (+ auto-enlarge to use freed space)
      • per-panel k badges
      • one shared legend: centered in the first empty grid cell (if any),
        otherwise placed outside on the right. Legend font auto-scales with
        figure size and then is lightly reduced by `legend_scale_factor`.
    """
    # ---------- harmonize ----------
    T = min(states_eval.shape[0], pred_paths.shape[0], obs_positions.shape[0])
    system_xy, pred_paths, obs_positions = states_eval[:T, :2], pred_paths[:T], obs_positions[:T]
    if not frame_indices:
        raise ValueError("frame_indices is empty.")
    for k in frame_indices:
        if not (0 <= k < T):
            raise ValueError(f"frame index {k} out of range [0,{T-1}]")

    N = max(0, pred_paths.shape[1] - 1)   # prediction horizon
    m = obs_positions.shape[1]
    trail_N = trail_len if trail_len is not None else N

    # ---------- grid ----------
    n_frames = len(frame_indices)
    if grid is None:
        import math
        rows = int(math.floor(math.sqrt(n_frames))) or 1
        cols = int(math.ceil(n_frames / rows))
        if rows * cols < n_frames:
            rows += 1
    else:
        rows, cols = grid

    # ---------- scaling / gaps ----------
    scale = float(figscale)
    use_no_axes_layout = (hide_axis_labels and hide_ticks)
    if use_no_axes_layout and (auto_enlarge_when_no_axes is not None):
        scale *= float(auto_enlarge_when_no_axes); hspace, wspace = gaps_no_axes
    elif (not use_no_axes_layout and label_outer_only and ticklabels_outer_only
          and (auto_enlarge_when_outer_only is not None)):
        scale *= float(auto_enlarge_when_outer_only); hspace, wspace = gaps_outer_only

    fig_w = max(1, cols) * figsize_per_ax[0] * scale
    fig_h = max(1, rows) * figsize_per_ax[1] * scale
    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # ---------- colors & legend proxies ----------
    sys_rgb, pred_rgb = mcolors.to_rgb(system_color), mcolors.to_rgb(pred_color)
    cmap = plt.get_cmap("tab10"); obst_cols = [cmap.colors[i % len(cmap.colors)] for i in range(m)]
    handles = [
        mlines.Line2D([], [], color=sys_rgb, lw=2, label=f"Last Steps (N={trail_N})"),
        mlines.Line2D([], [], marker="o", linestyle="None", color="red", markersize=7, label="System"),
        mlines.Line2D([], [], color=pred_rgb, lw=2, label=f"Predicted Horizon (N={N})"),
        *[mlines.Line2D([], [], color=obst_cols[i], lw=2, label=f"Obstacle {i+1}") for i in range(m)]
    ]
    if N > 0:
        handles.append(mlines.Line2D([], [], color=obst_cols[0], lw=1.2, ls="--", alpha=0.3,
                                     label=f"Obstacle (Predicted, N={N} Ahead)"))

    # legend font size (auto + gentle reduction)
    legend_fs_eff = legend_fontsize * (scale if legend_auto_scale else 1.0) * legend_scale_factor

    # ---------- helpers ----------
    def _trail_window(k, trail_len_local=trail_N):
        s = max(0, k - max(0, int(trail_len_local))); return s, k + 1
    pred_alpha_seq = np.linspace(0.35, 0.30, max(N, 1))

    def _render_one(ax, k: int):
        ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.35)
        # labels/ticks
        if not hide_axis_labels:
            ax.set_xlabel(r"X", fontsize=axis_labelsize, labelpad=axis_labelpad_xy[0])
            ax.set_ylabel(r"Y", fontsize=axis_labelsize, labelpad=axis_labelpad_xy[1])
        else:
            ax.set_xlabel(""); ax.set_ylabel("")
        if hide_ticks:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.tick_params(labelsize=tick_fontsize, width=tick_width)
        for s in ax.spines.values(): s.set_linewidth(spine_width)
        # view
        if camera == "follow":
            xk, yk = system_xy[k]
            half_w = follow_width/2.0 + (max(radii) if radii else 0.0)
            half_h = follow_height/2.0 + (max(radii) if radii else 0.0)
            ax.set_xlim(xk - half_w, xk + half_w); ax.set_ylim(yk - half_h, yk + half_h)
        else:
            span = constraints_x; ax.set_xlim(-1.1*span, +0.2*span); ax.set_ylim(-1.1*span, +0.2*span)
        # trail
        s0, e0 = _trail_window(k); tail_xy = system_xy[s0:e0]
        ax.plot(tail_xy[:,0], tail_xy[:,1], "-", lw=2, color=sys_rgb, zorder=2.0)
        if len(tail_xy) > 1:
            alphas = np.linspace(0.3, 1.0, len(tail_xy)-1)
            for p, a in zip(tail_xy[:-1], alphas):
                ax.plot(p[0], p[1], "o", ms=4.5, color=(*sys_rgb, a), zorder=2.1, markeredgewidth=0)
        # system point
        ax.plot(system_xy[k,0], system_xy[k,1], "o", ms=7, color="red", zorder=5.0)
        # predicted path
        if N > 0:
            ph = pred_paths[k]; future = ph[1:, :]
            poly = np.vstack((ph[0:1,:], future)); segs = np.stack([poly[:-1], poly[1:]], axis=1)
            lc = LineCollection(segs, linewidths=2, zorder=2.2)
            seg_cols = np.tile((*pred_rgb, 1.0), (N, 1)); seg_cols[:,3] = np.linspace(1.0, 0.35, N)
            lc.set_colors(seg_cols); ax.add_collection(lc)
            for j in range(N): ax.plot(future[j,0], future[j,1], "o", ms=5, color=pred_rgb, zorder=2.3)
        # obstacles
        for i, r_i in enumerate(radii):
            cx, cy = obs_positions[k, i]
            ax.add_patch(plt.Circle((cx, cy), r_i, fill=False, color=obst_cols[i], lw=2, zorder=1.0))
        if N > 0:
            for h in range(1, N+1):
                a = float(pred_alpha_seq[h-1]); t = min(k+h, T-1)
                for i, r_i in enumerate(radii):
                    cx, cy = obs_positions[t, i]
                    ax.add_patch(plt.Circle((cx, cy), r_i, fill=False,
                                            color=obst_cols[i], lw=1.2, ls="--", alpha=a, zorder=0.8))
        # k badge
        if k_annotation == "inside":
            pos = {"upper left":(0.02,0.98,"left","top"),
                   "upper right":(0.98,0.98,"right","top"),
                   "lower left":(0.02,0.02,"left","bottom"),
                   "lower right":(0.98,0.02,"right","bottom")}
            xfa,yfa,ha,va = pos.get(k_loc, pos["upper left"])
            bbox = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, lw=0.8) if k_box else None
            ax.text(xfa,yfa,k_fmt.format(k=k), transform=ax.transAxes, ha=ha, va=va,
                    fontsize=k_fontsize, bbox=bbox)
        elif k_annotation == "below":
            ax.annotate(k_fmt.format(k=k), xy=(0.5, -0.25), xycoords="axes fraction",
                        ha="center", va="top", fontsize=k_fontsize)

    # ---------- render frames ----------
    ax_list = axs.ravel().tolist()
    for idx, k in enumerate(frame_indices): _render_one(ax_list[idx], k)

    # ---------- hide extras; remember first empty for legend ----------
    filled_count = len(frame_indices)
    first_empty = None
    for j in range(filled_count, rows*cols):
        ax = ax_list[j]
        if first_empty is None: first_empty = ax
        ax.axis("off")            # keep the cell clean
        ax.set_visible(False)     # ensure later code won't re-enable it

    # ---------- outer-only post-pass (skip empty cells) ----------
    if (not hide_axis_labels and label_outer_only) or (not hide_ticks and ticklabels_outer_only):
        for r in range(rows):
            for c in range(cols):
                idx = r*cols + c
                if idx >= filled_count:   # skip the empty/legend cell(s)
                    continue
                ax = axs[r, c]
                is_first_col, is_last_row = (c == 0), (r == rows - 1)
                if not hide_axis_labels and label_outer_only:
                    ax.set_ylabel("Y" if is_first_col else "")
                    ax.set_xlabel("X" if is_last_row else "")
                if not hide_ticks and ticklabels_outer_only:
                    ax.tick_params(labelleft=is_first_col, labelbottom=is_last_row)
                    ax.tick_params(left=is_first_col, bottom=is_last_row)

    # ---------- legend (centered in empty cell if available) ----------
    if use_empty_cell_for_legend and first_empty is not None:
        leg_ax = first_empty
        leg_ax.set_visible(True)    # show the cell to host the legend (axes frame stays off)
        leg_ax.axis("off")
        # Centered placement: loc="center" with no bbox_to_anchor
        leg_ax.legend(handles=handles, loc="center",
                      framealpha=0.95, fontsize=legend_fs_eff,
                      borderaxespad=legend_borderaxespad, borderpad=legend_borderpad)
    else:
        # No empty cell → place outside on the right with margin
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.95, fontsize=legend_fs_eff,
                   borderaxespad=legend_borderaxespad, borderpad=legend_borderpad)
        fig.subplots_adjust(right=0.86)

    # ---------- save ----------
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig, axs



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
    
def noise_scale_by_distance(x, y, max_radius=0.5):
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
    mpc = MPC(layers_list, horizon, positions, radii, slack_penalty_eval, mode_params, modes)
    
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
                              patience_threshold, lr_decay_factor, horizon, modes, mode_params, positions, radii,
                              slack_penalty_MPC, slack_penalty_RL):
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
    Slack Penalty MPC: {slack_penalty_MPC}
    Slack Penalty RL: {slack_penalty_RL}

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