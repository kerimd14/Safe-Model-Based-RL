
import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
from gymnasium.spaces import Box
import casadi as cs
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque
import pandas as pd
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED, CONSTRAINTS_U



class RLclass:

        def __init__(
        self,
        params_innit,
        seed,
        alpha,
        gamma,
        decay_rate,
        layers_list,
        noise_scalingfactor,
        noise_variance,
        patience_threshold,
        lr_decay_factor,
        horizon,
        positions,
        radii,
        modes,
        mode_params,
        slack_penalty_MPC_L1, 
        slack_penalty_MPC_L2,
        slack_penalty_RL_L1,
        slack_penalty_RL_L2,
        violation_penalty,
        ):
            # Store random seed for reproducibility
            self.seed = seed

            # Create the environment
            self.env = env()
            
            # Penalty in RL stagecost on slacks
            self.slack_penalty_MPC_L1 = slack_penalty_MPC_L1
            self.slack_penalty_MPC_L2 = slack_penalty_MPC_L2,
            self.slack_penalty_RL_L1 = slack_penalty_RL_L1
            self.slack_penalty_RL_L2 = slack_penalty_RL_L2
            
            self.violation_penalty = violation_penalty

            # Initialize MPC and obstacle‐motion classes 
            self.mpc = MPC(layers_list, horizon, positions, radii, self.slack_penalty_MPC_L1, self.slack_penalty_MPC_L2, mode_params, modes)
            self.obst_motion = ObstacleMotion(positions, modes, mode_params)
            
            # layer list of input
            self.rnn_input_size = layers_list[0]
            self.layers_list = layers_list
            # Parameters of experiments and states
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit

            # Learning‐rate for parameter updates
            self.alpha = alpha
            
            # Build state bounds repeated over the horizon
            #np.tile takes an array and “tiles” (i.e. repeats) it to fill a larger array.
            self.X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
            self.X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)
            
            # Equality‐constraint bounds (Ax+Bu==0) - all zeros
            self.state_const_lbg = np.zeros(1*self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1*self.ns  * (self.horizon))

            # CBF safety constraints: h(x_{k+1})-h(x_k)+alpha*h(x_k) + s >= 0 → we invert so g =< 0
            # means the cbf constraint is bounded between -inf and zero --> for the g =< 0
            self.cbf_const_lbg = -np.inf * np.ones(self.mpc.rnn.obst.obstacle_num*(self.horizon))
            self.cbf_const_ubg = np.zeros(self.mpc.rnn.obst.obstacle_num*(self.horizon))
            
            
            # Discount factor for TD updates
            self.gamma = gamma

            # RNG for adding exploration noise
            self.np_random = np.random.default_rng(seed=self.seed)
            self.noise_scalingfactor = noise_scalingfactor
            self.noise_variance = noise_variance
            
            # Create CasADi mpc solver instances once for reuse
            self.solver_inst = self.mpc.MPC_solver()  #deterministic MPC solver
            self.solver_inst_random =self.mpc.MPC_solver_rand() # noisy MPC (the MPC with exploration noise)
            
            # Symbolic function to get the gradient of the MPC Lagrangian
            self.qlagrange_fn_jacob = self.mpc.generate_symbolic_mpcq_lagrange()
            print(f"generated the function")

            # Create CasADi qp solver instance once for reuse
            self.qp_solver = self.mpc.qp_solver_fn() # QP for constrained parameter updates
            print(f"casadi func created")
            # Learning‐rate scheduling
            self.decay_rate      = decay_rate
            self.patience_threshold = patience_threshold
            self.lr_decay_factor   = lr_decay_factor
            self.best_stage_cost   = np.inf
            self.best_params       = params_innit.copy()
            self.current_patience = 0

            
            # ADAM
            theta_vector_num = cs.vertcat(self.params_innit["rnn_params"])
            self.exp_avg = np.zeros(theta_vector_num.shape[0])
            self.exp_avg_sq = np.zeros(theta_vector_num.shape[0])
            self.adam_iter = 1
            
            print(f"before make rnn step")
            # hidden state 1 function 
            self.get_hidden_func = self.mpc.rnn.make_rnn_step()
            
            print(f"before flat input")
            self.flat_input_fn = self.mpc.make_flat_input_fn()
            
            
            # Warmstart variables storage
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  
            self.x_prev_QMPC        = cs.DM()  
            self.lam_x_prev_QMPC    = cs.DM()  
            self.lam_g_prev_QMPC    = cs.DM()  
            self.x_prev_VMPCrandom  = cs.DM()  
            self.lam_x_prev_VMPCrandom = cs.DM()  
            self.lam_g_prev_VMPCrandom = cs.DM()
            
            print(f"before make phi fn")
            #construct function to access h(x_{k+1},k+1)-alpha*h(x_{k},k)
            self.phi_func = self.mpc.make_phi_fn()
            
            print(f"before make h functions")
            # construct list to extract h(x_{k},k)
            self.h_func_list = self.mpc.rnn.obst.make_h_functions()
            
            self.spectral_radii_hist = []            
            print(f"initialization done")
            
            
        
        def save_figures(self, figures, experiment_folder, save_in_subfolder=False):
            # Choose to save or not save figure
            save_choice = True  
            
            if save_choice:
                # Decide which subfolder (if any) to use
                if save_in_subfolder == "Learning":
                    target_folder = os.path.join(experiment_folder, "learning_process")
                elif save_in_subfolder == "Evaluation":
                    target_folder = os.path.join(experiment_folder, "evaluation")
                else:
                    # No subfolder specified: save directly in experiment_folder
                    target_folder = experiment_folder

                # Create the directory if it doesn’t exist
                os.makedirs(target_folder, exist_ok=True)
                
                # Loop through (figure, filename) pairs
                for fig, filename in figures:
                    file_path = os.path.join(target_folder, filename)
                    # Save the matplotlib figure 
                    fig.savefig(file_path)
                    plt.close(fig)
                    print(f"Figure saved as: {file_path}")
            else:
                print("Figure not saved")
                
        def make_system_obstacle_animation_v2(
            self,
            states_eval: np.ndarray,      # (T,4) or (T,2)
            pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
            obs_positions: np.ndarray,    # (T, m, 2)
            radii: list,                  # (m,)
            constraints_x: float,         # used for static window
            out_path: str,                # output GIF path

            # display controls
            figsize=(6, 6),
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
            interval_ms: int = 150,       # live/preview speed (higher = slower)
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
                fig.subplots_adjust(right=0.80)
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
                
                
        def make_system_obstacle_animation_v3(self,
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

        def plot_B_update(self, B_update_history, experiment_folder):
            
            """"
            B_update_history  is a history if update vectors for the RL parameters
            """
            
            B_update = np.asarray(B_update_history)
            B_update = B_update.squeeze(-1)

            # Build labels for the first four diagonal P elements
            labels = [f"P[{i},{i}]" for i in range(4)]
            print(f"labels: {labels}")

            # The remaining columns correspond to RNN parameter updates
            nn_B_update = B_update[:, 4:]
            # Compute mean absolute update magnitude across RNN parameters for each iteration
            # take mean across rows (7,205) --> (7,)
            mean_mag = np.mean(np.abs(nn_B_update), axis=1)
            
            # #legend helper function
            # def safe_legend(loc="best", **kwargs):
            #     handles, labls = plt.gca().get_legend_handles_labels()
            #     if labls:
            #         plt.legend(loc=loc, **kwargs)

            # Plot updates for P parameters
            fig_p = plt.figure()
            for idx, lbl in enumerate(labels):
                plt.plot(B_update[:, idx], "o-", label=lbl)
            plt.xlabel("Update iteration")
            plt.ylabel("B_update")
            plt.title("P parameter B_update over training")
            plt.legend()
            plt.grid(True)
            # safe_legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            self.save_figures([(fig_p, "P_B_update_over_time")], experiment_folder)
            plt.close(fig_p)

            # Plot the RNN mean
            fig_nn = plt.figure()
            plt.plot(mean_mag, "o-", label="mean abs(NN_B_update)")
            plt.xlabel("Update iteration")
            plt.ylabel("Mean absolute B_update")
            plt.title("RNN mean across RNN params B_update magnitude over training")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(fig_nn, "NN_mean_B_update_over_time")], experiment_folder)
            plt.close(fig_nn)
            
        def plot_spectral_radius(self, experiment_folder):
            if not self.spectral_radii_hist:
                return
            arr = np.asarray(self.spectral_radii_hist)  # shape: (updates, num_recurrent_layers)
            fig = plt.figure()
            for j in range(arr.shape[1]):
                plt.plot(arr[:, j], "o-", label=rf"$\rho(W_{{hh,{j}}})$")
            plt.axhline(1, linestyle="--", label="target ρ")
            plt.xlabel("Update iteration")
            plt.ylabel("Spectral radius")
            plt.title("Recurrent spectral radii over training")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(fig, "spectral_radius_over_time.svg")], experiment_folder)
            # plt.close(fig)    
        
        def ADAM(self, iteration, gradient, exp_avg, exp_avg_sq,
            learning_rate, beta1, beta2, eps = 1e-8): 
            """
            Computes the update's change according to Adam algorithm.
            
            Args:
                iteration (int): current iteration number (aka how many times ADAM has been called).
                gradient (array-like): raw gradient vector (delta theta).
                exp_avg (np.ndarray): running first moment estimate (EMA of the gradient; tracks direction).
                exp_avg_sq (np.ndarray): running second moment estimate (EMA of squared gradient; tracks magnitude).
                learning_rate (float or array-like): base step size.
                beta1 (float): decay rate for the first moment (e.g. 0.9).
                beta2 (float): decay rate for the second moment (e.g. 0.999).
                eps (float): small constant to avoid division by zero.

            Returns:
                dtheta (np.ndarray): the computed parameter increment used for update (delta theta)
                exp_avg (np.ndarray): updated first moment estimate.
                exp_avg_sq (np.ndarray): updated second moment estimate.
            """
            gradient = np.asarray(gradient).flatten()

            exp_avg = beta1 * exp_avg + (1 - beta1) * gradient
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * np.square(gradient)

            bias_correction1 = 1 - beta1**iteration                
            bias_correction2 = 1 - beta2**iteration

            step_size = learning_rate / bias_correction1
            bias_correction2_sqrt = np.sqrt(bias_correction2)

            denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
            
            dtheta = -step_size * (exp_avg / denom)
            return dtheta, exp_avg, exp_avg_sq

        def update_learning_rate(self, current_stage_cost, params):
            """
            Update the learning rate based on the current stage cost metric.
            """

            if current_stage_cost < self.best_stage_cost:
                self.best_params = params.copy() 
                self.best_stage_cost = current_stage_cost
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience_threshold:
                old_alpha = self.alpha
                self.alpha *= self.lr_decay_factor  # decay 
                print(f"Learning rate decreased from {old_alpha} to {self.alpha} due to no stage cost improvement.")
                self.current_patience = 0  # reset 
                params = self.best_params  # revert to best params

            return params

        def noise_scale_by_distance(self, x, y, max_radius=1.0): #maxradius was 2
            
            
            """
            Compute a scaling factor for exploration noise based on distance from the origin. 
            Close to the origin, noise is scaled down; at max_radius, it is 1.

            Args:
                x (float): current x positon of the system.
                y (float): current y position of the system.
                max_radius (float): distance beyond which noise scaling caps at 1.

            Returns:
                float: a factor in [0, 1] by which to multiply noise.
            """
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)**2
            
        def check_whh_spectral_radii(self, params_rnn_list):
            """
            Check and print the spectral radii of the recurrent weight matrices in the RNN.

            Args:
                params_rnn_list (_type_): _description_
            """
            
            self.layers_list
            L = len(self.layers_list) - 1            # total layers
            for i in range(L-1): #all but last layer
                Whh = params_rnn_list[3*i + 2]  # recurrent weights
                eigvals = np.linalg.eigvals(Whh)
                rho = np.max(np.abs(eigvals)).real 
                print(f"Hidden layer {i+1} recurrent weight matrix spectral radius: {rho:.4f}")
                
        
        def RNN_warmstart(self, params):
            """
            Perform a forward pass through the RNN to obtain the initial hidden state.

            Args:
                x0 (ns,):  
                    Initial state of the system.
                params (dict):  
                    Dictionary of system and RNN parameters.

            Returns:
                hidden_t0:  
                    Initial hidden-state vectors for the RNN layers.
            """
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  
            
            
            state, _ = self.env.reset(seed=self.seed, options={})
            self.obst_motion.reset()
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
            
            hidden_in = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
                    for i in range(len(self.mpc.rnn.layers_list)-2)
                    ]
            
            warmup_steps = 50  # number of warmup steps

            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])
            for _ in range(warmup_steps):
                _, _, hidden_in, _, _, _ = self.V_MPC(params=params, x=state, 
                                                              xpred_list=xpred_list, 
                                                              ypred_list=ypred_list, 
                                                              hidden_in=hidden_in)
                
                # state, _, done, _, _ = self.env.step(action)
                
                _ = self.obst_motion.step()
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                
            # x_t0 = np.array(X).flatten()[:self.rnn_input_size]
            # # print(f"RNN warmstart raw:{x_t0}")
            # x_t0 = self.mpc.rnn.normalization_z(x_t0)
            # # print(f"RNN warmstart normalized:{self.mpc.rnn.normalization_z(x_t0)}")
            # params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])    
            # # initial hidden states are zero    
            # *h0, _ = self.get_hidden_func(*h0, x_t0, *params_rnn)
            h0 = hidden_in
            
            return h0
            
            

        def V_MPC(self, params, x, xpred_list, ypred_list, hidden_in):
            """
            Solve the value-function MPC problem for the current state.

            Args:
                params (dict):  
                    Dictionary of system and RNN parameters
                x (ns,):  
                    Current state of the system.
                xpred_list (m*(horizon+1),):  
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):  
                    Predicted obstacle y-positions over the horizon.
                hidden_in:  
                    Current hidden-state vectors for the RNN layers.

            Returns:
                u_opt (na,):  
                    The first optimal control action.
                V_val (solution["f"]):  
                    The optimal value function V(x).
                hidden_t1 :  
                    Updated hidden states after one RNN forward pass.
            """
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  
                                  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound, 
                                  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.diag(params["Q"])#cs.reshape(params["Q"], -1, 1)
            R_flat = cs.diag(params["R"])#cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], 
                                                       P_diag, Q_flat, R_flat,  params["rnn_params"], 
                                                       xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_VMPC,
                lam_x0 = self.lam_x_prev_VMPC, # warm‐start multipliers on x‐bounds
                lam_g0 = self.lam_g_prev_VMPC, # warm‐start multipliers on g
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # calculate new hidden state of the RNN for V_MPC
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            U = cs.reshape(solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na * self.horizon], self.na, self.horizon)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list, U)
            x_t0 = flat_input[:self.rnn_input_size]
            # print(f"VMPC raw:{x_t0}")
            x_t0 = self.mpc.rnn.normalization_z(x_t0)
            # print(f"VMPC normalized:{self.mpc.rnn.normalization_z(x_t0)}")
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])
            *hidden_t1, alpha_list = self.get_hidden_func(*hidden_in, x_t0, *params_rnn) #alpha_list is list of outputs of NN
            
            # warmstart variables for next iteration
            self.x_prev_VMPC     = solution["x"]
            self.lam_x_prev_VMPC = solution["lam_x"]
            self.lam_g_prev_VMPC = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the evaluation stage cost)
            self.S_VMPC = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]
            plan_xy = np.array(X[:2, :]).T
            return u_opt, solution["f"], hidden_t1, alpha_list, plan_xy, X
        
        def V_MPC_rand(self, params, x, rand, xpred_list, ypred_list, hidden_in):
            """
            Solve the value-function MPC problem with injected randomness.

            This is identical to V_MPC, but includes a random noise term in the optimization
            to encourage exploration.

            Args:
                params (dict):
                    Dictionary of system and RNN parameters:
                x (ns,):
                    Current system state vector.
                rand (na,1):
                    Random noise vector added to first control action in MPC objective
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current RNN hidden-state from previous time step.

            Returns:
                u_opt (na,):
                    The first optimal control action (with randomness).
                hidden_t1 (list of MX):
                    Updated RNN hidden-state 
            """
            
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])
            

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.diag(params["Q"])#cs.reshape(params["Q"], -1, 1)
            R_flat = cs.diag(params["R"])#cs.reshape(params["R"], -1, 1)


            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], 
                                                              P_diag, Q_flat, R_flat, params["rnn_params"], 
                                                              rand, xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_VMPCrandom,
                lam_x0 = self.lam_x_prev_VMPCrandom,
                lam_g0 = self.lam_g_prev_VMPCrandom,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # calculate new hidden state of the RNN for V_MPC_rand
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            U = cs.reshape(solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na * self.horizon], self.na, self.horizon)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list, U)
            x_t0 = flat_input[:self.rnn_input_size]
            # print(f"VMPCrand raw:{x_t0}")
            x_t0=self.mpc.rnn.normalization_z(x_t0)
            # print(f"VMPCrand normalized:{x_t0}")
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])  
            *hidden_t1, alpha_list = self.get_hidden_func(*hidden_in, x_t0, *params_rnn) #alpha_list is list of outputs of NN
            
            # warmstart variables for next iteration
            self.x_prev_VMPCrandom = solution["x"]
            self.lam_x_prev_VMPCrandom = solution["lam_x"]
            self.lam_g_prev_VMPCrandom = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the RL stage cost)
            self.S_VMPC_rand = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]

            return u_opt, hidden_t1, alpha_list, X

        def Q_MPC(self, params, action, x, xpred_list, ypred_list, hidden_in):
            
            """"
            
            Solve the Q-value MPC problem for current state and current action.
            
            Similar to V_MPC, but includes the action in the optimization and computes the Q-value.
            
            Args:
                params (dict):
                    Dictionary of system and RNN parameters.
                action (na,):
                    Current control action vector.
                x (ns,):
                    Current state of the system.
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current hidden-state vectors for the RNN layers.
            Returns:
                x_opt (ns*(horizon+1),):
                    Optimal state trajectory over the horizon.
                Q_val (solution["f"]):
                    Optimal Q-value for the current state and action.
                lagrange_mult_g (solution["lam_g"]):
                    Lagrange multipliers for the constraints.
                lam_lbx (solution["lam_x"]):
                    Lagrange multipliers for the lower bounds on x.
                lam_ubx (solution["lam_x"]):
                    Lagrange multipliers for the upper bounds on x.
                lam_p (solution["lam_p"]):
                    Lagrange multipliers for the parameters.
                hidden_t1 (list of MX):
                    Updated hidden states after one RNN forward pass.
                    """
                     
            # Build input‐action bounds (note horizon−1 controls remain free after plugging in `action`)
            U_lower_bound = -CONSTRAINTS_U*np.ones(self.na * (self.horizon-1))
            U_upper_bound = CONSTRAINTS_U*np.ones(self.na * (self.horizon-1))

            #Assemble full lbx/ubx: [ x0; X(1…H); action; remaining U; slack ]
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, 
                                  np.asarray(action).flatten(), U_lower_bound,  
                                  np.zeros(self.mpc.rnn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound,
                                  np.asarray(action).flatten(), U_upper_bound, 
                                  np.inf*np.ones(self.mpc.rnn.obst.obstacle_num *self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.diag(params["Q"])#cs.reshape(params["Q"], -1, 1)
            R_flat = cs.diag(params["R"])#cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, 
                                                       Q_flat, R_flat, params["rnn_params"], 
                                                       xpred_list, ypred_list, *hidden_in),
                x0    = self.x_prev_QMPC,
                lam_x0 = self.lam_x_prev_QMPC,
                lam_g0 = self.lam_g_prev_QMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
           
            # Extract lagrange multipliers needed for the lagrangian:
            lagrange_mult_g = solution["lam_g"] 
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]
            
            # calculate new hidden state of the RNN for Q_MPC
            X = cs.reshape(solution["x"][:self.ns * (self.horizon+1)], self.ns, self.horizon + 1)
            U = cs.reshape(solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na * self.horizon], self.na, self.horizon)
            flat_input = self.flat_input_fn(X, xpred_list, ypred_list, U)
            x_t0 = flat_input[:self.rnn_input_size]
            x_t0 = self.mpc.rnn.normalization_z(x_t0)
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])  
            *hidden_t1, _ = self.get_hidden_func(*hidden_in, x_t0, *params_rnn) 
            
            # warmstart variables for next iteration
            self.x_prev_QMPC = solution["x"]
            self.lam_x_prev_QMPC = solution["lam_x"]
            self.lam_g_prev_QMPC = solution["lam_g"]
            
            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p, hidden_t1, X
            
        def stage_cost(self, action, state, S, hx):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            hx = np.array(hx)
            
            violations = np.clip(-hx, 0, None)
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action 
                + self.slack_penalty_RL_L1*(np.sum(S)/(self.horizon+self.mpc.rnn.obst.obstacle_num))
                + 0.5 * self.slack_penalty_RL_L2* (np.sum(S**2) / (self.horizon+self.mpc.rnn.obst.obstacle_num)) 
                + np.sum(self.violation_penalty*violations)
            )
        
        def evaluation_step(self, params, experiment_folder, episode_duration):
            """
            Run an evaluation episode using the current parameters and plot results.
            Args:
                params (dict):
                    Dictionary of system and RNN parameters.
                experiment_folder (str):
                    Path to the experiment folder where results will be saved.
                episode_duration (int):
                    Number of steps in the evaluation episode.
            """
            
            hidden_in = self.RNN_warmstart(params)
            
            
            state, _ = self.env.reset(seed=self.seed, options={})
            self.obst_motion.reset()
            
            states_eval = [state]
            actions_eval = []
            stage_cost_eval = []
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
            hx = [ float(hf(cs.DM(state), xpred_list[0:self.mpc.rnn.obst.obstacle_num], ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                             for hf in self.h_func_list ] 
            hx_list = [hx]
            
            #for RNN outputs
            alphas = []
            
            
            # hidden_in_VMPC = [cs.DM.zeros(self.mpc.rnn.layers_list[i+1], 1) 
            #         for i in range(len(self.mpc.rnn.layers_list)-2)
            #         ]
            
            obs_positions = [self.obst_motion.current_positions()]
            
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  
            
            
            slacks_eval = []   # will become shape (T, m, N)

            m = self.mpc.rnn.obst.obstacle_num
            N = self.horizon

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
                # If flat of length m*N --> reshape column-major
                flat = S_raw.reshape(-1)
                if flat.size == m * N:
                    return flat.reshape(m, N, order="F")
                raise ValueError(f"Unexpected slack shape {S_raw.shape}, cannot make (m={m}, N={N}).")
            
            

            plans_eval = []   # each element will be (N+1, 2)
            


            for i in range(episode_duration):
                action, _, hidden_in, alpha, plan_xy, _ = self.V_MPC(params=params, x=state, 
                                                              xpred_list=xpred_list, 
                                                              ypred_list=ypred_list, 
                                                              hidden_in=hidden_in)
                
                
                S_now_mN = unflatten_slack(self.S_VMPC, m, N)   # shape (m, N)
                slacks_eval.append(S_now_mN)  
                plans_eval.append(plan_xy)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED in EVALUATION")
                alphas.append(alpha)
                
                action = cs.fmin(cs.fmax(cs.DM(action), -CONSTRAINTS_U), CONSTRAINTS_U)
                stage_cost_eval.append(self.stage_cost(action, state, self.S_VMPC, hx))
                
                # print(f"evaluation step {i}, action: {action}, slack: {np.sum(5e4*self.S_VMPC)}")
                state, _, done, _, _ = self.env.step(action)
                states_eval.append(state)
                actions_eval.append(action)
                hx = [ float(hf(cs.DM(state), xpred_list[0:self.mpc.rnn.obst.obstacle_num], ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                             for hf in self.h_func_list ]
                hx_list.append(hx) 
                # hx_list.append(np.array([ float(hf(cs.DM(x), xpred_list[0:self.mpc.rnn.obst.obstacle_num], ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                #              for hf in self.h_func_list ]))

                _ = self.obst_motion.step()
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

                obs_positions.append(self.obst_motion.current_positions())

            states_eval = np.array(states_eval)
            actions_eval = np.array(actions_eval)
            stage_cost_eval = np.array(stage_cost_eval)
            stage_cost_eval = stage_cost_eval.reshape(-1)
            obs_positions = np.array(obs_positions) 
            hx_list = np.vstack(hx_list)
            alphas = np.array(alphas)
            slacks_eval = np.stack(slacks_eval, axis=0)
            plans_eval = np.array(plans_eval) 
            
            sum_stage_cost = np.sum(stage_cost_eval)
            print(f"Stage Cost: {sum_stage_cost}")

            figstates=plt.figure()
            plt.plot(
                states_eval[:, 0], states_eval[:, 1],
                "o-"
            )

            # Plot the obstacle
            for (cx, cy), r in zip(self.mpc.rnn.obst.positions, self.mpc.rnn.obst.radii):
                        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
            plt.gca().add_patch(circle)
            plt.xlim([-CONSTRAINTS_X[0], 0])
            plt.ylim([-CONSTRAINTS_X[1], 0])

            # Set labels and title
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(r"Trajectories")
            plt.legend()
            plt.axis("equal")
            plt.grid()
            self.save_figures([(figstates,
            f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")

            figactions=plt.figure()
            plt.plot(actions_eval[:, 0], "o-", label=r"Action 1")
            plt.plot(actions_eval[:, 1], "o-", label=r"Action 2")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Action")
            plt.title(r"Actions")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figactions,
            f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,"Evaluation")

            figstagecost=plt.figure()
            plt.plot(stage_cost_eval, "o-")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Cost")
            plt.title(r"Stage Cost")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figstagecost,
            f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")
            
            figsvelocity=plt.figure()
            plt.plot(states_eval[:, 2], "o-", label=r"Velocity x")
            plt.plot(states_eval[:, 3], "o-", label=r"Velocity y")    
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(r"Velocity Value")
            plt.title(r"Velocity Plot")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figsvelocity,
            f"velocity_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder, "Evaluation")
            
            for i in range(hx_list.shape[1]):
                fig_hi = plt.figure()
                plt.plot(hx_list[:,i], "o", label=rf"$h_{{{i+1}}}(x_k)$")
                plt.xlabel(r"Iteration $k$")
                plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
                plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
                plt.grid()
                self.save_figures([(fig_hi,
                            f"hx_obstacle_{i+1}_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                            experiment_folder, "Evaluation")
                
            # Alphas from RNN
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
            self.save_figures([(fig_alpha,
                        f"alpha_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                        experiment_folder, "Evaluation")
            
            T, m, N = slacks_eval.shape
            t_eval = np.arange(T)  # or your actual time vector

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

                self.save_figures(
                    [(fig_slack_i, f"slack_obs{oi+1}_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                    experiment_folder,
                    "Evaluation",
                )
                # plt.close(fig_slack_i)  
            
            target_folder = os.path.join(experiment_folder, "evaluation")
            out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost}.gif")

            T_pred = plans_eval.shape[0]

            
            out_gif = os.path.join(target_folder, f"system_and_obstaclewithobstpred_{self.eval_count}_SC_{sum_stage_cost}.gif")
            
            self.make_system_obstacle_animation_v3(
                states_eval[:T_pred],
                plans_eval,
                obs_positions[:T_pred],
                self.mpc.rnn.obst.radii,
                CONSTRAINTS_X[0],
                out_gif,
                trail_len=self.horizon,      # fade the tail to last H
                camera="follow",
                follow_width=1.0,             # view width around agent when following
                follow_height=1.0,
            )
            self.update_learning_rate(sum_stage_cost, params)

            self.eval_count += 1 
            
            self.stage_cost_valid.append(sum_stage_cost)

            return 
        
        def parameter_updates(self, params, B_update_avg):
            
            #TODO: Finish commenting and cleaning this funciton and onwards

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])
            Q_diag = cs.diag(params["Q"])
            R_diag = cs.diag(params["R"])


            theta_vector_num = cs.vertcat(params["rnn_params"])

            identity = np.eye(theta_vector_num.shape[0])

            alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]))

            
            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 
            
            # #SGD
            # dtheta = self.alpha * B_update_avg


            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, dtheta)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")


            
            # constrained update qp update
            solution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, -dtheta), 
                    lbg=cs.vertcat(-np.inf*np.ones(theta_vector_num.shape[0])),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                )
            stats = self.qp_solver.stats()

            # if the qp fails, do not update the parameters
            if stats["success"] == False:
                print("QP NOT SUCCEEDED")
                theta_vector_num = theta_vector_num
            else:
                theta_vector_num = theta_vector_num + solution["x"]


            params["rnn_params"] = theta_vector_num
            
            params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])
            self.check_whh_spectral_radii(params_rnn)
            rhos = []
            L = len(self.layers_list) - 1            # total layers
            for i in range(L-1):                      # recurrent layers only (all but last)
                Whh = np.array(params_rnn[3*i + 2])   # Wih, bih, Whh triplets → Whh index = 2 within triplet
                eig = np.linalg.eigvals(Whh)
                rhos.append(float(np.max(np.abs(eig)).real))
            self.spectral_radii_hist.append(rhos)

            return params
        

        def rl_trainingloop(self, episode_duration, num_episodes, replay_buffer, episode_updatefreq, experiment_folder):
    
            #for the for loop
            params = self.params_innit
            hidden_in = self.RNN_warmstart(params)
    
            #to store for plotting
            params_history_P = [self.params_innit["P"]]
            
            x, _ = self.env.reset(seed=self.seed, options={})
            # reset obstacle motion

            stage_cost_history = []
            sum_stage_cost_history = []
            TD_history = []
            TD_temp = []
            TD_episode = []
            B_update_history = []
            grad_temp = []
            obs_positions = [self.obst_motion.current_positions()]
            phi_list = []
            S_list = []
            S_list_VMPC = []
            lag_g_list = []
      
            B_update_buffer = deque(maxlen=replay_buffer)
            

            states = [(x)]

            actions = []
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
            
            hx =  [ float(hf(cs.DM(x), xpred_list[0:self.mpc.rnn.obst.obstacle_num], ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                             for hf in self.h_func_list ]
            
            hx_list = [hx]
            
            #for RNN outputs
            alphas = []

            #intialize
            k = 0
            self.error_happened = False
            self.eval_count = 1
            
            self.stage_cost_valid = []  

            
            for i in range(1,episode_duration*num_episodes):  
                  
                noise = self.noise_scalingfactor*self.noise_scale_by_distance(x[0],x[1])
                rand = noise * self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u, hidden_in, alpha, _ = self.V_MPC_rand(params=params, x=x, rand = rand, xpred_list=xpred_list, 
                                    ypred_list=ypred_list, hidden_in=hidden_in)
                u = cs.fmin(cs.fmax(cs.DM(u), -CONSTRAINTS_U), CONSTRAINTS_U)

                alphas.append(alpha)
                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    self.error_happened = True

     
                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _, _, _ = self.Q_MPC(params=params, 
                                                                                                   action=u, 
                                                                                                   x=x, 
                                                                                                   xpred_list=xpred_list,
                                                                                                   ypred_list=ypred_list,
                                                                                                hidden_in=hidden_in)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    self.error_happened = True

                S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1):]
                stage_cost = self.stage_cost(action=u,state=x, S=self.S_VMPC_rand, hx = hx)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)
                hx = [ float(hf(cs.DM(x), xpred_list[0:self.mpc.rnn.obst.obstacle_num], 
                                ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                             for hf in self.h_func_list ]
                hx_list.append(hx)


                _, Vcost, _, _, _, _ = self.V_MPC(params=params, 
                                                         x=x, 
                                                         xpred_list=xpred_list, 
                                                         ypred_list=ypred_list, 
                                                         hidden_in=hidden_in)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED")
                    self.error_happened = True

                # TD update
                TD = (stage_cost) + self.gamma*Vcost - Qcost

                U = solution[self.ns * (self.horizon+1):self.na * (self.horizon) + self.ns * (self.horizon+1)] 
                X = solution[:self.ns * (self.horizon+1)] 
                
                    
                params_rnn = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])     
                phi = self.phi_func(X, 
                                    U,
                                    cs.DM(xpred_list), 
                                    cs.DM(ypred_list), 
                                    *hidden_in, 
                                    *params_rnn)

                phi_list.append(np.array(phi).reshape(self.horizon, self.mpc.m))
                

                # calculate numeric jacobian of qlagrange
                qlagrange_numeric_jacob=  self.qlagrange_fn_jacob(
                    params["A"],
                    params["B"],
                    params["b"],
                    params["Q"],
                    params["R"],
                    params["P"],
                    lam_lbx,
                    lam_ubx,
                    lagrange_mult_g,
                    X, U, S, 
                    params["rnn_params"],
                    xpred_list, 
                    ypred_list,
                    *hidden_in 
                )

                # first order update
                #removed minus for just notatiojn wise
                B_update = TD*qlagrange_numeric_jacob
                grad_temp.append(qlagrange_numeric_jacob)
                B_update_buffer.append(B_update)
                        
                stage_cost_history.append(stage_cost)
                if self.error_happened == False:
                    TD_episode.append(TD)
                    TD_temp.append(TD)
                else:
                    TD_temp.append(cs.DM(np.nan))
                    self.error_happened = False
                    
                _ = self.obst_motion.step()
                
                obs_positions.append(self.obst_motion.current_positions())
        
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                
                if (k == episode_duration):                     
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:
                        
        
                        B_update_avg = np.mean(B_update_buffer, 0)
                        
                        
                        B_update_history.append(B_update_avg)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.mean(TD_episode))

                    stage_cost_history = []
                    TD_episode = []


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 50) == 0:
                        states = np.array(states)
                        actions = np.asarray(actions)
                        TD_temp = np.asarray(TD_temp) 
                        obs_positions = np.array(obs_positions)
                        hx_list = np.vstack(hx_list)
                        alphas = np.array(alphas)

                        figstate=plt.figure()
                        plt.plot(
                            states[:, 0], states[:, 1],
                            "o-"
                        )

                        # Plot the obstacle
                        for (cx, cy), r in zip(self.mpc.rnn.obst.positions, self.mpc.rnn.obst.radii):
                            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                            plt.gca().add_patch(circle)
                        plt.xlim([-CONSTRAINTS_X[0], 0])
                        plt.ylim([-CONSTRAINTS_X[1], 0])

                        # Set labels and title
                        plt.xlabel(r"$x$")
                        plt.ylabel(r"$y$")
                        plt.title(r"Trajectories of states while policy is trained$")
                        plt.legend()
                        plt.axis("equal")
                        plt.grid()
                        self.save_figures([(figstate,
                            f"position_plotat_{i}.svg")],
                            experiment_folder, "Learning")


                        figvelocity=plt.figure()
                        plt.plot(states[:, 2], "o-", label=r"Velocity x")
                        plt.plot(states[:, 3], "o-", label=r"Velocity y")    
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Velocity Value")
                        plt.title(r"Velocity Plot")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figvelocity,
                            f"figvelocity{i}.svg")],
                            experiment_folder, "Learning")

                        # Plot TD
                        indices = np.arange(len(TD_temp))
                        figtdtemp = plt.figure(figsize=(10, 5))
                        plt.scatter(indices,TD_temp, label=r"TD")
                        plt.yscale('log')
                        plt.title(r"TD Over Training (Log Scale) - Colored by Proximity")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"TD")
                        plt.legend()
                        plt.grid(True)
                        self.save_figures([(figtdtemp,
                            f"TD_plotat_{i}.svg")],
                            experiment_folder, "Learning")

                        figactions=plt.figure()
                        plt.plot(actions[:, 0], "o-", label=r"Action 1")
                        plt.plot(actions[:, 1], "o-", label=r"Action 2")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Action")
                        plt.title(r"Actions")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figactions,
                            f"action_plotat_{i}.svg")],
                            experiment_folder, "Learning")

                        gradst = np.asarray(grad_temp)
                        gradst = gradst.squeeze(-1)


                        labels = [f"P[{i},{i}]" for i in range(4)]
                        nn_grads = gradst[:, 4:]
                        # take mean across rows (7,205) --> (7,)
                        mean_mag = np.mean(np.abs(nn_grads), axis=1)

                        P_figgrad = plt.figure()
                        
                        for idx, lbl in enumerate(labels):
                                plt.plot(gradst[:, idx], "o-", label=lbl)
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"P gradient")
                        plt.title(r"P parameter gradients over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(P_figgrad,
                            f"P_grad_plotat_{i}.svg")],
                            experiment_folder, "Learning")



                        NN_figgrad = plt.figure()
                        plt.plot(mean_mag, "o-", label=r"mean abs(rnn grad)")

                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"rnn Mean absolute gradient")
                        plt.title(r"rnn mean acoss rnn params gradient magnitude over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(NN_figgrad,
                            f"RNN_grad_plotat_{i}.svg")],
                            experiment_folder, "Learning")
                        # plt.show()
                        
                        
                        for i in range(hx_list.shape[1]):
                            fig_hi = plt.figure()
                            plt.plot(hx_list[:,i], "o", label=rf"$h_{{{i+1}}}(x_k)$")
                            plt.xlabel(r"Iteration $k$")
                            plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
                            plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
                            plt.grid()
                            self.save_figures([(fig_hi,
                                        f"hx_obstacle_plotat_{i}.svg")],
                                        experiment_folder, "Learning")
                            
                        # Alphas from RNN
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
                        self.save_figures([(fig_alpha,
                                    f"alpha_plotat_{i}.svg")],
                                    experiment_folder, "Learning")

                        target_folder = os.path.join(experiment_folder, "learning_process")
                        out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost_history[-1]}.gif")
                        self. make_system_obstacle_animation(
                        states,
                        obs_positions,
                        self.mpc.rnn.obst.radii,
                        CONSTRAINTS_X[0],
                        out_gif,
                        )
                        
                        # #evaluation step
                        self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
            
                    # self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)

                    # reset the environment and the obstacle motion
                    
                    hidden_in = self.RNN_warmstart(params)
                    x, _ = self.env.reset(seed=self.seed, options={})
                    self.obst_motion.reset()
                    k=0
                    
                    
                    self.x_prev_VMPC        = cs.DM()  
                    self.lam_x_prev_VMPC    = cs.DM()  
                    self.lam_g_prev_VMPC    = cs.DM()  

                    self.x_prev_QMPC        = cs.DM()  
                    self.lam_x_prev_QMPC    = cs.DM()  
                    self.lam_g_prev_QMPC    = cs.DM()  

                    self.x_prev_VMPCrandom  = cs.DM()  
                    self.lam_x_prev_VMPCrandom = cs.DM()  
                    self.lam_g_prev_VMPCrandom = cs.DM()
            
                    states = [(x)]
                    TD_temp = []
                    actions = []
                    grad_temp = []
                    obs_positions = [self.obst_motion.current_positions()]
                    xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                    phi_list = []
                    S_list = []
                    S_list_VMPC = []
                    lag_g_list = []
                    
                    
                    hx =  [ float(hf(cs.DM(x), xpred_list[0:self.mpc.rnn.obst.obstacle_num], ypred_list[0:self.mpc.rnn.obst.obstacle_num])) 
                             for hf in self.h_func_list ]
            
                    hx_list = [hx]
                    #for RNN outputs
                    alphas = []
                    
                    print("reset")
                    
                # k counter    
                k+=1
                
                #counter
                if i % 1000 == 0:
                    print(f"{i}/{episode_duration*num_episodes}")  

            #show trajectories
            # plt.show()
            # plt.close()

            params_history_P = np.asarray(params_history_P)

            TD_history = np.asarray(TD_history)
            sum_stage_cost_history = np.asarray(sum_stage_cost_history)

            self.plot_B_update(B_update_history, experiment_folder)
       

            figP = plt.figure(figsize=(10, 5))
            plt.plot(params_history_P[:, 0, 0], label=r"$P_{1,1}$")
            plt.plot(params_history_P[:, 1, 1], label=r"$P_{2,2}$")
            plt.plot(params_history_P[:, 2, 2], label=r"$P_{3,3}$")
            plt.plot(params_history_P[:, 3, 3], label=r"$P_{4,4}$")
            # plt.title("Parameter: P",        fontsize=24)
            plt.xlabel(r"Update Number",     fontsize=20)
            plt.ylabel(r"Value",             fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figP,
                   f"P.svg")],
                 experiment_folder)
            

            figstagecost = plt.figure()
            plt.plot(sum_stage_cost_history, 'o', label=r"Stage Cost")
            plt.yscale('log')
            # plt.title("Stage Cost Over Training (Log Scale)", fontsize=24)
            plt.xlabel(r"Episode Number",                    fontsize=20)
            plt.ylabel(r"Stage Cost",                        fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figstagecost,
                   f"stagecost.svg")],
                 experiment_folder)
            

            figtd = plt.figure()
            plt.plot(TD_history, 'o', label=r"TD")
            plt.yscale('log')
            plt.title(r"TD Over Training (Log Scale)")
            plt.xlabel(r"Episode Number")
            plt.ylabel(r"TD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figtd,
                   f"TD.svg")],
                 experiment_folder)

            cost = np.array(sum_stage_cost_history)
            episodes = np.arange(len(cost))

            # choose window size
            window = 100

            # wrap in a pandas Series so rolling().mean()/.std() produce full‐length outputs
            s = pd.Series(cost)
            running_mean = s.rolling(window, center=True, min_periods=1).mean().values
            running_std  = s.rolling(window, center=True, min_periods=1).std().values

            # plt
            figstagecost_nice = plt.figure(figsize=(10,5))
            ax = figstagecost_nice.add_subplot(1,1,1)

            # running mean
            ax.plot(episodes, running_mean, '-', linewidth=2, label=rf"Stage Cost mean ({window}-ep)")

            # ±1σ band
            ax.fill_between(episodes,
                            running_mean - running_std,
                            running_mean + running_std,
                            alpha=0.3,
                            label=rf"Stage Cost std ({window}-ep)")

            if np.any(cost > 0):
                ax.set_yscale('log')

            ax.set_xlabel(r"Episode Number", fontsize=20)
            ax.set_ylabel(r"Stage Cost",     fontsize=20)
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.legend(fontsize=16)
            figstagecost_nice.tight_layout()
            self.save_figures([(figstagecost_nice,
                   f"stagecost_smoothed.svg")],
                 experiment_folder)
            
            self.plot_spectral_radius(experiment_folder)
            
            npz_payload = {
            "episodes": episodes,
            "stage_cost": cost,
            "td": TD_history,
            "running_mean": running_mean,
            "running_std": running_std,
            "smoothing_window": np.array([window], dtype=int),
            "params_history_P": params_history_P,
            "spectral_radii_hist": np.array(self.spectral_radii_hist),
            "stage_cost_valid": self.stage_cost_valid,
            }
            
            np.savez_compressed(os.path.join(experiment_folder, "training_data.npz"), **npz_payload)

            return self.best_params
        
        