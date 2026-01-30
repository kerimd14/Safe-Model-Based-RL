
import numpy as np
import os # to communicate with the operating system
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
from Classes import MPC, ObstacleMotion
from matplotlib.colors import Normalize, PowerNorm
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm
from config import SAMPLING_TIME, SEED, NUM_STATES, NUM_INPUTS, CONSTRAINTS_X, CONSTRAINTS_U
from matplotlib import rcParams
import matplotlib.colors as mcolors

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
                

def MPC_func(x, mpc, params, solver_inst, xpred_list, ypred_list, x_prev, lam_x_prev, lam_g_prev):
        
        fwd_func = mpc.nn.numerical_forward()
        
        alpha = []
        h_func_list = [h_func for h_func in mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
        alpha.append(cs.DM(fwd_func(x, h_func_list,  params["nn_params"])))

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

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], V, P_diag, Q_flat, R_flat, xpred_list, ypred_list),
            x0    = x_prev,
            lam_x0 = lam_x_prev,
            lam_g0 = lam_g_prev,
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg= lbg,
        )

        g_resid = solution["g"][4:]        # vector of all g(x)

        print(f"g reid: {g_resid}")

        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]
        X_plan = cs.reshape(solution["x"][:mpc.ns * (mpc.horizon+1)], mpc.ns, mpc.horizon + 1)
        plan_xy = np.array(X_plan[:2, :]).T
        
        #warm start variables
        x_prev = solution["x"]
        lam_x_prev = solution["lam_x"]
        lam_g_prev= solution["lam_g"]

        return u_opt, solution["f"], alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, plan_xy

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
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("system + Moving Obstacles")

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
    k_box: bool = True, k_fontsize: int = 25, k_fmt: str = "k={k}",

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

    # ---- caps for this run ----
    trail_len: int | None = None,   # if None → 6 (last steps)
    obs_lookahead: int = 6,         # predicted obstacles shown H steps ahead
    pred_cap: int = 100,              # predicted path segments capped to 6

    # ---- output ----
    out_path: str | None = None, dpi: int = 200,
):
    """
    Variant that:
      • Shows ONLY the last 6 steps of the system trail (override with trail_len).
      • Draws the prediction horizon but caps it to 6 steps (pred_cap).
      • Draws predicted obstacles H steps ahead (obs_lookahead, default 6).
      • Legend shows exact numbers (no ≤).
    """
    import os, numpy as np, matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    from matplotlib.collections import LineCollection

    # ---------- harmonize ----------
    T = min(states_eval.shape[0], pred_paths.shape[0], obs_positions.shape[0])
    system_xy, pred_paths, obs_positions = states_eval[:T, :2], pred_paths[:T], obs_positions[:T]
    if not frame_indices:
        raise ValueError("frame_indices is empty.")
    for k in frame_indices:
        if not (0 <= k < T):
            raise ValueError(f"frame index {k} out of range [0,{T-1}]")

    # True model horizon per-step (pred_paths is length N+1 per k); cap to pred_cap
    N_true = max(0, pred_paths.shape[1] - 1)
    N = min(N_true, int(pred_cap))

    m = obs_positions.shape[1]
    trail_N = trail_len if trail_len is not None else 6  # last 6 steps

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
        *[mlines.Line2D([], [], color=obst_cols[i], lw=2, label=f"Obstacle {i+1}") for i in range(m)],
        mlines.Line2D([], [], color=obst_cols[0], lw=1.2, ls="--", alpha=0.3,
                      label=f"Obstacle (Predicted, {int(obs_lookahead)})"),
    ]

    # legend font size (auto + gentle reduction)
    legend_fs_eff = legend_fontsize * (scale if legend_auto_scale else 1.0) * legend_scale_factor

    # ---------- helpers ----------
    def _trail_window(k, trail_len_local=trail_N):
        s = max(0, k - max(0, int(trail_len_local))); return s, k + 1

    pred_alpha_seq = np.linspace(1.0, 0.35, max(N, 1))

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

        # trail (last 6 default)
        s0, e0 = _trail_window(k); tail_xy = system_xy[s0:e0]
        ax.plot(tail_xy[:,0], tail_xy[:,1], "-", lw=2, color=sys_rgb, zorder=2.0)
        if len(tail_xy) > 1:
            alphas = np.linspace(0.3, 1.0, len(tail_xy)-1)
            for p, a in zip(tail_xy[:-1], alphas):
                ax.plot(p[0], p[1], "o", ms=4.5, color=(*sys_rgb, a), zorder=2.1, markeredgewidth=0)

        # system point
        ax.plot(system_xy[k,0], system_xy[k,1], "o", ms=7, color="red", zorder=5.0)

        # predicted path (cap to N=pred_cap)
        if N > 0:
            ph_full = pred_paths[k]                # shape (N_true+1, 2)
            ph = ph_full[: N + 1, :]               # take up to N steps ahead + current
            future = ph[1:, :]
            poly = np.vstack((ph[0:1, :], future))
            segs = np.stack([poly[:-1], poly[1:]], axis=1)
            lc = LineCollection(segs, linewidths=2, zorder=2.2)
            seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
            seg_cols[:, 3] = pred_alpha_seq
            lc.set_colors(seg_cols); ax.add_collection(lc)
            for j in range(min(N, future.shape[0])):
                ax.plot(future[j,0], future[j,1], "o", ms=5, color=pred_rgb, zorder=2.3)

        # obstacles (current)
        for i, r_i in enumerate(radii):
            cx, cy = obs_positions[k, i]
            ax.add_patch(plt.Circle((cx, cy), r_i, fill=False, color=obst_cols[i], lw=2, zorder=1.0))

        # predicted obstacles only (H steps)
        H = int(obs_lookahead)
        if H > 0:
            for h in range(1, H + 1):
                t = min(k + h, T - 1)
                a = np.interp(h, [1, max(H, 1)], [0.35, 0.30])  # gentle fade
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
        ax.axis("off"); ax.set_visible(False)

    # ---------- outer-only post-pass (skip empty cells) ----------
    if (not hide_axis_labels and label_outer_only) or (not hide_ticks and ticklabels_outer_only):
        for r in range(rows):
            for c in range(cols):
                idx = r*cols + c
                if idx >= filled_count:
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
        leg_ax.set_visible(True)
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center",
                      framealpha=0.95, fontsize=legend_fs_eff,
                      borderaxespad=legend_borderaxespad, borderpad=legend_borderpad)
    else:
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.95, fontsize=legend_fs_eff,
                   borderaxespad=legend_borderaxespad, borderpad=legend_borderpad)
        fig.subplots_adjust(right=0.86)

    # ---------- save ----------
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig, axs

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
    
    obst_motion.reset()  # reset the obstacle motion to initial positions

   
    state, _ = env.reset(seed=SEED, options={})
    states = [state]
    actions = []
    stage_cost = []
    g_resid_lst = [] 
    lam_g_hist = []   
    
    # extract list of h functions
    h_func_list = mpc.nn.obst.make_h_functions()

    alphas = []
    

    xpred_list, ypred_list = obst_motion.predict_states(horizon)

    hx = [ np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]) ]


    solver_inst = mpc.MPC_solver_noslack() 
    
    plans = []
    
    #for plotting the moving obstacle
    obs_positions = [obst_motion.current_positions()]
    
    x_prev, lam_x_prev, lam_g_prev = cs.DM(), cs.DM(), cs.DM()  # initialize warm start variables

    for i in range(episode_duration):

        action, _, alpha, g_resid, x_prev, lam_x_prev, lam_g_prev, plan_xy = MPC_func(state, mpc, params, solver_inst, xpred_list, ypred_list, x_prev, lam_x_prev, lam_g_prev)

        alphas.append(alpha)
        plans.append(plan_xy)


        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        print(f"state from env: {state}")
        states.append(state)
        actions.append(action)
        g_resid_lst.append(-g_resid)
        
        arr = np.array(lam_g_prev).flatten()
    
        lam_g_hist.append(arr)

        stage_cost.append(stage_cost_func(action, state))

        print(i)
        
        #object moves
        _ = obst_motion.step()
        
        xpred_list, ypred_list = obst_motion.predict_states(horizon)
        
        hx.append(np.array([ float(hf(cs.DM(state), xpred_list[0:mpc.nn.obst.obstacle_num], ypred_list[0:mpc.nn.obst.obstacle_num])) for hf in h_func_list ]))
        
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
    
    print(f"obs_positions: {obs_positions}")
    
    print(f"state_positions: {states}")

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
    suffix = 'after' if after_updates else 'before'
    cols = [f"lam_g_{i}" for i in range(lam_g_hist.shape[1])]
    df = pd.DataFrame(lam_g_hist, columns=cols)


    df = df.round(3)


    table_str = df.to_string(index=False)

    txt_path = os.path.join(experiment_folder, f"lam_g_prev_{suffix}.txt")
    with open(txt_path, 'w') as f:
        f.write(table_str)

    fig_states = plt.figure()
    plt.plot(states[:, 0], states[:, 1], "o-", label="trajectory")
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
        plt.gca().add_patch(circle)
    plt.xlim([-CONSTRAINTS_X[0], 0.1*CONSTRAINTS_X[0]])
    plt.ylim([-CONSTRAINTS_X[1], 0.1*CONSTRAINTS_X[1]])
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
        hx_figs.append((fig_hi, f"hx_obstacle_{i+1}_{'after' if after_updates else 'before'}.png"))

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
        hx_col_figs.append((fig_hi_col, f"hx_colored_obstacle_{i+1}_{'after' if after_updates else 'before'}.png"))

    fig8 = plt.figure()
    plt.plot(states[:,0], states[:,1], color='gray', alpha=0.5)
    sc1 = plt.scatter(states[:,0], states[:,1], c=iters, cmap=cmap, norm=norm, s=40)
    cb1 = plt.colorbar(sc1, label='Time Step $k$')   # grab the Colorbar
    cb1.set_label('Time Step $k$', fontsize=16)       # label font size
    cb1.ax.tick_params(labelsize=12)    
    circle = plt.Circle((-2, -2.25), 1.5, color='k', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5,0]); plt.ylim([-5,0])
    plt.xlabel('$X$',fontsize=20); plt.ylabel('$Y$', fontsize=20)
    # plt.title('Trajectory Colored by Iteration')
    plt.axis('equal'); plt.grid(); plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    # Save Figures
    
    fig_dyn = plot_traj_colored_only_in_region(
    states=states,
    obs_positions=obs_positions,
    radii=radii,
    modes=modes,
    xlim=(-CONSTRAINTS_X[0]-1.0, 0.5*CONSTRAINTS_X[0]),
    ylim=(-CONSTRAINTS_X[1]-0.5, 0.5*CONSTRAINTS_X[1]),
    rect_alpha=0.10,
    cmap_name="turbo",
    gamma=0.9,
    alpha_min=0.2,
    alpha_max=1.0,
    influence_dist=0.7,
    draw_moving_circles=True
)



    figs_to_save = [
        (fig_states,    f"states_trajectory_{'after' if after_updates else 'before'}.png"),
        (fig_actions,   f"actions_{'after' if after_updates else 'before'}.png"),
        (fig_stagecost, f"stagecost_{'after' if after_updates else 'before'}.png"),
        (fig_alpha,     f"alpha_{'after' if after_updates else 'before'}.png"),
        (fig_velocity,  f"velocity_{'after' if after_updates else 'before'}.png"),
        (fig8, f"states_colored_MPCregular_{'afterupdates' if after_updates else 'beforeupdates'}.svg")
        (fig_dyn,       f"traj_dynamic_{'afterupdates' if after_updates else 'beforeupdates'}.svg")
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

    T_pred = plans.shape[0]
    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")
    print(f"Stage Cost: {stage_cost.sum():.3f}")
    
    suffix   = "after" if after_updates else "before"
    svg_dir  = os.path.join(experiment_folder, f"snapshots_{suffix}")
    os.makedirs(svg_dir, exist_ok=True)
     
    # keep lengths consistent with your GIFs
    
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
    states[:T_pred],plans, obs_positions[:T_pred], radii, CONSTRAINTS_X[0],
    frame_indices=[5, 10, 13, 14, 15, 16, 20, 25],
    grid=(3, 3), 
    use_empty_cell_for_legend=True,
    label_outer_only=True,          # <- only borders labeled
    ticklabels_outer_only=True,
    legend_auto_scale=True,
    legend_scale_factor=0.8,
    axis_labelsize=35, tick_fontsize=20, axis_labelpad_xy=(24,24),
    k_fontsize=25,
    figsize_per_ax=(5.0, 5.0),
    auto_enlarge_when_outer_only=2,   # make panels bigger
    gaps_outer_only=(0.01, 0.01),        # tighter gaps
    k_annotation="inside", k_loc="upper left", k_fmt="k={k}",
    figscale=1, 
    out_path=os.path.join(svg_dir, f"NN_snapshots_{'after' if after_updates else 'before'}.svg"), dpi=500  # or .png
    
    )

    return stage_cost.sum()
    
def noise_scale_by_distance(x, y, max_radius=3):
            # i might remove this because it doesnt allow for exploration of the last states which is important
            
            
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)**2
            


def plot_traj_colored_only_in_region(
    states, obs_positions, radii, modes,
    xlim=(-6, 1), ylim=(-5.5, 1.0),
    region="auto", shrink=0.92,
    draw_region=True,
    region_label="Region where moving footprints are drawn",
    rect_alpha=0.10,
    cmap_name="turbo",
    gamma=0.9,
    milestone_every=5,
    draw_moving_circles=True,
    # binary opacity controls
    alpha_min=0.2,
    alpha_max=1.0,
    influence_dist=0.3,
    # arrow styling
    arrow_len_scale=0.6,    # arrow length as multiple of obstacle radius
    arrow_offset=0.08,      # start arrow just outside circle: r + offset
    out_path=None,
):
    T = states.shape[0]
    k = np.arange(T)
    pos = states[:, :2]

    # --- region (auto from movers) ---
    mv_idx = [i for i, md in enumerate(modes) if md.lower() != "static"]
    if region == "auto" and len(mv_idx) > 0:
        centers = obs_positions[:, mv_idx, :]
        r_mv    = np.array([radii[i] for i in mv_idx])[None]
        xmin = np.min(centers[...,0] - r_mv); xmax = np.max(centers[...,0] + r_mv)
        ymin = np.min(centers[...,1] - r_mv); ymax = np.max(centers[...,1] + r_mv)
        cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
        wx, wy = (xmax-xmin)*shrink, (ymax-ymin)*shrink
        xmin, xmax = cx - wx/2, cx + wx/2
        ymin, ymax = cy - wy/2, cy + wy/2
    elif region != "auto":
        (xmin, xmax), (ymin, ymax) = region
    else:
        (xmin, xmax), (ymin, ymax) = (xlim, ylim)

    inside  = (pos[:,0] >= xmin) & (pos[:,0] <= xmax) & (pos[:,1] >= ymin) & (pos[:,1] <= ymax)
    outside = ~inside

    # --- shared colormap/norm (only use k inside for scaling) ---
    cmap = cm.get_cmap(cmap_name)
    if np.any(inside):
        ki = k[inside]
        vmin, vmax = float(ki.min()), float(ki.max())
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if gamma else Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=0, vmax=max(1, T-1))

    # --- proximity → binary alpha ---
    if len(mv_idx) > 0:
        centers_mv = obs_positions[:, mv_idx, :]          # (T,M,2)
        radii_mv   = np.array([radii[i] for i in mv_idx]) # (M,)
        diff       = pos[:, None, :] - centers_mv         # (T,M,2)
        dist       = np.linalg.norm(diff, axis=2)         # (T,M)
        clearance  = dist - radii_mv[None, :]             # (T,M)
        near       = clearance <= influence_dist          # (T,M) bool
        alpha_ci   = np.where(near, alpha_max, alpha_min) # per (t, obstacle)
        near_any   = near.any(axis=1)                     # per t
        alpha_traj = np.where(near_any, alpha_max, alpha_min)
    else:
        alpha_ci   = None
        near_any   = np.zeros(T, dtype=bool)
        alpha_traj = np.full(T, alpha_min)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.8, 6.6))

    # region rectangle
    legend_handles = []
    if draw_region:
        rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                         facecolor=(0.6, 0.6, 0.6, rect_alpha),
                         edgecolor="0.35", linestyle='--', linewidth=1.4,
                         zorder=0, label=region_label)
        ax.add_patch(rect)
        legend_handles.append(rect)

    # base path + outside points (neutral)
    ax.plot(pos[:,0], pos[:,1], color="0.65", alpha=0.8, lw=1.6, zorder=1)
    if np.any(outside):
        ax.scatter(pos[outside,0], pos[outside,1], c="0.65", s=38, edgecolor="none", zorder=2)

    # inside points (colored; per-point alpha; black outline when near any)
    if np.any(inside):
        ki = k[inside]
        sc = ax.scatter(pos[inside,0], pos[inside,1],
                        c=ki, cmap=cmap, norm=norm, s=48,
                        edgecolor="none", zorder=3)
        cols = cmap(norm(ki))
        cols[:, 3] = np.clip(alpha_traj[inside], 0.0, 1.0)
        sc.set_facecolors(cols)

        # black outline ONLY for "near" points
        near_pts = inside & near_any
        if np.any(near_pts):
            cols_near = cmap(norm(k[near_pts])); cols_near[:, 3] = np.clip(alpha_traj[near_pts], 0.0, 1.0)
            ax.scatter(pos[near_pts,0], pos[near_pts,1],
                       color=cols_near, s=48, edgecolors="black", linewidths=0.9, zorder=3.2)

        if milestone_every and milestone_every > 0:
            mk = inside & (k % milestone_every == 0)
            if np.any(mk):
                cols_mk = cmap(norm(k[mk])); cols_mk[:, 3] = np.clip(alpha_traj[mk], 0.0, 1.0)
                ax.scatter(pos[mk,0], pos[mk,1], color=cols_mk, s=95, edgecolor="none", zorder=3.1)

        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Iteration $k$", fontsize=14)
        cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 6).round().astype(int))

    # static obstacles (black)
    for i, md in enumerate(modes):
        if md.lower() == "static":
            cx0, cy0 = obs_positions[0, i]
            ax.add_patch(plt.Circle((cx0, cy0), radii[i], fill=False, color="k", lw=2, zorder=2))
            legend_handles.append(Line2D([0],[0], marker='o', lw=0,
                                         markerfacecolor='none', markeredgecolor='k',
                                         markersize=8, label="Static obstacle"))

    # moving obstacles (solid circle + direction arrow), drawn only when agent is inside region
    if draw_moving_circles and len(mv_idx) > 0 and np.any(inside):
        ks_show = np.nonzero(inside)[0]
        for j, i in enumerate(mv_idx):
            r = radii[i]
            for t in ks_show:
                cx, cy = obs_positions[t, i]
                # circle color & alpha (binary by proximity)
                col = list(cmap(norm(t)));  col[3] = float(np.clip(alpha_ci[t, j], 0.0, 1.0))

                # draw circle; add black outline if "near"
                if alpha_ci[t, j] >= alpha_max - 1e-12:
                    ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                            linestyle='-', linewidth=2.6,
                                            edgecolor='k', alpha=1.0, zorder=1.45))
                    ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                            linestyle='-', linewidth=1.8,
                                            edgecolor=col, alpha=col[3], zorder=1.6))
                else:
                    ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                            linestyle='-', linewidth=1.6,
                                            edgecolor=col, alpha=col[3], zorder=1.5))

                # ---- direction arrow (velocity from t to t+1 or t-1) ----
                t2 = t+1 if t < T-1 else t-1
                vx, vy = obs_positions[t2, i] - obs_positions[t, i]
                vnorm = np.hypot(vx, vy)
                if vnorm > 1e-9:
                    ux, uy = vx / vnorm, vy / vnorm
                    # place arrow just outside the circle, pointing along motion
                    L  = arrow_len_scale * r
                    sx = cx + (r + arrow_offset) * ux
                    sy = cy + (r + arrow_offset) * uy
                    ex = sx + L * ux
                    ey = sy + L * uy

                    # black underlay if near
                    if alpha_ci[t, j] >= alpha_max - 1e-12:
                        arrow_k = FancyArrowPatch((sx, sy), (ex, ey),
                                                  arrowstyle='-|>', mutation_scale=10 + 8*r,
                                                  linewidth=2.6, color='k', alpha=1.0, zorder=1.65)
                        ax.add_patch(arrow_k)

                    arrow_col = list(col);  # same RGBA as circle
                    arrow = FancyArrowPatch((sx, sy), (ex, ey),
                                            arrowstyle='-|>', mutation_scale=10 + 8*r,
                                            linewidth=1.8, color=arrow_col, alpha=arrow_col[3],
                                            zorder=1.7)
                    ax.add_patch(arrow)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=10, frameon=True)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.5)
    ax.set_xlabel("$X$", fontsize=18); ax.set_ylabel("$Y$", fontsize=18)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig

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
