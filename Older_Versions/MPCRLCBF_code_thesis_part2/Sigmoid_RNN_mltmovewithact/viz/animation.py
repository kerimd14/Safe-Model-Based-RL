# viz/animation.py
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, PowerNorm
from matplotlib import rcParams


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

def make_system_obstacle_animation_v2(
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

    # --- predicted obstacle outlines for the next N steps (ghosted) ---
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
        half_w = follow_width  / 2.0 + (max(radii) if len(radii) else 0.0)
        half_h = follow_height / 2.0 + (max(radii) if len(radii) else 0.0)
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

        # predicted obstacle outlines at k+1..k+N
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
        
        
def plot_traj_segments(
        states,                 # (T, ns) with first two columns = XY
        obs_positions,          # (T, n_obs, 2)
        radii,                  # list/array len n_obs
        modes,                  # list len n_obs, e.g. ["static","moving",...]
        steps_per_fig=60,       # number of timesteps per figure
        start_k=0,              # start index (inclusive)
        end_k=None,             # end index (exclusive). None -> T

        # view settings
        xlim=(-6, 1), ylim=(-5.5, 1.0),
        moving_view=False,      # auto-zoom around current segment
        view_pad=0.5,           # padding (in units) around segment bounds

        # coloring
        cmap_name="turbo",
        custom_cmap=None,       # pass a Colormap to override cmap_name
        gamma=0.9,              # None or float for PowerNorm

        # past trajectory (grey)
        draw_past_as_grey=True,
        past_color="0.65", past_alpha=0.9, past_lw=1.6, past_ms=38,

        # drawing toggles
        draw_path=True,         # draw line segments
        marker_size=48,
        milestone_every=0,      # 0/None disables milestone markers

        # moving obstacles
        draw_moving_circles=True,
        arrow_len_scale=0.6,
        arrow_offset=0.08,      # base offset from circle (ignored if arrow_touch_circle=True)
        arrow_linewidth=2.0,
        arrow_touch_circle=True, # arrows start on the circle boundary
        arrow_separation=0.12,   # NEW: vertical nudge for left/right arrows (× radius)

        # output
        save_dir="fig_segments",
        fname_prefix="traj",
        dpi=300,
        show=False,             # show figures on screen (True) or close them (False)
    ):
        """
        Generate a series of figures, each covering 'steps_per_fig' timesteps.

        - Past trajectory is grey (and connected to current segment).
        - Current segment colored by iteration with full-spectrum colormap per figure.
        - Moving obstacles drawn with direction arrows; if arrow_touch_circle=True,
        the arrow stems start exactly on the obstacle circle. arrow_separation
        vertically offsets left/right arrows to avoid overlap.
        - If moving_view=True, each figure auto-zooms to the current segment with padding.
        """
        T = states.shape[0]
        if end_k is None:
            end_k = T
        assert 0 <= start_k < end_k <= T, "Invalid start/end indices."

        pos = states[:, :2]
        n_obs = obs_positions.shape[1]
        mv_idx = [i for i, md in enumerate(modes) if md.lower() != "static"]

        os.makedirs(save_dir, exist_ok=True)

        # Chunk the indices
        parts = []
        s = start_k
        while s < end_k:
            e = min(s + steps_per_fig, end_k)
            parts.append((s, e))  # [s, e)
            s = e

        saved_files = []

        for p_idx, (s, e) in enumerate(parts, start=1):
            k_segment = np.arange(s, e)
            k_past = np.arange(start_k, s) if draw_past_as_grey else np.array([], dtype=int)

            # Per-figure colormap with full sweep
            cmap = custom_cmap if (custom_cmap is not None) else cm.get_cmap(cmap_name)
            if gamma:
                norm = PowerNorm(gamma=gamma, vmin=float(s), vmax=float(e - 1))
            else:
                norm = Normalize(vmin=float(s), vmax=float(e - 1))

            fig, ax = plt.subplots(figsize=(7.8, 6.6))

            # ----- PAST (grey) -----
            if draw_past_as_grey and len(k_past) > 0:
                if draw_path and len(k_past) > 1:
                    ax.plot(pos[k_past, 0], pos[k_past, 1],
                            color=past_color, alpha=past_alpha, lw=past_lw, zorder=1)
                ax.scatter(pos[k_past, 0], pos[k_past, 1],
                        c=past_color, s=past_ms, edgecolor="none",
                        alpha=past_alpha, zorder=2)
                # connect last past point to first current point
                if s > start_k:
                    i0, i1 = s - 1, s
                    ax.plot([pos[i0, 0], pos[i1, 0]],
                            [pos[i0, 1], pos[i1, 1]],
                            color=past_color, alpha=past_alpha, lw=past_lw, zorder=1.2)

            # ----- CURRENT (colored) -----
            if draw_path and len(k_segment) > 1:
                for i0, i1 in zip(k_segment[:-1], k_segment[1:]):
                    ax.plot([pos[i0, 0], pos[i1, 0]],
                            [pos[i0, 1], pos[i1, 1]],
                            color=cmap(norm(i0)), lw=1.8, alpha=1.0, zorder=2.5)

            ax.scatter(pos[k_segment, 0], pos[k_segment, 1],
                    c=k_segment, cmap=cmap, norm=norm,
                    s=marker_size, edgecolor="none", zorder=3)

            # milestones (optional)
            if milestone_every and milestone_every > 0:
                mk_mask = (k_segment - start_k) % milestone_every == 0
                mk = k_segment[mk_mask]
                if mk.size > 0:
                    ax.scatter(pos[mk, 0], pos[mk, 1],
                            c=mk, cmap=cmap, norm=norm,
                            s=marker_size * 2.0, edgecolor="none", zorder=3.1)

            # colorbar with only start/end labels
            dummy = ax.scatter([np.nan], [np.nan], c=[k_segment[0]],
                            cmap=cmap, norm=norm, s=0)
            cb = plt.colorbar(dummy, ax=ax)
            cb.set_label("Iteration $k$", fontsize=14)
            cb.set_ticks([s, e - 1])
            cb.set_ticklabels([f"{s}", f"{e - 1}"])

            # show start/end on the canvas too
            ax.text(0.5, 1.02, f"start k = {s}", ha='center', va='bottom',
                    transform=ax.transAxes, fontsize=11)
            ax.text(0.5, -0.10, f"end k = {e - 1}", ha='center', va='top',
                    transform=ax.transAxes, fontsize=11)

            # ----- Obstacles -----
            # static: draw at t=0
            for i, md in enumerate(modes):
                if md.lower() == "static":
                    cx0, cy0 = obs_positions[0, i]
                    ax.add_patch(plt.Circle((cx0, cy0), radii[i],
                                            fill=False, color="k", lw=2, zorder=1.4))

            # moving: only within current segment
            if draw_moving_circles and len(mv_idx) > 0:
                for j, i in enumerate(mv_idx):
                    r = radii[i]
                    for t in k_segment:
                        cx, cy = obs_positions[t, i]
                        col = cmap(norm(t))
                        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                                linestyle='-', linewidth=1.8,
                                                edgecolor=col, alpha=1.0, zorder=1.5))

                        # direction arrow t -> t+1 (or backwards at segment end)
                        t2 = t + 1 if t < min(e - 1, T - 1) else max(t - 1, 0)
                        vx, vy = obs_positions[t2, i] - obs_positions[t, i]
                        vnorm = float(np.hypot(vx, vy))
                        if vnorm > 1e-12:
                            ux, uy = vx / vnorm, vy / vnorm
                            L  = arrow_len_scale * r

                            # start point on circle boundary (or with offset)
                            base = r if arrow_touch_circle else (r + arrow_offset)

                            # --- NEW: vertical nudge to separate opposite arrows ---
                            oy = 0.0
                            if vx > 0:
                                oy = +arrow_separation * r  # right: nudge up
                            elif vx < 0:
                                oy = -arrow_separation * r  # left : nudge down
                            # -------------------------------------------------------

                            sx = cx + base * ux
                            sy = cy + base * uy + oy
                            ex = sx + L * ux
                            ey = sy + L * uy

                            ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey),
                                                        arrowstyle='-|>',
                                                        mutation_scale=10 + 8 * r,
                                                        linewidth=arrow_linewidth,
                                                        color=col, alpha=1.0, zorder=1.7))

            # legend (optional)
            legend_handles = []
            if draw_past_as_grey and len(k_past) > 0:
                legend_handles.append(Line2D([0], [0], color=past_color, lw=past_lw,
                                            label="Past trajectory"))
            if any(md.lower() == "static" for md in modes):
                legend_handles.append(Line2D([0], [0], marker='o', lw=0,
                                            markerfacecolor='none', markeredgecolor='k',
                                            markersize=8, label="Static obstacle"))
            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper left", fontsize=10, frameon=True)

            # ----- View / axes -----
            if moving_view:
                # auto-zoom to current segment with padding
                xs = pos[k_segment, 0]
                ys = pos[k_segment, 1]
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                if xmin == xmax: xmin -= 1e-3; xmax += 1e-3
                if ymin == ymax: ymin -= 1e-3; ymax += 1e-3
                ax.set_xlim(xmin - view_pad, xmax + view_pad)
                ax.set_ylim(ymin - view_pad, ymax + view_pad)
            else:
                ax.set_xlim(xlim); ax.set_ylim(ylim)

            ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.5)
            ax.set_xlabel("$X$", fontsize=18); ax.set_ylabel("$Y$", fontsize=18)
            ax.set_title(f"Trajectory part {p_idx}: k ∈ [{s}, {e-1}]", fontsize=14)
            plt.tight_layout()

            # save
            fname = os.path.join(save_dir, f"{fname_prefix}_part{p_idx:02d}.png")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            saved_files.append(fname)
            if not show:
                plt.close(fig)

        return saved_files

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
    
    