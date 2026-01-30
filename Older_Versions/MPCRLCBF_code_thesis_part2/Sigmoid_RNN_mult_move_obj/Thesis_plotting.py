# THE FILE RESPONIBLE FOR PLOTTING THE RESULTS THAT GO IN THE THESIS

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from Functions import make_system_obstacle_svg_frames_v3, make_system_obstacle_montage_v1, make_system_obstacle_animation_v3
from matplotlib.legend_handler import HandlerBase
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,                         # use LaTeX for *all* text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],     # matches default LaTeX
    "mathtext.fontset": "cm",                    # fallback if usetex is off
    "axes.unicode_minus": False,                 # proper minus sign when TeX is off
    # Optional but handy:
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
})
# def plot_stagecost_compare(nn_path: str, rnn_path: str, out_dir: str = "thesis_plots"):
#     """
#     Compare NN and RNN smoothed stage costs from saved training_data.npz files.

#     Parameters
#     ----------
#     nn_path : str
#         Path to NN results .npz file (training_data.npz).
#     rnn_path : str
#         Path to RNN results .npz file (training_data.npz).
#     out_dir : str
#         Directory where the plot will be saved.
#     """
#     # Ensure output folder exists
#     os.makedirs(out_dir, exist_ok=True)

#     # Load NN data
#     nn_data = np.load(nn_path)
#     nn_episodes = nn_data["episodes"]
#     nn_mean = nn_data["running_mean"]
#     nn_std = nn_data["running_std"]
#     nn_window = int(nn_data["smoothing_window"][0])

#     # Load RNN data
#     rnn_data = np.load(rnn_path)
#     rnn_episodes = rnn_data["episodes"]
#     rnn_mean = rnn_data["running_mean"]
#     rnn_std = rnn_data["running_std"]
#     rnn_window = int(rnn_data["smoothing_window"][0])

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # NN curve + shaded band
    
    
#     ax.plot(nn_episodes, nn_mean, "-", linewidth=2, label=f"NN mean ({nn_window}-ep)")
#     ax.fill_between(nn_episodes, nn_mean - nn_std, nn_mean + nn_std, alpha=0.3)

#     # RNN curve + shaded band
#     ax.plot(rnn_episodes, rnn_mean, "-", linewidth=2, label=f"RNN mean ({rnn_window}-ep)")
#     ax.fill_between(rnn_episodes, rnn_mean - rnn_std, rnn_mean + rnn_std, alpha=0.3)

#     ax.set_yscale("log")
#     ax.set_xlabel("Episode Number", fontsize=16)
#     ax.set_ylabel("Stage Cost", fontsize=16)
#     ax.tick_params(labelsize=12)
#     ax.grid(True)
#     ax.legend(fontsize=14)

#     fig.tight_layout()

#     out_path = os.path.join(out_dir, "stagecost_compare_NN_vs_RNN.pdf")
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"Stage cost comparison plot saved at: {out_path}")

# # -------------------------
# # NPZ LOADER
# # -------------------------
# def _load_sim_npz(npz_path: str | Path):
#     p = Path(npz_path).expanduser().resolve()
#     d = np.load(p, allow_pickle=False)
#     return dict(
#         states        = d["states"],                # (T, >=2)
#         plans         = d["plans"],                 # (T, N+1, 2)
#         obs_positions = d["obs_positions"],         # (T, m, 2)
#         radii         = d["meta__radii"].ravel(),
#         constraints_x = float(np.asarray(d["meta__constraints_x"]).ravel()[0]),
#         suffix        = str(np.asarray(d["meta__run_tag"]).ravel()[0]),
#     )

# # -------------------------
# # RENDER ONE PANEL
# # -------------------------
# def _render_before_after_panel(
#     ax, k, B, A,
#     *,
#     before_color="C0",
#     after_color="red",
#     pred_before_color="orange",
#     pred_after_color="purple",
#     constraints_x=10.0,
#     follow=False, follow_width=6.0, follow_height=6.0,
#     tail_len=None, tail_fade_len=6,
#     tick_fontsize=14, axis_labelsize=20,
#     spine_width=1.6, tick_width=1.6,
#     k_fontsize=14, show_k=True
# ):
#     sysB = B["states"][:,:2]
#     sysA = A["states"][:,:2]
#     predB = B["plans"]; predA = A["plans"]
#     obs   = A["obs_positions"]
#     radii = A["radii"]

#     T   = min(sysB.shape[0], sysA.shape[0], predB.shape[0], predA.shape[0], obs.shape[0])
#     k   = int(np.clip(k, 0, T-1))
#     Np1 = predB.shape[1]
#     N   = max(0, Np1-1)
#     L   = N if tail_len is None else int(tail_len)

#     ax.set_aspect("equal", "box")
#     ax.grid(True, alpha=0.35)
#     ax.tick_params(labelsize=tick_fontsize, width=tick_width)
#     for s in ax.spines.values(): s.set_linewidth(spine_width)

#     # view / camera
#     if follow:
#         xkB, ykB = sysB[k]; xkA, ykA = sysA[k]
#         cx = 0.5*(xkB + xkA); cy = 0.5*(ykB + ykA)
#         pad = (max(radii) if len(radii) else 0.0)
#         ax.set_xlim(cx - follow_width/2 - pad,  cx + follow_width/2 + pad)
#         ax.set_ylim(cy - follow_height/2 - pad, cy + follow_height/2 + pad)
#     else:
#         span = float(constraints_x)
#         ax.set_xlim(-1.1*span, +0.2*span)
#         ax.set_ylim(-1.1*span, +0.2*span)

#     ax.set_xlabel("X", fontsize=axis_labelsize)
#     ax.set_ylabel("Y", fontsize=axis_labelsize)

#     # trail with fading markers on last K steps
#     def _draw_trail(xy, color):
#         s0 = max(0, k - L); e0 = k+1
#         tail = xy[s0:e0]
#         ax.plot(tail[:,0], tail[:,1], "-", lw=2.2, color=color, zorder=2.0)
#         if len(tail) > 1 and tail_fade_len > 1:
#             npts = min(len(tail)-1, tail_fade_len-1)
#             alphas = np.linspace(0.35, 1.0, npts)
#             for p, a in zip(tail[-(npts+1):-1], alphas):
#                 ax.plot(p[0], p[1], "o", ms=4.8, color=(*mcolors.to_rgb(color), a),
#                         zorder=2.1, markeredgewidth=0)

#     # draw trails
#     _draw_trail(sysB, before_color)
#     _draw_trail(sysA, after_color)

#     # current points (both hollow for visibility)
#     ax.plot(sysB[k,0], sysB[k,1], "o", ms=8.0, mec=before_color, mfc="white",
#             mew=2.0, zorder=5.1)
#     ax.plot(sysA[k,0], sysA[k,1], "o", ms=8.0, mec=after_color, mfc="white",
#             mew=2.0, zorder=5.2)

#     # predicted horizons
#     def _add_horizon(ph, col):
#         if N <= 0: return
#         future = ph[k, 1:, :]
#         poly = np.vstack((ph[k, 0:1, :], future))
#         segs = np.stack([poly[:-1], poly[1:]], axis=1)
#         lc = LineCollection(segs, linewidths=2.0, zorder=2.3)
#         rgb = mcolors.to_rgb(col)
#         seg_cols = np.tile((*rgb, 1.0), (N, 1))
#         seg_cols[:, 3] = np.linspace(1.0, 0.35, N)
#         lc.set_colors(seg_cols)
#         ax.add_collection(lc)
#         for j in range(N):
#             ax.plot(future[j,0], future[j,1], "o", ms=5.0, color=col, zorder=2.35)

#     _add_horizon(predB, pred_before_color)
#     _add_horizon(predA, pred_after_color)

#     # obstacles (current + dashed future silhouettes)
#     cmap = plt.get_cmap("tab10")
#     for i, r in enumerate(radii):
#         cx, cy = obs[k, i]
#         col = cmap.colors[i % len(cmap.colors)]
#         ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=2.2, zorder=1.0))
#     if N > 0:
#         alphas = np.linspace(0.35, 0.30, N)
#         for h in range(1, N+1):
#             t = min(k+h, T-1)
#             a = float(alphas[h-1])
#             for i, r in enumerate(radii):
#                 cx, cy = obs[t, i]
#                 col = cmap.colors[i % len(cmap.colors)]
#                 ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=1.4,
#                                         ls="--", alpha=a, zorder=0.9))

#     # k badge
#     if show_k:
#         ax.text(0.02, 0.98, f"$k$ = {k}", transform=ax.transAxes,  # capitalized K
#                 ha="left", va="top", fontsize=k_fontsize,
#                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

# # -------------------------
# # LEGEND (capitalized, object names, wrap to next line if needed)
# # -------------------------



# def _add_long_legend(
#     fig, *,
#     N, m_obst,
#     # colors (keep in sync with your plot)
#     before_line_color="C0",
#     after_line_color="red",
#     pred_before_color="orange",
#     pred_after_color="purple",
#     pred_obstacle_colors=("orange", "C0"),   # will be shown side-by-side
#     # layout / sizing
#     legend_fontsize=18,
#     pad_frac=0.16,           # height reserved at bottom for the legend strip
#     legend_max_cols=6,       # wrap if many entries
#     line_lw=2.6, marker_size=8,
#     handlelength=2.6, columnspacing=1.2, handletextpad=0.7, labelspacing=0.8,
#     left_pad=0.02, right_pad=0.02         # tighten white space on left/right
# ):
#     # --- proxies (LINES with a filled dot) ---
#     before_trail = mlines.Line2D([], [], color=before_line_color, lw=line_lw,
#         marker="o", markersize=marker_size, markerfacecolor=before_line_color,
#         markeredgecolor=before_line_color, label=f"Before: Last Steps (N={N})")
#     after_trail  = mlines.Line2D([], [], color=after_line_color,  lw=line_lw,
#         marker="o", markersize=marker_size, markerfacecolor=after_line_color,
#         markeredgecolor=after_line_color,  label=f"After: Last Steps (N={N})")
#     before_pred  = mlines.Line2D([], [], color=pred_before_color, lw=line_lw-0.3,
#         marker="o", markersize=marker_size, markerfacecolor=pred_before_color,
#         markeredgecolor=pred_before_color, label=f"Before: Predicted Horizon (N={N})")
#     after_pred   = mlines.Line2D([], [], color=pred_after_color,  lw=line_lw-0.3,
#         marker="o", markersize=marker_size, markerfacecolor=pred_after_color,
#         markeredgecolor=pred_after_color,  label=f"After: Predicted Horizon (N={N})")

#     # standalone “current” markers (no line)
#     before_curr = mlines.Line2D([], [], marker="o", linestyle="None",
#         markerfacecolor="white", markeredgecolor=before_line_color,
#         markeredgewidth=2.0, markersize=marker_size+2, label="Before: Current")
#     after_curr  = mlines.Line2D([], [], marker="o", linestyle="None",
#         markerfacecolor="white", markeredgecolor=after_line_color,
#         markeredgewidth=2.0, markersize=marker_size+2, label="After: Current")

#     # Obstacles (current), named individually
#     cmap = plt.get_cmap("tab10")
#     obst_handles = [
#         mlines.Line2D([], [], color=cmap.colors[i % len(cmap.colors)],
#                       lw=line_lw-0.2, label=f"Obstacle {i+1}")
#         for i in range(m_obst)
#     ]

#     # ONE entry that shows TWO dashed predicted obstacle colors, SIDE-BY-SIDE
#     pred_obst1 = mlines.Line2D([], [], color=pred_obstacle_colors[0], lw=line_lw-0.4, ls="--")
#     pred_obst2 = mlines.Line2D([], [], color=pred_obstacle_colors[1], lw=line_lw-0.4, ls="--")
#     pred_obst_tuple = (pred_obst1, pred_obst2)  # side-by-side with HandlerTuple default
#     pred_obst_label = f"Obstacle (Predicted, N={N} Ahead)"

#     # collect
#     handles = [before_trail, after_trail, before_pred, after_pred,
#                before_curr, after_curr, *obst_handles, pred_obst_tuple]
#     labels  = [h.get_label() for h in [before_trail, after_trail, before_pred, after_pred, before_curr, after_curr]]
#     labels += [f"Obstacle {i+1}" for i in range(m_obst)]
#     labels += [pred_obst_label]

#     # --- reserve a dedicated bottom strip & tighten side margins ---
#     bottom = max(0.05, pad_frac)
#     fig.subplots_adjust(bottom=bottom, left=left_pad, right=1.0-right_pad)  # ← trims extra white on sides

#     # [left, bottom, width, height] in figure fraction
#     leg_ax = fig.add_axes([left_pad, 0.0, 1.0 - left_pad - right_pad, bottom - 0.01])
#     leg_ax.axis("off")

#     ncol = min(len(handles), legend_max_cols)

#     leg_ax.legend(
#         handles=handles,
#         labels=labels,
#         loc="center",
#         ncol=ncol,
#         fontsize=legend_fontsize,
#         frameon=False,
#         borderaxespad=0.0,
#         handlelength=handlelength,
#         columnspacing=columnspacing,
#         handletextpad=handletextpad,
#         labelspacing=labelspacing,
#         handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},  # side-by-side
#     )
# def _render_before_after_panel(
#     ax, k, B, A,
#     *,
#     before_color="C0",
#     after_color="red",
#     pred_before_color="orange",
#     pred_after_color="purple",
#     constraints_x=10.0,
#     follow=False, follow_width=6.0, follow_height=6.0,
#     tail_len=None, tail_fade_len=6,
#     tick_fontsize=14,                      # sizes still respected
#     spine_width=1.6, tick_width=1.6,
#     k_fontsize=14, show_k=True,
#     set_axis_labels=False, axis_labelsize=20  # NEW: normally False; labels set in post-pass
# ):
#     import matplotlib.colors as mcolors
#     from matplotlib.collections import LineCollection
#     import numpy as np
#     import matplotlib.pyplot as plt

#     sysB = B["states"][:,:2]
#     sysA = A["states"][:,:2]
#     predB = B["plans"]; predA = A["plans"]
#     obs   = A["obs_positions"]
#     radii = A["radii"]

#     T   = min(sysB.shape[0], sysA.shape[0], predB.shape[0], predA.shape[0], obs.shape[0])
#     k   = int(np.clip(k, 0, T-1))
#     Np1 = predB.shape[1]
#     N   = max(0, Np1-1)
#     L   = N if tail_len is None else int(tail_len)

#     ax.set_aspect("equal", "box")
#     ax.grid(True, alpha=0.35)
#     ax.tick_params(labelsize=tick_fontsize, width=tick_width)
#     for s in ax.spines.values(): s.set_linewidth(spine_width)

#     # view / camera
#     if follow:
#         xkB, ykB = sysB[k]; xkA, ykA = sysA[k]
#         cx = 0.5*(xkB + xkA); cy = 0.5*(ykB + ykA)
#         pad = max(radii) if len(radii) else 0.0
#         ax.set_xlim(cx - follow_width/2 - pad,  cx + follow_width/2 + pad)
#         ax.set_ylim(cy - follow_height/2 - pad, cy + follow_height/2 + pad)
#     else:
#         span = float(constraints_x)
#         ax.set_xlim(-1.1*span, +0.2*span)
#         ax.set_ylim(-1.1*span, +0.2*span)

#     if set_axis_labels:           # usually False; we set labels in a post-pass
#         ax.set_xlabel("X", fontsize=axis_labelsize)
#         ax.set_ylabel("Y", fontsize=axis_labelsize)

#     # trail with fade
#     def _draw_trail(xy, color):
#         s0 = max(0, k - L); e0 = k+1
#         tail = xy[s0:e0]
#         ax.plot(tail[:,0], tail[:,1], "-", lw=2.2, color=color, zorder=2.0)
#         if len(tail) > 1 and tail_fade_len > 1:
#             npts = min(len(tail)-1, tail_fade_len-1)
#             alphas = np.linspace(0.35, 1.0, npts)
#             for p, a in zip(tail[-(npts+1):-1], alphas):
#                 ax.plot(p[0], p[1], "o", ms=4.8, color=(*mcolors.to_rgb(color), a),
#                         zorder=2.1, markeredgewidth=0)

#     _draw_trail(sysB, before_color)
#     _draw_trail(sysA, after_color)

#     # current points (hollow for visibility)
#     ax.plot(sysB[k,0], sysB[k,1], "o", ms=8.0, mec=before_color, mfc="white", mew=2.0, zorder=5.1)
#     ax.plot(sysA[k,0], sysA[k,1], "o", ms=8.0, mec=after_color,  mfc="white", mew=2.0, zorder=5.2)

#     # predicted horizons
#     def _add_horizon(ph, col):
#         if N <= 0: return
#         future = ph[k, 1:, :]
#         poly = np.vstack((ph[k, 0:1, :], future))
#         segs = np.stack([poly[:-1], poly[1:]], axis=1)
#         lc = LineCollection(segs, linewidths=2.0, zorder=2.3)
#         rgb = mcolors.to_rgb(col)
#         seg_cols = np.tile((*rgb, 1.0), (N, 1))
#         seg_cols[:, 3] = np.linspace(1.0, 0.35, N)
#         lc.set_colors(seg_cols)
#         ax.add_collection(lc)
#         for j in range(N):
#             ax.plot(future[j,0], future[j,1], "o", ms=5.0, color=col, zorder=2.35)

#     _add_horizon(predB, pred_before_color)
#     _add_horizon(predA, pred_after_color)

#     # obstacles (current + dashed future)
#     cmap = plt.get_cmap("tab10")
#     for i, r in enumerate(radii):
#         cx, cy = obs[k, i]
#         col = cmap.colors[i % len(cmap.colors)]
#         ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=2.2, zorder=1.0))
#     if N > 0:
#         alphas = np.linspace(0.35, 0.30, N)
#         for h in range(1, N+1):
#             t = min(k+h, T-1)
#             a = float(alphas[h-1])
#             for i, r in enumerate(radii):
#                 cx, cy = obs[t, i]
#                 col = cmap.colors[i % len(cmap.colors)]
#                 ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=1.4,
#                                         ls="--", alpha=a, zorder=0.9))

#     if show_k:
#         ax.text(0.02, 0.98, f"$k$ = {k}", transform=ax.transAxes,
#                 ha="left", va="top", fontsize=k_fontsize,
#                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))


# def make_montage_pdf(
#     npz_before, npz_after,
#     out_root="thesis_plots",
#     frame_indices=None,
#     figsize_per_ax=(10, 9),
#     figscale=1.35,
#     follow=True, follow_width=1.5, follow_height=1.5,
#     tail_len=None, tail_fade_len=6,
#     # text sizes
#     tick_fontsize=25, axis_labelsize=30, k_fontsize=35,
#     # legend
#     legend_fontsize=35, legend_max_cols=8, legend_pad_frac=0.16,
#     # NEW spacing/margins (wspace=0 packs tightly)
#     wspace: float = 0.0,
#     left_pad: float = 0.01, right_pad: float = 0.01,
#     # NEW outer-only policy
#     y_axis_first_col_only: bool = True,
#     x_axis_on_all: bool = True,
# ):
#     B = _load_sim_npz(npz_before)
#     A = _load_sim_npz(npz_after)

#     import numpy as np
#     T = min(B["states"].shape[0], A["states"].shape[0],
#             B["plans"].shape[0],  A["plans"].shape[0],
#             B["obs_positions"].shape[0], A["obs_positions"].shape[0])

#     if not frame_indices:
#         n = 8
#         step = max(1, T // n)
#         frame_indices = list(range(0, min(T, step*n), step))

#     Np1 = B["plans"].shape[1]; N = max(0, Np1-1)
#     m_obst = A["obs_positions"].shape[1]

#     cols = len(frame_indices)
#     fig_w = cols * figsize_per_ax[0] * figscale
#     fig_h = 1 * figsize_per_ax[1] * figscale

#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(1, cols, figsize=(fig_w, fig_h), squeeze=False)
#     # leave space for legend, set tight horizontal spacing and trimmed side margins
#     fig.subplots_adjust(wspace=wspace, left=left_pad, right=1.0-right_pad, bottom=legend_pad_frac)

#     # render panels (do NOT set axis labels here)
#     for i, k in enumerate(frame_indices):
#         _render_before_after_panel(
#             axs[0, i], k, B, A,
#             constraints_x=A["constraints_x"],
#             follow=follow, follow_width=follow_width, follow_height=follow_height,
#             tail_len=tail_len, tail_fade_len=tail_fade_len,
#             tick_fontsize=tick_fontsize, axis_labelsize=axis_labelsize,
#             k_fontsize=k_fontsize,
#             spine_width=2.2, tick_width=2.2,
#             set_axis_labels=False
#         )

#     # ---- outer-only post-pass ----
#     for c in range(cols):
#         ax = axs[0, c]
#         first_col = (c == 0)
#         # Y only on first column
#         if y_axis_first_col_only:
#             if first_col:
#                 ax.set_ylabel("Y", fontsize=axis_labelsize)
#                 ax.tick_params(labelleft=True, left=True)
#             else:
#                 ax.set_ylabel("")
#                 ax.tick_params(labelleft=False, left=False)
#         # X on all panels
#         if x_axis_on_all:
#             ax.set_xlabel("X", fontsize=axis_labelsize)
#             ax.tick_params(labelbottom=True, bottom=True)

#     # legend strip
#     _add_long_legend(
#         fig, N=N, m_obst=m_obst,
#         legend_fontsize=legend_fontsize,
#         pad_frac=legend_pad_frac,
#         legend_max_cols=legend_max_cols,
#         before_line_color="C0", after_line_color="red",
#         pred_before_color="orange", pred_after_color="purple",
#         pred_obstacle_colors=("orange", "C0"),
#         left_pad=left_pad, right_pad=right_pad
#     )

#     from pathlib import Path
#     out_root = Path(out_root).expanduser().resolve()
#     out_root.mkdir(parents=True, exist_ok=True)
#     out_path = out_root / f"montage_{B['suffix']}_vs_{A['suffix']}.pdf"
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
#     plt.close(fig)
#     print(f"[done] montage saved to {out_path}")

def save_rnn_gif(
    npz_path,
    out_root="thesis_plots",

    # view
    camera="follow",
    follow_width=2.6,
    follow_height=2.6,
    figsize=(6.5, 6),

    # timing
    fps=12,                 # global speed (maps to base_fps)
    segment_range=None,     # e.g. (9, 20) inclusive, or None
    segment_fps=None,       # e.g. 6 to slow inside segment, or 24 to speed up
    stride=1,               # subsample frames: 1=no skip, 2=every other, etc.

    # styling / output
    dpi=140,
    system_color="grey",
    pred_color="#000000FF",
    save_mp4=False,
    show=False,
):
    """
    Load a simulation_*.npz and save an animated GIF (and optional MP4).
    Keeps your usual options and adds per-segment speed control.
    """
    D = _load_sim_npz(npz_path)  # uses your existing loader
    states = D["states"]
    plans  = D["plans"]
    obs    = D["obs_positions"]

    # stride (shorten the sequence globally before any segment timing)
    if stride > 1:
        states = states[::stride]
        plans  = plans[::stride]
        obs    = obs[::stride]

    # output path
    out_dir = Path(out_root).expanduser().resolve() / f"gifs_{D['suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ""
    if segment_range and segment_fps:
        k0, k1 = segment_range
        tag = f"_seg{k0}-{k1}_{segment_fps:g}fps"
    out_path = out_dir / f"RNN_{D['suffix']}{tag}.gif"

    make_system_obstacle_animation_v3(
        states_eval=states,
        pred_paths=plans,
        obs_positions=obs,
        radii=D["radii"],
        constraints_x=D["constraints_x"],
        out_path=str(out_path),

        # visuals
        figsize=figsize,
        dpi=dpi,
        system_color=system_color,
        pred_color=pred_color,

        # camera
        camera=camera,
        follow_width=follow_width,
        follow_height=follow_height,

        # speed control
        base_fps=fps,                   # global playback speed
        segment_range=segment_range,    # e.g. (9, 20)
        segment_fps=segment_fps,        # e.g. 6 (slower) or 24 (faster)

        # files
        save_gif=True,
        save_mp4=save_mp4,
        mp4_path=str(out_path.with_suffix(".mp4")) if save_mp4 else None,
        show=show,
    )
    print(f"[gif] saved {out_path}")
    if save_mp4:
        print(f"[mp4] saved {out_path.with_suffix('.mp4')}")
    
class TwoColorDash(HandlerBase):
    """Legend handler for a 2-Line2D tuple: draws two dashed lines,
    vertically stacked so both colors are visible even in a narrow box."""
    def __init__(self, pad=0.18, sep=0.55, min_lw=1.2):
        super().__init__()
        self.pad = pad
        self.sep = sep
        self.min_lw = min_lw

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        l1, l2 = orig_handle
        x0 = xdescent + width * self.pad
        x1 = xdescent + width * (1.0 - self.pad)
        ymid = ydescent + 0.5 * height
        dy = 0.5 * height * self.sep

        def dash(proto, y):
            return mlines.Line2D(
                [x0, x1], [y, y], transform=trans,
                color=proto.get_color(),
                lw=max(self.min_lw, proto.get_linewidth()),
                ls=proto.get_linestyle() or "--",
                solid_capstyle="round",
            )

        return [dash(l1, ymid + dy), dash(l2, ymid - dy)]


# -------------------------- helpers --------------------------

def _load_sim_npz(npz_path):
    d = np.load(Path(npz_path).expanduser().resolve(), allow_pickle=False)
    return dict(
        states        = d["states"],
        plans         = d["plans"],
        obs_positions = d["obs_positions"],
        radii         = d["meta__radii"].ravel(),
        constraints_x = float(np.asarray(d["meta__constraints_x"]).ravel()[0]),
        suffix        = str(np.asarray(d["meta__run_tag"]).ravel()[0]),
    )


# ------------------- panel renderer (before/after) -------------------

def _render_before_after_panel(
    ax, k, B, A, *,
    before_color="purple", after_color="red",
    pred_before_color=None, pred_after_color=None,
    constraints_x=10.0,
    follow=True, follow_width=1.5, follow_height=1.5,
    tail_len=None, tail_fade_len=6,
    tick_fontsize=25, spine_width=2.2, tick_width=2.2,
    k_fontsize=35
):
    if pred_before_color is None:
        pred_before_color = before_color
    if pred_after_color is None:
        pred_after_color = after_color

    sysB = B["states"][:, :2]
    sysA = A["states"][:, :2]
    predB = B["plans"]; predA = A["plans"]
    obs   = A["obs_positions"]; radii = A["radii"]

    T   = min(sysB.shape[0], sysA.shape[0], predB.shape[0], predA.shape[0], obs.shape[0])
    k   = int(np.clip(k, 0, T - 1))
    Np1 = predB.shape[1]; N = max(0, Np1 - 1)
    L   = N if tail_len is None else int(tail_len)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_anchor("C")
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=tick_fontsize, width=tick_width)
    for s in ax.spines.values():
        s.set_linewidth(spine_width)

    if follow:
        xkB, ykB = sysB[k]; xkA, ykA = sysA[k]
        cx, cy = 0.5 * (xkB + xkA), 0.5 * (ykB + ykA)
        pad_r = max(radii) if len(radii) else 0.0
        side = max(follow_width, follow_height) + 2 * pad_r
        half = side / 2.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    else:
        span = float(constraints_x)
        half = 0.65 * span
        ax.set_xlim(-half, +half)
        ax.set_ylim(-half, +half)

    # --- trails (solid) ---
    def _trail(xy, col):
        s0 = max(0, k - L); tail = xy[s0:k + 1]
        ax.plot(tail[:, 0], tail[:, 1], "-", lw=2.2, color=col, zorder=2.0)
        if len(tail) > 1 and tail_fade_len > 1:
            npts = min(len(tail) - 1, tail_fade_len - 1)
            alphas = np.linspace(0.35, 1.0, npts)
            for p, a in zip(tail[-(npts + 1):-1], alphas):
                ax.plot(p[0], p[1], "o", ms=4.8, color=(*mcolors.to_rgb(col), a),
                        zorder=2.1, markeredgewidth=0)

    _trail(sysB, before_color)
    _trail(sysA, after_color)

    ax.plot(sysB[k, 0], sysB[k, 1], "o", ms=8.0, mec=before_color, mfc="white", mew=2.0, zorder=5.1)
    ax.plot(sysA[k, 0], sysA[k, 1], "o", ms=8.0, mec=after_color,  mfc="white", mew=2.0, zorder=5.2)

    # --- dashed horizons ---
    def _horizon(ph, col):
        if N <= 0: return
        future = ph[k, 1:, :]
        poly = np.vstack((ph[k, 0:1, :], future))
        segs = np.stack([poly[:-1], poly[1:]], axis=1)
        lc = LineCollection(segs, linewidths=2.0, zorder=2.3)
        rgb = mcolors.to_rgb(col)
        cols = np.tile((*rgb, 1.0), (N, 1))
        cols[:, 3] = np.linspace(1.0, 0.35, N)
        lc.set_colors(cols)
        lc.set_linestyle("--")
        ax.add_collection(lc)
        for j in range(N):
            ax.plot(future[j, 0], future[j, 1], "o", ms=5.0, color=col, zorder=2.35)

    _horizon(predB, pred_before_color)
    _horizon(predA, pred_after_color)

    # --- obstacles (current + dashed future) ---
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(radii):
        cx, cy = obs[k, i]; col = cmap.colors[i % len(cmap.colors)]
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=2.2, zorder=1.0))
    if N > 0:
        alphas = np.linspace(0.35, 0.30, N)
        for h in range(1, N + 1):
            t = min(k + h, T - 1); a = float(alphas[h - 1])
            for i, r in enumerate(radii):
                cx, cy = obs[t, i]; col = cmap.colors[i % len(cmap.colors)]
                ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=col, lw=1.4, ls="--", alpha=a, zorder=0.9))

    ax.text(0.02, 0.98, f"$k$ = {k}", transform=ax.transAxes,
            ha="left", va="top", fontsize=k_fontsize,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))


# ------------------------------ legend strip ------------------------------

def _add_long_legend(fig, *, N, m_obst,
                     legend_fontsize=35,
                     pad_frac=0.075,         # total bottom margin height (as fraction of figure)
                     legend_max_cols=6):
    """
    Add a long legend as a figure legend, centered in the bottom margin.

    pad_frac: bottom margin reserved for legend (same value as used in
              fig.subplots_adjust(bottom=pad_frac) in the main function).
    """

    before_line_color, after_line_color = "purple", "red"
    line_lw, marker_size = 2.6, 8

    before_trail = mlines.Line2D([], [], color=before_line_color, lw=line_lw,
        marker="o", ms=marker_size, mfc=before_line_color, mec=before_line_color,
        label=f"Before: Last Steps (N={N})")
    after_trail  = mlines.Line2D([], [], color=after_line_color,  lw=line_lw,
        marker="o", ms=marker_size, mfc=after_line_color,  mec=after_line_color,
        label=f"After: Last Steps (N={N})")

    before_pred  = mlines.Line2D([], [], color=before_line_color, lw=line_lw-0.3,
        linestyle="--", marker="o", ms=marker_size, mfc=before_line_color, mec=before_line_color,
        label=f"Before: Predicted Horizon (N={N})")
    after_pred   = mlines.Line2D([], [], color=after_line_color,  lw=line_lw-0.3,
        linestyle="--", marker="o", ms=marker_size, mfc=after_line_color,  mec=after_line_color,
        label=f"After: Predicted Horizon (N={N})")

    before_curr = mlines.Line2D([], [], marker="o", ls="None", mfc="white",
        mec=before_line_color, mew=2.0, ms=marker_size+8, label="Before: Current")
    after_curr  = mlines.Line2D([], [], marker="o", ls="None", mfc="white",
        mec=after_line_color,  mew=2.0, ms=marker_size+8, label="After: Current")

    cmap = plt.get_cmap("tab10")
    obstacles = [
        mlines.Line2D([], [], color=cmap.colors[i % len(cmap.colors)],
                      lw=line_lw-0.2, label=f"Obstacle {i+1}")
        for i in range(m_obst)
    ]

    # two dashed colors stacked horizontally
    pred_obst_tuple = (
        mlines.Line2D([], [], color="orange", lw=line_lw-0.4, ls="--"),
        mlines.Line2D([], [], color="C0",    lw=line_lw-0.4, ls="--"),
    )

    handles = [before_trail, after_trail, before_pred, after_pred,
               before_curr, after_curr, *obstacles, pred_obst_tuple]
    labels  = [h.get_label() for h in
               [before_trail, after_trail, before_pred, after_pred,
                before_curr, after_curr]]
    labels += [f"Obstacle {i+1}" for i in range(m_obst)]
    labels += [f"Obstacle (Predicted, N={N} Ahead)"]

    ncol = min(len(handles), legend_max_cols)

    # place legend in the center of the bottom margin
    legend_y = pad_frac / 2.0   # halfway inside bottom margin band
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        fontsize=legend_fontsize,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.8,
        columnspacing=1.3,
        handletextpad=0.8,
        bbox_to_anchor=(0.5, legend_y),
        handler_map={tuple: TwoColorDash(pad=0.18, sep=0.55)},
    )


# ------------------------------ montage + cost plot ------------------------------

def make_montage_pdf_2x4(npz_before, npz_after, out_root="thesis_plots",
                         frame_indices=None, figsize_per_ax=(10,9), figscale=1.35,
                         wspace=0.1, hspace=0.08,
                         left_pad=0.03, right_pad=0.01,
                         legend_fontsize=45, legend_max_cols=6, legend_pad_frac=0.05,
                         tick_fontsize=35, axis_labelsize=35, k_fontsize=40,
                         follow=True, follow_width=1.5, follow_height=1.5,
                         tail_len=None, tail_fade_len=6):
    B = _load_sim_npz(npz_before)
    A = _load_sim_npz(npz_after)
    T = min(B["states"].shape[0], A["states"].shape[0],
            B["plans"].shape[0], A["plans"].shape[0],
            B["obs_positions"].shape[0], A["obs_positions"].shape[0])

    if not frame_indices:
        n = 8; step = max(1, T // n)
        frame_indices = list(range(0, min(T, step * n), step))
    if len(frame_indices) > 8:
        frame_indices = frame_indices[:8]

    nrows, ncols = 2, 4
    fig_w = ncols * figsize_per_ax[0] * figscale
    fig_h = nrows * figsize_per_ax[1] * figscale
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=left_pad, right=1.0 - right_pad,
                        bottom=legend_pad_frac, top=1.0,
                        wspace=wspace, hspace=hspace)
    fig.set_facecolor("white")

    for idx, k in enumerate(frame_indices):
        r, c = divmod(idx, ncols)
        _render_before_after_panel(axs[r, c], k, B, A,
            constraints_x=A["constraints_x"],
            follow=follow, follow_width=follow_width, follow_height=follow_height,
            tail_len=tail_len, tail_fade_len=tail_fade_len,
            tick_fontsize=tick_fontsize, k_fontsize=k_fontsize)
    for blank in range(len(frame_indices), nrows * ncols):
        r, c = divmod(blank, ncols); axs[r, c].axis("off")

    N = max(0, B["plans"].shape[1] - 1)
    m_obst = A["obs_positions"].shape[1]
    _add_long_legend(fig, N=N, m_obst=m_obst,
        legend_fontsize=legend_fontsize, legend_max_cols=legend_max_cols,
        pad_frac=legend_pad_frac, left_pad=left_pad, right_pad=right_pad)

    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"montage_{B['suffix']}_vs_{A['suffix']}_RNN.pdf"
    fig.savefig(out_path, dpi=200, pad_inches=0.02)
    plt.close(fig)
    print(f"[done] montage saved to {out_path}")
    return out_path


def plot_stagecost_compare_validation(nn_file, rnn_file):
    data_nn = np.load(nn_file, allow_pickle=True)
    data_rnn = np.load(rnn_file, allow_pickle=True)
    if "stage_cost_valid" not in data_nn or "stage_cost_valid" not in data_rnn:
        raise KeyError("Both NPZs must contain 'stage_cost_valid'.")

    vals_nn = np.asarray(data_nn["stage_cost_valid"]).reshape(-1)
    vals_rnn = np.asarray(data_rnn["stage_cost_valid"]).reshape(-1)
    x_nn = np.arange(1, len(vals_nn) + 1)
    x_rnn = np.arange(1, len(vals_rnn) + 1)

    plt.figure()
    plt.plot(x_nn, vals_nn, "o-", label="NN-CBF")
    plt.plot(x_rnn, vals_rnn, "o-", label="RNN-CBF")
    plt.xlabel("Evaluation #", size=16)
    plt.ylabel("Stage Cost", size=16)
    plt.grid(True)
    plt.legend(fontsize=14)

    out_dir = os.path.dirname(nn_file)
    out_path = os.path.join(out_dir, "stage_cost_valid_comparison.pdf")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved validation comparison plot at: {out_path}")
    return out_path

def make_montage_pdf_2x4_compare(
    nn_before_npz, nn_after_npz,
    rnn_before_npz, rnn_after_npz,
    out_root="thesis_plots",
    frame_indices=None,
    figsize_per_ax=(10, 9),
    figscale=1.0,
    wspace=0.10,
    hspace=0.10,
    top_margin=0.98,       # tiny white space at top
    gap_height=0.07,       # relative height of white gap between NN and RNN
    left_pad=0.06,
    right_pad=0.01,
    legend_fontsize=30,
    legend_max_cols=6,
    legend_pad_frac= 0.099,  # bottom margin reserved for legend
    tick_fontsize=25,
    axis_labelsize=30,     # kept for API compatibility
    k_fontsize=30,
    follow=True,
    follow_width=1.0,
    follow_height=1.0,
    tail_len=None,
    tail_fade_len=6,
    nn_label="NN-CBF",
    rnn_label="RNN-CBF",
    label_fontsize=40,
):
    """
    Combined montage figure:

    - NN-CBF (rows 0–1)
    - white gap (row 2)
    - RNN-CBF (rows 3–4)
    """

    # ---- Load ----
    B_nn = _load_sim_npz(nn_before_npz)
    A_nn = _load_sim_npz(nn_after_npz)
    B_rn = _load_sim_npz(rnn_before_npz)
    A_rn = _load_sim_npz(rnn_after_npz)

    T = min(B_nn["states"].shape[0], A_nn["states"].shape[0],
            B_rn["states"].shape[0], A_rn["states"].shape[0])

    # ---- Frame selection ----
    if not frame_indices:
        step = max(1, T // 8)
        frame_indices = list(range(0, min(T, 8 * step), step))
    frame_indices = frame_indices[:8]

    nrows, ncols = 5, 4

    fig_w = ncols * figsize_per_ax[0] * figscale
    fig_h = nrows * figsize_per_ax[1] * figscale
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Height ratios: NN row, NN row, gap, RNN row, RNN row
    height_ratios = [1.0, 1.0, gap_height, 1.0, 1.0]

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    axs = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j] = fig.add_subplot(gs[i, j])

    # Layout margins (bottom leaves room for legend)
    fig.subplots_adjust(
        left=left_pad,
        right=1.0 - right_pad,
        top=top_margin,
        bottom=legend_pad_frac,
    )
    fig.set_facecolor("white")

    # Gap row blank
    for c in range(ncols):
        axs[2, c].axis("off")

    # ---- Fill NN (rows 0–1) and RNN (rows 3–4) ----
    for idx, k in enumerate(frame_indices):
        r, c = divmod(idx, ncols)   # r in {0,1}

        _render_before_after_panel(
            axs[r, c], k, B_nn, A_nn,
            constraints_x=A_nn["constraints_x"],
            follow=follow,
            follow_width=follow_width,
            follow_height=follow_height,
            tail_len=tail_len,
            tail_fade_len=tail_fade_len,
            tick_fontsize=tick_fontsize,
            k_fontsize=k_fontsize,
        )

        _render_before_after_panel(
            axs[r + 3, c], k, B_rn, A_rn,
            constraints_x=A_rn["constraints_x"],
            follow=follow,
            follow_width=follow_width,
            follow_height=follow_height,
            tail_len=tail_len,
            tail_fade_len=tail_fade_len,
            tick_fontsize=tick_fontsize,
            k_fontsize=k_fontsize,
        )

    # ---- Legend (compact at bottom) ----
    N = B_nn["plans"].shape[1] - 1
    m_obst = A_nn["obs_positions"].shape[1]
    _add_long_legend(
        fig,
        N=N,
        m_obst=m_obst,
        legend_fontsize=legend_fontsize,
        pad_frac=legend_pad_frac,
        legend_max_cols=legend_max_cols,
    )

    # ---- Center vertical labels using actual positions ----
    fig.canvas.draw()

    nn_top = axs[0, 0].get_position().y1
    nn_bottom = axs[1, 0].get_position().y0
    nn_center = 0.5 * (nn_top + nn_bottom)

    rnn_top = axs[3, 0].get_position().y1
    rnn_bottom = axs[4, 0].get_position().y0
    rnn_center = 0.5 * (rnn_top + rnn_bottom)

    fig.text(
        0.015, nn_center,
        nn_label,
        rotation=90,
        va="center",
        ha="left",
        fontsize=label_fontsize,
        fontweight="bold",
    )
    fig.text(
        0.015, rnn_center,
        rnn_label,
        rotation=90,
        va="center",
        ha="left",
        fontsize=label_fontsize,
        fontweight="bold",
    )

    # ---- Save ----
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"montage_NN_RNN.pdf"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[done] combined montage saved to {out_path}")
    return out_path

def _savefig(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{name}.pdf"#f"{name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def plot_stagecost_compare(
    nn_path: str,
    rnn_path: str,
    out_dir: str = "thesis_plots",
    max_drop_frac_nn: float = 0.5,
    max_drop_frac_rnn: float = 0.5,
):
    """
    Compare NN and RNN smoothed stage costs from saved training_data.npz files.

    max_drop_frac_* : float in [0,1]
        Asymmetric cap for the lower std band.
        Example: 0.5 means the lower band won't go below 50% of the mean.
    """
    os.makedirs(out_dir, exist_ok=True)
    eps = 1e-12  # for log-scale safety

    # --- Load NN ---
    nn_data = np.load(nn_path)
    nn_episodes = nn_data["episodes"]
    nn_mean = nn_data["running_mean"]
    nn_std = nn_data["running_std"]
    nn_window = int(nn_data["smoothing_window"][0])

    # --- Load RNN ---
    rnn_data = np.load(rnn_path)
    rnn_episodes = rnn_data["episodes"]
    rnn_mean = rnn_data["running_mean"]
    rnn_std = rnn_data["running_std"]
    rnn_window = int(rnn_data["smoothing_window"][0])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # NN curve + asymmetrically capped band
    ax.plot(nn_episodes, nn_mean, "-", linewidth=2, label=f"NN mean ({nn_window}-ep)")
    nn_lower_raw = nn_mean - nn_std
    nn_lower_cap = nn_mean * (1.0 - max_drop_frac_nn)
    nn_lower = np.maximum(nn_lower_raw, nn_lower_cap)
    nn_lower = np.clip(nn_lower, eps, None)  # keep positive for log scale
    ax.fill_between(nn_episodes, nn_lower, nn_mean + nn_std, alpha=0.3)

    # RNN curve + asymmetrically capped band
    ax.plot(rnn_episodes, rnn_mean, "-", linewidth=2, label=f"RNN mean ({rnn_window}-ep)")
    rnn_lower_raw = rnn_mean - rnn_std
    rnn_lower_cap = rnn_mean * (1.0 - max_drop_frac_rnn)
    rnn_lower = np.maximum(rnn_lower_raw, rnn_lower_cap)
    rnn_lower = np.clip(rnn_lower, eps, None)
    ax.fill_between(rnn_episodes, rnn_lower, rnn_mean + rnn_std, alpha=0.3)

    ax.set_yscale("log")
    ax.set_xlabel("Episode Number", fontsize=16)
    ax.set_ylabel("Stage Cost", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=14)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "stagecost_compare_NN_vs_RNN.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Stage cost comparison plot saved at: {out_path}")

def plot_simulation_npz(npz_path: str | Path, out_root: str | Path = "thesis_plots",
                        frame_indices: list[int] | None = None, stride: int = 1) -> None:
    """
    Load one simulation_*.npz and:
      - render per-frame SVGs
      - render montage
      - plot states/actions/stagecost/velocities/alphas/h(x)
      - dump lam_g table if present
    Saves into thesis_plots/snapshots_{suffix}/ and thesis_plots/metrics_{suffix}/
    """
    npz_path = Path(npz_path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)

    # required arrays (saved by your NPZBuilder)
    states        = data["states"]              # (T, >=4)
    plans         = data["plans"]               # (T_pred, N+1, 2)
    obs_positions = data["obs_positions"]       # (T, m, 2)

    # meta (saved via meta(...))
    radii         = data["meta__radii"].ravel()
    constraints_x = float(np.asarray(data["meta__constraints_x"]).ravel()[0])
    suffix        = str(np.asarray(data["meta__run_tag"]).ravel()[0])  # 'before' or 'after'

    # optional arrays
    actions      = data["actions"]                   if "actions"      in data.files else None
    stage_cost   = data["stage_cost"].reshape(-1)    if "stage_cost"   in data.files else None
    alphas       = data["alphas"]                    if "alphas"       in data.files else None
    hx           = data["hx"]                        if "hx"           in data.files else None
    slacks_eval  = data["slacks_eval"]               if "slacks_eval"  in data.files else None
    lam_g_hist   = data["lam_g_hist"]                if "lam_g_hist"   in data.files else None

    print(f"slacks_eval shape: {slacks_eval.shape if slacks_eval is not None else None}")
    print(f"data keys: {data.files}")

    # output dirs
    out_root    = Path(out_root).expanduser().resolve()
    svg_dir     = out_root / f"snapshots_{suffix}"
    metrics_dir = out_root / f"metrics_{suffix}"
    svg_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ====== 1) Per-frame SVGs ======
    T_pred = plans.shape[0]
    make_system_obstacle_svg_frames_v3(
        states_eval=states[:T_pred],
        pred_paths=plans,
        obs_positions=obs_positions[:T_pred],
        radii=radii,
        constraints_x=constraints_x,
        svg_dir=str(svg_dir),
        svg_prefix=f"system_{suffix}",
        start=0, stop=T_pred, stride=stride,
        camera="follow",
        follow_width=1.0,
        follow_height=1.0,
        legend_outside=True,
        keep_text_as_text=True,
        pad_inches=0.05,
    )

    # ====== 2) Montage ======
    if frame_indices is None:
        frame_indices = [6, 11, 14, 16, 17, 18, 20, 26]
    frame_indices = [k for k in frame_indices if 0 <= k < T_pred]
    if len(frame_indices) == 0:
        frame_indices = list(np.linspace(0, max(T_pred - 1, 0), 9, dtype=int))

    montage_path = svg_dir / f"RNN_snapshots_{suffix}.pdf"
    make_system_obstacle_montage_v1(
        states[:T_pred], plans, obs_positions[:T_pred], radii, constraints_x,
        frame_indices=frame_indices, grid=(3, 3),
        use_empty_cell_for_legend=True,
        label_outer_only=True, ticklabels_outer_only=True,
        legend_auto_scale=True, legend_scale_factor=0.8,
        axis_labelsize=30, tick_fontsize=20, axis_labelpad_xy=(24,24),
        k_fontsize=25, figsize_per_ax=(5.0, 5.0),
        auto_enlarge_when_outer_only=2, gaps_outer_only=(0.01, 0.01),
        k_annotation="inside", k_loc="upper left", k_fmt="k={k}",
        figscale=1, out_path=str(montage_path), dpi=500
    )
    print(f"[montage] saved {montage_path}")
    # ====== 3) Basic timeseries figures ======
    # State trajectory (x vs y) + initial obstacle circles
    fig = plt.figure()
    plt.plot(states[:, 0], states[:, 1], "o-", label=r"trajectory")
    # draw obstacles at t=0
    if obs_positions.shape[0] > 0:
        for (cx, cy), r in zip(obs_positions[0], radii):
            circle = plt.Circle((cx, cy), r, fill=False, linewidth=2, edgecolor="k")
            plt.gca().add_patch(circle)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"State Trajectory")
    plt.axis("equal")
    plt.grid(); plt.legend()
    _savefig(fig, metrics_dir, f"states_trajectory_{suffix}")

    # Actions
    if actions is not None:
        fig = plt.figure()
        plt.plot(actions[:, 0], "o-", label=r"Action 1")
        if actions.shape[1] > 1:
            plt.plot(actions[:, 1], "o-", label=r"Action 2")
        plt.xlabel(r"Time Step $k$"); plt.ylabel(r"Action")
        plt.title(r"Actions Over Time"); plt.grid(); plt.legend()
        _savefig(fig, metrics_dir, f"actions_{suffix}")

    # Stage cost
    if stage_cost is not None:
        fig = plt.figure()
        plt.plot(stage_cost, "o-")
        plt.xlabel(r"Time Step $k$"); plt.ylabel(r"Stage Cost")
        plt.title(r"Stage Cost Over Time"); plt.grid()
        _savefig(fig, metrics_dir, f"stagecost_{suffix}")

    # Velocities
    if states.shape[1] >= 4:
        fig = plt.figure()
        plt.plot(states[:, 2], "o-", label=r"$v_x$")
        plt.plot(states[:, 3], "o-", label=r"$v_y$")
        plt.xlabel(r"Time Step $k$"); plt.ylabel(r"Velocity")
        plt.title(r"Velocities Over Time"); plt.grid(); plt.legend()
        _savefig(fig, metrics_dir, f"velocity_{suffix}")

    # Alphas
    if alphas is not None:
        fig = plt.figure()
        A = np.squeeze(alphas)
        if A.ndim == 1:
            plt.plot(A, "o-", label=r"$i=1$")
        else:
            for i in range(A.shape[1]):
                plt.plot(A[:, i], "o-", label=rf"$i={i+1}$")
        plt.xlabel(r"Time Step $k$", fontsize = 25); plt.ylabel(r"$\gamma_{k,i}^{\text{RNN}}$", fontsize = 25)
        #plt.title(r"Neural-Network Outputs $\alpha_i(x_k)$")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(); plt.legend(loc="lower left", fontsize=14) #fontsize="small")
        _savefig(fig, metrics_dir, f"alpha_{suffix}")

    # h(x) time series + colored scatter
    if hx is not None:
        N = hx.shape[0]
        for i in range(hx.shape[1]):
            fig = plt.figure()
            plt.plot(hx[:, i], "o-", label=rf"$h_{{{i+1}}}(x_k)$")
            plt.xlabel(r"Time Step $k$"); plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
            plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
            plt.grid()
            _savefig(fig, metrics_dir, f"hx_obstacle_{i+1}_{suffix}")

            # colored-by-iteration
            fig = plt.figure()
            cmap = cm.get_cmap("nipy_spectral", N)
            norm = Normalize(vmin=0, vmax=N-1)
            sc1 = plt.scatter(np.arange(N), hx[:, i], c=np.arange(N), cmap=cmap, norm=norm, s=20)
            plt.xlabel(r"Time Step $k$", fontsize = 20); plt.ylabel(rf"$h_{{{i+1}}}(x_k)$", fontsize = 20)
            #plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ colored by iteration")
            # plt.colorbar(label=r"Iteration $k$")
            cb1 = plt.colorbar(sc1, label=r'Time Step $k$') 
            cb1.set_label('Time Step $k$', fontsize=16)       # label font size
            cb1.ax.tick_params(labelsize=12)  
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid()
            _savefig(fig, metrics_dir, f"hx_colored_obstacle_{i+1}_{suffix}")
    if slacks_eval is not None:
        # slacks_eval shape: (T, m, N)  -> T time steps, m obstacles, N horizon
        T, m, Nh = slacks_eval.shape
        t_eval = np.arange(T)
        for oi in range(m):
            fig = plt.figure(figsize=(10, 4))
            for j in range(Nh):
                plt.plot(t_eval, slacks_eval[:, oi, j], marker="o", linewidth=1.2,
                        label=rf"$\sigma_{{{j+1},{oi+1}}}$")
            plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
            plt.xlabel(r"Time Step $k$", fontsize = 20)
            plt.ylabel(rf"Slack $\sigma_{{j,{oi+1}}}(k)$", fontsize = 20)
            # plt.title(rf"Obstacle {oi+1}: slacks across prediction horizon")
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(ncol=min(3, Nh), fontsize=14) #fontsize="small")#, fontsize="small")
            plt.tight_layout()
            _savefig(fig, metrics_dir, f"slack_obs{oi+1}_{suffix}")        

def plot_simulation_before(path_or_dir: str | Path, out_root: str | Path = "thesis_plots",
                           **kwargs) -> None:
    p = Path(path_or_dir).expanduser().resolve()
    npz = p / "simulation_before.npz" if p.is_dir() else p
    plot_simulation_npz(npz, out_root=out_root, **kwargs)

def plot_simulation_after(path_or_dir: str | Path, out_root: str | Path = "thesis_plots",
                          **kwargs) -> None:
    p = Path(path_or_dir).expanduser().resolve()
    npz = p / "simulation_after.npz" if p.is_dir() else p
    plot_simulation_npz(npz, out_root=out_root, **kwargs)

if __name__ == "__main__":
    # Example usage (replace with your actual paths)
    nn_file = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_NN_mult_move_obj\NNSigmoid_42_noP_diag_stagecost_54554049.82\training_data_nn.npz"
    rnn_file = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_331_stagecost_43554669.31\training_data.npz"
    # plot_stagecost_compare(nn_file, rnn_file)
    
    # nn_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_331_stagecost_43554669.31\thesis_data_rnn\simulation_before.npz"
    # nn_file_after_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_331_stagecost_43554669.31\thesis_data_rnn\simulation_after.npz"

    rnn_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_328_stagecost_79133737.14\thesis_data_rnn\simulation_before.npz"
    rnn_file_after_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_328_stagecost_79133737.14\thesis_data_rnn\simulation_after.npz"

#     plot_simulation_before(rnn_file_before_sim, out_root="thesis_plots")
#     plot_simulation_after(rnn_file_after_sim,  out_root="thesis_plots")
    
#     make_montage_pdf_2x4(
#     rnn_file_before_sim,
#     rnn_file_after_sim,
#     out_root="thesis_plots",
#     frame_indices=[10, 11, 14, 16, 17, 18, 20, 26],  # exactly 8 frames
#     figsize_per_ax=(10, 6), figscale=1,
#     #wspace=0.02, hspace=0.08,        # tighten or loosen gaps
#     tick_fontsize=25, axis_labelsize=30, k_fontsize=30,
#     legend_fontsize=30, legend_max_cols=5,
#     follow=True, follow_width=1, follow_height=1,
# )
    
    #     # NEW: save GIFs
    # save_rnn_gif(
    # rnn_file_before_sim,
    # out_root="thesis_plots",
    # camera="follow", follow_width=2.6, follow_height=2.6,
    # fps=12, dpi=140, stride=1, save_mp4=False,
    # # optional segment slow-down between k=9..20:
    # segment_range=(9, 20), segment_fps=3,   # slower inside segment
    # )

    # save_rnn_gif(
    #     rnn_file_after_sim,
    #     out_root="thesis_plots",
    #     camera="follow", follow_width=2.6, follow_height=2.6,
    #     fps=12, dpi=140, stride=1, save_mp4=False,
    #     # or speed up the same segment:
    #     segment_range=(9, 20), segment_fps=3,  # faster inside segment
    # )
    
# make_montage_pdf(
#     rnn_file_before_sim,
#     rnn_file_after_sim,
#     out_root="thesis_plots",
#     frame_indices=[6, 11, 14, 16, 17, 18, 20, 26],
#     figsize_per_ax=(8, 10),
#     figscale=0.7,
#     follow=True, follow_width=1.5, follow_height=1.5,  # tighter zoom
#     tick_fontsize=15, axis_labelsize=20, k_fontsize=25, # bigger text sizes
#     legend_fontsize=25, legend_max_cols=5,               # wrap sooner if needed
#     wspace=0.0,                 # packed
#     left_pad=0.01, right_pad=0.01,
#     y_axis_first_col_only=True, # keep Y only on far left
#     x_axis_on_all=True,    
# )

nn_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_NN_mult_move_obj\NNSigmoid_39_stagecost_54109949.35\thesis_data_nn\simulation_before.npz"
nn_file_after_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_NN_mult_move_obj\NNSigmoid_39_stagecost_54109949.35\thesis_data_nn\simulation_after.npz"


make_montage_pdf_2x4_compare(
    nn_before_npz=nn_file_before_sim,
    nn_after_npz=nn_file_after_sim,
    rnn_before_npz=rnn_file_before_sim,
    rnn_after_npz=rnn_file_after_sim,
    out_root="thesis_plots",
    frame_indices=[10, 11, 14, 16, 17, 18, 20, 26],  # exactly 8 frames
    figsize_per_ax=(10, 4.5),
    figscale=1.0,
    tick_fontsize=25,
    axis_labelsize=30,
    k_fontsize=30,
    legend_fontsize=35,
    legend_max_cols=5,
    follow=True,
    follow_width=1.0,
    follow_height=1.0,
    nn_label="NN-CBF",
    rnn_label="RNN-CBF",
    label_fontsize=40,
)