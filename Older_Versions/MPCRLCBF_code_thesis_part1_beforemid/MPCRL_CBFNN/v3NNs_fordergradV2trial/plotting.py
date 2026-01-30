import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,                         # use LaTeX for *all* text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],     # matches default LaTeX
    "mathtext.fontset": "cm",                    # fallback if usetex is off
    "axes.unicode_minus": False,                 # proper minus sign when TeX is off
    # Optional but handy:
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
    
    "axes.labelsize": 24,   # x and y axis labels
    "axes.titlesize": 24,   # subplot titles (plt.title / ax.set_title)
    "xtick.labelsize": 18,  # x tick labels
    "ytick.labelsize": 18,  # y tick labels
})

# ==============================================================
#                        UTILITIES
# ==============================================================

def _savefig(fig, out_dir, name, ext=".pdf"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}{ext}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def _traj_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    d = np.diff(xy, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))

def _meta(data, k, default=None):
    key = f"meta__{k}"
    if key in data.files:
        v = np.asarray(data[key])
        if v.size == 1:
            if np.issubdtype(v.dtype, np.number):
                return float(v.ravel()[0])
            return str(v.ravel()[0])
        return v
    return default

def _detect_kind(files: set[str]) -> str:
    if "alphas" in files:
        return "nn"
    if "omegas" in files:
        return "optd"
    return "unknown"

# ==============================================================
#                     CUSTOM COLOR MAP
# ==============================================================

def emphasized_cmap(global_T: int,
                    early_n: int = 100,
                    tail_span: float = 0.1,    # fraction of cmap reserved for k>early_n
                    gamma: float = 1.2,        # >1 slows color change before early_n
                    base: str = "nipy_spectral"
                   ) -> LinearSegmentedColormap:
    """
    Use full base colormap (purple→blue→cyan→green→yellow→red) within 0..early_n,
    then only the final 'tail_span' (red/orange) after early_n. Continuous at the split.
    """
    global_T = max(int(global_T), 2)
    early_n = int(min(max(early_n, 1), global_T - 1))
    early_frac = early_n / (global_T - 1)
    base_cmap = cm.get_cmap(base)

    t = np.linspace(0, 1, global_T)    # normalized index along time
    out = np.empty_like(t)

    first_end = 1.0 - float(tail_span)  # end of the pre-tail mapping

    # First segment: [0, early_frac] -> [0, first_end] with gamma shaping
    mask = t <= early_frac
    if np.any(mask):
        out[mask] = (t[mask] / early_frac) ** gamma * first_end

    # Tail segment: (early_frac, 1] -> [first_end, 1] (continuous)
    if np.any(~mask):
        out[~mask] = first_end + (t[~mask] - early_frac) / (1 - early_frac) * float(tail_span)

    colors = base_cmap(out)
    return LinearSegmentedColormap.from_list("emphasized_nipy_continuous", colors)

def multiples_100_ticks(global_T: int):
    if global_T <= 1:
        return [0]
    ticks = list(range(0, global_T, 100))
    if (global_T - 1) not in ticks:
        ticks.append(global_T - 1)
    return ticks

def mark_colorbar_boundaries(cb, global_T: int, show_external_markers: bool = True):
    """
    Put ticks at each 100 iterations. Draw short markers OUTSIDE the bar so the gradient stays clean.
    """
    ticks = multiples_100_ticks(global_T)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))

    if not show_external_markers:
        return
    for k in ticks[1:]:  # skip 0
        y = (k) / max(global_T - 1, 1)
        cb.ax.plot([1.02, 1.08], [y, y], transform=cb.ax.transAxes,
                   color='k', lw=0.8, alpha=0.8, clip_on=False)

# ==============================================================
#                      MAIN PLOT FUNCTION
# ==============================================================

def plot_simulation_npz(npz_path: str | Path,
                        out_root: str | Path = "thesis_plots",
                        prefer_ext: str = ".pdf",
                        global_T: int | None = None) -> None:
    """
    Loads an NPZ and produces plots for both NN (alphas) and OPTD (omegas)
    using one shared colormap & normalization across all runs.
    """
    npz_path = Path(npz_path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)
    files = set(data.files)
    print(f"[loaded] {npz_path}")
    print(f"[keys]   {sorted(files)}")

    # Required arrays
    states     = np.asarray(data["states"], dtype=np.float64)
    actions    = np.asarray(data["actions"], dtype=np.float64)
    stage_cost = np.asarray(data["stage_cost"], dtype=np.float64).reshape(-1)
    hx         = np.asarray(data["hx"], dtype=np.float64).reshape(-1)
    g_resid    = np.asarray(data["g_resid"], dtype=np.float64).reshape(-1)

    # Detect type
    kind = _detect_kind(files)
    if kind == "nn":
        series_name = r"$\gamma_{k,1}^{\text{NN}}$"
        series_key  = "alphas"
        series      = np.asarray(data["alphas"], dtype=np.float64).reshape(-1)
    elif kind == "optd":
        series_name = r"$\omega^{\star}$"
        series_key  = "omegas"
        series      = np.asarray(data["omegas"], dtype=np.float64).reshape(-1)
    else:
        raise ValueError("NPZ must contain either 'alphas' or 'omegas'.")

    # Meta (optional)
    obs_center = _meta(data, "obs_center", np.array([-2.0, -2.25], float))
    obs_radius = float(_meta(data, "obs_radius", 1.5))
    xlim       = _meta(data, "xlim", np.array([-5.0, 0.0], float))
    ylim       = _meta(data, "ylim", np.array([-5.0, 0.0], float))
    run_tag    = str(_meta(data, "run_tag", "before")).lower()
    Pw         = _meta(data, "Pw", np.nan)
    omega0     = _meta(data, "omega0", np.nan)

    # Derived
    T = states.shape[0]
    iters = np.arange(T, dtype=int)
    vx = states[:, 2] if states.shape[1] >= 3 else np.zeros(T)
    vy = states[:, 3] if states.shape[1] >= 4 else np.zeros(T)

    # One shared colormap & norm
    if global_T is None:
        global_T = T
    cmap = emphasized_cmap(global_T, early_n=100, tail_span=0.1, gamma=1.2, base="nipy_spectral_r")
    norm = Normalize(vmin=0, vmax=max(global_T - 1, 1))

    # Margins
    Lm = min(len(hx) - 1, len(series))
    if kind == "nn":
        margin = hx[1:1+Lm] - hx[:Lm] + series[:Lm] * hx[:Lm]
    else:
        margin = hx[1:1+Lm] - (1.0 - series[:Lm]) * hx[:Lm]
    Lc = min(len(margin), len(g_resid))
    margin = margin[:Lc]
    g_resid_aligned = g_resid[:Lc]
    iters_margin = np.arange(Lc)

    # Output
    out_root = Path(out_root).expanduser().resolve()
    metrics_dir = out_root / f"metrics_{run_tag}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- figures ----------------

    # Trajectory
    fig = plt.figure()
    label = "trajectory"
    if kind == "optd" and np.isfinite(Pw) and np.isfinite(omega0):
        label = f"Trajectory (Pw={Pw:.0f}, $\\omega_0$={omega0:.3g})"
    plt.plot(states[:, 0], states[:, 1], "o-", label=label)
    circle = plt.Circle(tuple(obs_center), obs_radius, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim(xlim); plt.ylim(ylim)
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("Trajectories")
    plt.legend(); plt.axis("equal"); plt.grid(True)
    _savefig(fig, metrics_dir, f"{kind}_states_MPCregular_{run_tag}", ext=prefer_ext)

    # Actions
    fig = plt.figure()
    if actions.ndim == 2 and actions.shape[1] >= 1: plt.plot(actions[:, 0], "o-", label="Action 1")
    if actions.ndim == 2 and actions.shape[1] >= 2: plt.plot(actions[:, 1], "o-", label="Action 2")
    plt.xlabel("Time Step $k$"); plt.ylabel("Action")
    plt.title("Actions"); plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_actions_MPCregular_{run_tag}", ext=prefer_ext)

    # Stage cost
    fig = plt.figure()
    plt.plot(stage_cost, "o-", label="stage cost")
    plt.xlabel("Time Step $k$"); plt.ylabel("Cost")
    plt.title("Stage Cost"); plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_stagecost_MPCregular_{run_tag}", ext=prefer_ext)

    # Alpha/Omega
    fig = plt.figure()
    plt.plot(series, "o-", label=series_name)
    plt.xlabel("Time Step $k$"); plt.ylabel(series_name)
    plt.title(series_name); plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_{series_key}_MPCregular_{run_tag}", ext=prefer_ext)

    # Velocity
    fig = plt.figure()
    plt.plot(vx, "o-", label="Velocity x")
    plt.plot(vy, "o-", label="Velocity y")
    plt.xlabel("Time Step $k$"); plt.ylabel("Velocity")
    plt.title("Velocity Plot"); plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_velocity_MPCregular_{run_tag}", ext=prefer_ext)

    # h(x)
    fig = plt.figure()
    plt.plot(hx, "o-", label=r"$h(x_k)$")
    plt.xlabel("Time Step $k$"); plt.ylabel(r"$h(x_k)$")
    plt.title(r"$h(x_k)$"); plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_Hx_MPCregular_{run_tag}", ext=prefer_ext)

    # Margin vs threshold
    fig = plt.figure()
    plt.plot(margin, "o-", label="margin")
    plt.plot(g_resid_aligned, "o-", label="g_resid")
    plt.axhline(0, color='r', linestyle='--', label='safety threshold')
    plt.xlabel(r'Time Step $k$')
    plt.ylabel(r'$h(x_{k+1}) - (1-\alpha_k)\,h(x_k)$')
    plt.title('CBF Safety Margin over Time')
    plt.legend(); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_marghx_MPCregular_{run_tag}", ext=prefer_ext)

    # ----- colored variants (global cmap/norm) -----

    # Trajectory colored
    fig = plt.figure()
    plt.plot(states[:, 0], states[:, 1], color='gray', alpha=0.5)
    sc = plt.scatter(states[:, 0], states[:, 1], c=iters, cmap=cmap, norm=norm, s=40)
  
    # cb = plt.colorbar(sc)
    # cb.set_label('Time Step $k$', fontsize=22)
    # cb.ax.tick_params(labelsize=18)
    # mark_colorbar_boundaries(cb, global_T)
    
    circle = plt.Circle(tuple(obs_center), obs_radius, color='k', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim(xlim); plt.ylim(ylim)
    plt.xlabel('$x$', fontsize=28); plt.ylabel('$y$', fontsize=28)
    plt.axis('equal'); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_states_colored_MPCregular_{run_tag}", ext=prefer_ext)

    # Alpha/Omega colored
    
    # fig = plt.figure()

    # k = np.arange(len(series))
    # series_log = np.log(series + 1e-12)  # log transform

    # plt.plot(k, series_log, color='gray', alpha=0.5)
    # sc2 = plt.scatter(k, series_log,
    #                 c=k, cmap=cmap, norm=norm, s=40)

    # # ----- focus on the small values, ignore spike -----
    # # option 1: ignore points above a threshold in the ORIGINAL scale
    # small_mask = series < 0.6          # tweak 0.6 to whatever “non-spike” is
    # # option 2 (alternative): drop the largest N values
    # # small_mask = series < np.partition(series, -3)[-3]   # keep all but top-3

    # ymin = series_log[small_mask].min()
    # ymax = series_log[small_mask].max()
    # pad  = 0.05 * (ymax - ymin + 1e-9)
    # plt.ylim(ymin - pad, ymax + pad)
    # # ----------------------------------------------

    # plt.xlabel('Time Step $k$', fontsize=28)
    # plt.ylabel(r'$\log(\omega^\star)$', fontsize=28)
    # plt.grid(True)
    # plt.tight_layout()

    # _savefig(fig, metrics_dir,
    #         f"{kind}_{series_key}_colored_LOGFOCUSED_MPCregular_{run_tag}",
    #         ext=prefer_ext)
    fig = plt.figure()
    
    plt.plot(series, color='gray', alpha=0.5)
    sc2 = plt.scatter(np.arange(len(series)), series,
                      c=np.arange(len(series)), cmap=cmap, norm=norm, s=40)
    # cb2 = plt.colorbar(sc2)
    # cb2.set_label('Time Step $k$', fontsize=20)
    # cb2.ax.tick_params(labelsize=20)
    # mark_colorbar_boundaries(cb2, global_T)
    plt.xlabel('Time Step $k$', fontsize=28); plt.ylabel(series_name, fontsize=28)
    plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_{series_key}_colored_MPCregular_{run_tag}", ext=prefer_ext)

    #  # Alpha/Omega colored  + zoomed inset on early spike
    # fig, ax = plt.subplots()

    # k = np.arange(len(series))

    # # main plot
    # ax.plot(series, color='gray', alpha=0.5)
    # sc2 = ax.scatter(k, series,
    #                 c=k, cmap=cmap, norm=norm, s=40)

    # ax.set_xlabel('Time Step $k$', fontsize=28)
    # ax.set_ylabel(series_name, fontsize=28)
    # ax.grid(True)

    # # ---- inset zoom on spike (k ≈ 0…25) ----
    # # make inset bigger by using larger percentages
    # axins = inset_axes(ax, width="65%", height="65%",
    #                loc="upper right",
    #                borderpad=1.0)   # ↑ increase/decrease to move inset away from axes frame

    # # same data in inset: line + dots
    # axins.plot(k, series, color='gray', alpha=0.7)
    # axins.scatter(k, series,
    #             c=k, cmap=cmap, norm=norm, s=25)

    # # zoom range in k
    # k_max_zoom = min(18, len(series) - 1)
    # x1, x2 = -0.5, k_max_zoom + 0.5
    # axins.set_xlim(x1, x2)

    # # y-limits based on early values (with some padding so points don’t sit on border)
    # ywin = series[:k_max_zoom + 1]
    # y_min = float(np.min(ywin))
    # y_max = float(np.max(ywin))
    # pad = 0.07 * (y_max - y_min + 1e-9)   # ↑ increase this to move box further from points
    # y1, y2 = y_min - pad, y_max + pad
    # axins.set_ylim(y1, y2)

    # # axis ticks + numbers, but no axis labels
    # axins.tick_params(labelleft=True, labelbottom=True)
    # axins.grid(True, alpha=0.4)

    # # SOLID black border for inset
    # for spine in axins.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linestyle('-')
    #     spine.set_linewidth(1.4)

    # # --------- ZOOM BOX ON MAIN AXES ---------
    # # dashed rectangle around zoom region (x1..x2, y1..y2)
    # zoom_rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
    #                     fill=False, ec='black', ls='--', lw=1.2)
    # ax.add_patch(zoom_rect)

    # # --------- CUSTOM CONNECTOR LINES ---------
    # # 1) top-right of zoom box  -> top-left of inset
    # con1 = ConnectionPatch(xyA=(x2, y2), coordsA=ax.transData,
    #                     xyB=(0, 1),  coordsB=axins.transAxes,
    #                     color='black', lw=1.2, ls='--')
    # ax.add_artist(con1)

    # # 2) bottom-right of zoom box -> bottom-left of inset
    # con2 = ConnectionPatch(xyA=(x2, y1), coordsA=ax.transData,
    #                     xyB=(0, 0),  coordsB=axins.transAxes,
    #                     color='black', lw=1.2, ls='--')
    # ax.add_artist(con2)

    # plt.tight_layout()
    # _savefig(fig, metrics_dir, f"{kind}_{series_key}_colored_MPCregular_{run_tag}", ext=prefer_ext)
    
    # h(x) colored
    fig = plt.figure()
    plt.plot(hx, color='gray', alpha=0.5)
    sc3 = plt.scatter(np.arange(len(hx)), hx,
                      c=np.arange(len(hx)), cmap=cmap, norm=norm, s=40)
    cb3 = plt.colorbar(sc3)
    cb3.set_label('Time Step $k$', fontsize=28)
    cb3.ax.tick_params(labelsize=13)
    mark_colorbar_boundaries(cb3, global_T)
    plt.xlabel('Time Step $k$', fontsize=28); plt.ylabel(r'$h(x_k)$', fontsize=28)
    plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_Hx_colored_MPCregular_{run_tag}", ext=prefer_ext)

    # Margin colored
    fig = plt.figure()
    plt.plot(margin, color='gray', alpha=0.5)
    sc4 = plt.scatter(iters_margin, margin,
                      c=iters_margin, cmap=cmap, norm=norm, s=40)
    cb4 = plt.colorbar(sc4)
    cb4.set_label('Time Step $k$', fontsize=16)
    cb4.ax.tick_params(labelsize=13)
    mark_colorbar_boundaries(cb4, global_T)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(r'Time Step $k$'); plt.ylabel(r'$h(x_{k+1}) - (1-\alpha_k)\,h(x_k)$')
    plt.title('Margin Colored by Time Step'); plt.grid(True); plt.tight_layout()
    _savefig(fig, metrics_dir, f"{kind}_marghx_colored_MPCregular_{run_tag}", ext=prefer_ext)

    # Summaries
    traj_len = _traj_length(states[:, :2])
    total_cost = float(np.sum(stage_cost))
    print(f"Total trajectory length: {traj_len:.3f} units")
    print(f"Stage cost sum:         {total_cost:.3f}")

# ==============================================================
#                BEFORE/AFTER WRAPPERS + HELPERS
# ==============================================================

def plot_simulation_before(optd_npz_or_dir, nn_npz_or_dir, out_root="thesis_plots", prefer_ext=".pdf", global_T=None):
    p = Path(optd_npz_or_dir).expanduser().resolve()
    plot_simulation_npz(p / "simulation_before.npz" if p.is_dir() else p, out_root=out_root, prefer_ext=prefer_ext, global_T=global_T)
    p = Path(nn_npz_or_dir).expanduser().resolve()
    plot_simulation_npz(p / "simulation_before.npz" if p.is_dir() else p, out_root=out_root, prefer_ext=prefer_ext, global_T=global_T)

def plot_simulation_after(optd_npz_or_dir, nn_npz_or_dir, out_root="thesis_plots", prefer_ext=".pdf", global_T=None):
    p = Path(optd_npz_or_dir).expanduser().resolve()
    plot_simulation_npz(p / "simulation_after.npz" if p.is_dir() else p, out_root=out_root, prefer_ext=prefer_ext, global_T=global_T)
    p = Path(nn_npz_or_dir).expanduser().resolve()
    plot_simulation_npz(p / "simulation_after.npz" if p.is_dir() else p, out_root=out_root, prefer_ext=prefer_ext, global_T=global_T)

def _len_for_colormap(npz_path: str | Path) -> int:
    npz_path = Path(npz_path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as d:
        return int(len(d["hx"])) if "hx" in d.files else int(len(d["states"]))

def _max_global_T(paths: list[str | Path]) -> int:
    return max(_len_for_colormap(p) for p in paths)

def _infer_label_prefix(npz_path: Path, default: str = "train") -> str:
    """
    Try to infer a short label ('nn', 'optd', ...) from the path.
    Falls back to `default` if nothing obvious is found.
    """
    s = str(npz_path).lower()
    if "nn" in s and "optd" not in s:
        return "nn"
    if "optd" in s:
        return "optd"
    return default


def plot_training_P_npz(npz_path: str | Path,
                        out_root: str | Path = "thesis_plots",
                        prefer_ext: str = ".pdf",
                        label_prefix: str | None = None) -> None:
    """
    Load a training_data.npz and plot ONLY the diagonal entries of P over
    the parameter-update index, then save that figure.

    Expects 'params_history_P' in the NPZ, with shape (n_updates, 4, 4)
    as in your training code.
    """
    npz_path = Path(npz_path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)

    if "params_history_P" not in data.files:
        raise KeyError(f"'params_history_P' not found in {npz_path}")

    params_history_P = np.asarray(data["params_history_P"], dtype=np.float64)

    # updates on the x-axis
    updates = np.arange(params_history_P.shape[0])

    # name / label shorthands
    if label_prefix is None:
        label_prefix = _infer_label_prefix(npz_path)

    # where to save
    out_root = Path(out_root).expanduser().resolve()
    train_dir = out_root / "training_P"
    train_dir.mkdir(parents=True, exist_ok=True)

    # ---- recreate your P-plot ----
    figP = plt.figure(figsize=(10, 6))
    plt.plot(params_history_P[:, 0, 0], label=r"$i=1$")
    plt.plot(params_history_P[:, 1, 1], label=r"$i=2$")
    plt.plot(params_history_P[:, 2, 2], label=r"$i=3$")
    plt.plot(params_history_P[:, 3, 3], label=r"$i=4$")

    plt.xlabel("Update Number", fontsize=38)
    plt.ylabel(r"$F_{\theta_{i,i}}$", fontsize=38)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=23)
    plt.grid(True)
    plt.tight_layout()

    # filename: e.g. 'nn_P_training', 'optd_P_training'
    base_name = f"{label_prefix}_P_training"
    _savefig(figP, train_dir, base_name, ext=prefer_ext)


def plot_training_P_both(optd_npz: str | Path,
                         nn_npz: str | Path,
                         out_root: str | Path = "thesis_plots",
                         prefer_ext: str = ".pdf") -> None:
    """
    Convenience wrapper: plot P-evolution for both OPTD and NN
    training_data.npz files.
    """
    plot_training_P_npz(optd_npz, out_root=out_root,
                        prefer_ext=prefer_ext, label_prefix="optd")
    plot_training_P_npz(nn_npz,  out_root=out_root,
                        prefer_ext=prefer_ext, label_prefix="nn")

# ==============================================================
#                            MAIN
# ==============================================================

if __name__ == "__main__":
    nn_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFNN\v3NNs_fordergradV2trial\NNforderADAM_43_stagecost_6661.54\thesis_data_mpcregular\mpc_regular_before.npz"
    nn_file_after_sim  = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFNN\v3NNs_fordergradV2trial\NNforderADAM_43_stagecost_6661.54\thesis_data_mpcregular\mpc_regular_after.npz"

    optd_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFOPTDforder\OPTDforder_RMSprop_trial_45_stagecost_7130.67\thesis_data_mpcregular\mpc_regular_before.npz"
    optd_file_after_sim  = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFOPTDforder\OPTDforder_RMSprop_trial_45_stagecost_7130.67\thesis_data_mpcregular\mpc_regular_after.npz"

    # One global T for consistent colors everywhere
    global_T = _max_global_T([
        nn_file_before_sim, nn_file_after_sim,
        optd_file_before_sim, optd_file_after_sim
    ])
    print(f"[colormap] global_T = {global_T}")

    # Change prefer_ext to ".svg" if you want SVGs
    plot_simulation_before(optd_file_before_sim, nn_file_before_sim,
                           out_root="thesis_plots_4", prefer_ext=".pdf", global_T=global_T)
    plot_simulation_after(optd_file_after_sim,  nn_file_after_sim,
                           out_root="thesis_plots_4", prefer_ext=".pdf", global_T=global_T)
    
    # nn_training_npz = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFNN\v3NNs_fordergradV2trial\NNforderADAM_43_stagecost_6661.54\training_data.npz"
    nn_training_npz = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFNN\v3NNs_fordergradV2trial\NNforderADAM_33_stagecost_6971.54\training_data.npz"
    optd_training_npz = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part1_beforemid\MPCRL_CBFOPTDforder\OPTDforder_RMSprop_trial_45_stagecost_7130.67\training_data.npz"

    plot_training_P_both(optd_training_npz, nn_training_npz,
                         out_root="thesis_plots_4", prefer_ext=".pdf")