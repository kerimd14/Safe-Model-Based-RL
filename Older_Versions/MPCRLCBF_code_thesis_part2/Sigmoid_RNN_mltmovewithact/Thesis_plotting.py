# THE FILE RESPONIBLE FOR PLOTTING THE RESULTS THAT GO IN THE THESIS

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from Functions import make_system_obstacle_svg_frames_v3, make_system_obstacle_montage_v1

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

#     out_path = os.path.join(out_dir, "stagecost_compare_NN_vs_RNN.png")
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"Stage cost comparison plot saved at: {out_path}")
def _savefig(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{name}.svg"
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
    out_path = os.path.join(out_dir, "stagecost_compare_NN_vs_RNN.svg")
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

    montage_path = svg_dir / f"RNN_snapshots_{suffix}.svg"
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
        plt.xlabel(r"Iteration $k$"); plt.ylabel(r"Action")
        plt.title(r"Actions Over Time"); plt.grid(); plt.legend()
        _savefig(fig, metrics_dir, f"actions_{suffix}")

    # Stage cost
    if stage_cost is not None:
        fig = plt.figure()
        plt.plot(stage_cost, "o-")
        plt.xlabel(r"Iteration $k$"); plt.ylabel(r"Stage Cost")
        plt.title(r"Stage Cost Over Time"); plt.grid()
        _savefig(fig, metrics_dir, f"stagecost_{suffix}")

    # Velocities
    if states.shape[1] >= 4:
        fig = plt.figure()
        plt.plot(states[:, 2], "o-", label=r"$v_x$")
        plt.plot(states[:, 3], "o-", label=r"$v_y$")
        plt.xlabel(r"Iteration $k$"); plt.ylabel(r"Velocity")
        plt.title(r"Velocities Over Time"); plt.grid(); plt.legend()
        _savefig(fig, metrics_dir, f"velocity_{suffix}")

    # Alphas
    if alphas is not None:
        fig = plt.figure()
        A = np.squeeze(alphas)
        if A.ndim == 1:
            plt.plot(A, "o-", label=r"$\gamma_1$")
        else:
            for i in range(A.shape[1]):
                plt.plot(A[:, i], "o-", label=rf"$\gamma_{{{i+1}}}$")
        plt.xlabel(r"Iteration $k$", fontsize = 20); plt.ylabel(r"$\gamma \ Values $", fontsize = 20)
        #plt.title(r"Neural-Network Outputs $\alpha_i(x_k)$")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(); plt.legend(loc="upper left", fontsize=14) #fontsize="small")
        _savefig(fig, metrics_dir, f"alpha_{suffix}")

    # h(x) time series + colored scatter
    if hx is not None:
        N = hx.shape[0]
        for i in range(hx.shape[1]):
            fig = plt.figure()
            plt.plot(hx[:, i], "o-", label=rf"$h_{{{i+1}}}(x_k)$")
            plt.xlabel(r"Iteration $k$"); plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
            plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
            plt.grid()
            _savefig(fig, metrics_dir, f"hx_obstacle_{i+1}_{suffix}")

            # colored-by-iteration
            fig = plt.figure()
            cmap = cm.get_cmap("nipy_spectral", N)
            norm = Normalize(vmin=0, vmax=N-1)
            sc1 = plt.scatter(np.arange(N), hx[:, i], c=np.arange(N), cmap=cmap, norm=norm, s=20)
            plt.xlabel(r"Iteration $k$", fontsize = 20); plt.ylabel(rf"$h_{{{i+1}}}(x_k)$", fontsize = 20)
            #plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ colored by iteration")
            # plt.colorbar(label=r"Iteration $k$")
            cb1 = plt.colorbar(sc1, label=r'Iteration $k$') 
            cb1.set_label('Iteration $k$', fontsize=16)       # label font size
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
                        label=rf"$S_{{{oi+1},{j+1}}}$")
            plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
            plt.xlabel(r"Iteration $k$", fontsize = 20)
            plt.ylabel(rf"Slack $S_{{{oi+1},j}}(k)$", fontsize = 20)
            # plt.title(rf"Obstacle {oi+1}: slacks across prediction horizon")
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(ncol=min(4, Nh), fontsize=14) #fontsize="small")#, fontsize="small")
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
    nn_file = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_NN_mult_move_obj\NNSigmoid_31_stagecost_86137060.04\training_data_nn.npz"
    rnn_file = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_313_stagecost_87625408.95\training_data.npz"
    plot_stagecost_compare(nn_file, rnn_file)
    
    nn_file_before_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_328_stagecost_79133737.14\thesis_data_rnn\simulation_before.npz"
    nn_file_after_sim = r"C:\Users\kerim\OneDrive\Documents\Thesis_CODE\MPCRLCBF_code_thesis_part2\Sigmoid_RNN_mult_move_obj\RNN_mult_move_obj_experiment_328_stagecost_79133737.14\thesis_data_rnn\simulation_after.npz"

    plot_simulation_before(nn_file_before_sim, out_root="thesis_plots")
    plot_simulation_after(nn_file_after_sim,  out_root="thesis_plots")