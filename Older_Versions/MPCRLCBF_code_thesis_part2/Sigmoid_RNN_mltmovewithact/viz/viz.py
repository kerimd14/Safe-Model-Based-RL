# viz/viz.py
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
import matplotlib.pyplot as plt


def save_notes(experiment_folder, notes, filename="notes.txt"):
    os.makedirs(experiment_folder, exist_ok=True)
    notes_path = os.path.join(experiment_folder, filename)
    with open(notes_path, "w") as file:
        file.write(notes)

def save_figures(figures, experiment_folder, save_in_subfolder=False):
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
            # add .svg if user forgot extension
            if not any(filename.lower().endswith(ext) for ext in [".png", ".svg", ".pdf", ".jpg", ".jpeg"]):
                filename = filename + ".svg"

            file_path = os.path.join(target_folder, filename)
            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Figure saved as: {file_path}")
    else:
        print("Figure not saved")


def plot_B_update(B_update_history, experiment_folder):
    """
    B_update_history is a history of update vectors for the RL parameters
    """
    if len(B_update_history) == 0:
        return

    B_update = np.asarray(B_update_history)
    B_update = np.squeeze(B_update)

    # Build labels for the first four diagonal P elements (kept like your original)
    labels = [f"P[{i},{i}]" for i in range(4)]
    print(f"labels: {labels}")

    # The remaining columns correspond to RNN parameter updates
    if B_update.shape[1] > 4:
        nn_B_update = B_update[:, 4:]
        mean_mag = np.mean(np.abs(nn_B_update), axis=1)
    else:
        mean_mag = None

    # Plot updates for P parameters
    fig_p = plt.figure()
    for idx, lbl in enumerate(labels):
        if idx < B_update.shape[1]:
            plt.plot(B_update[:, idx], "o-", label=lbl)
    plt.xlabel("Update iteration")
    plt.ylabel("B_update")
    plt.title("P parameter B_update over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figures([(fig_p, "P_B_update_over_time.svg")], experiment_folder, "Learning")

    # Plot the RNN mean
    if mean_mag is not None:
        fig_nn = plt.figure()
        plt.plot(mean_mag, "o-", label="mean abs(NN_B_update)")
        plt.xlabel("Update iteration")
        plt.ylabel("Mean absolute B_update")
        plt.title("RNN mean across RNN params B_update magnitude over training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_figures([(fig_nn, "NN_mean_B_update_over_time.svg")], experiment_folder, "Learning")


def plot_spectral_radius(spectral_radii_hist, experiment_folder):
    """
    spectral_radii_hist shape: (updates, num_recurrent_layers)
    """
    if spectral_radii_hist is None:
        return

    arr = np.asarray(spectral_radii_hist)
    if arr.size == 0:
        return

    fig = plt.figure()
    if arr.ndim == 1:
        plt.plot(arr, "o-", label=r"$\rho(W_{hh})$")
    else:
        for j in range(arr.shape[1]):
            plt.plot(arr[:, j], "o-", label=rf"$\rho(W_{{hh,{j}}})$")
    plt.axhline(1, linestyle="--", label="target ρ")
    plt.xlabel("Update iteration")
    plt.ylabel("Spectral radius")
    plt.title("Recurrent spectral radii over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figures([(fig, "spectral_radius_over_time.svg")], experiment_folder, "Learning")


def training_snapshot(
    states,
    actions,
    td_values,
    obs_positions,
    hx_list,
    alphas,
    positions,
    radii,
    constraints_x=None,
    experiment_folder=None,
    step=None,
):
    """
    Create and save plots summarizing training at a given step.
    Saves into experiment_folder/learning_process.
    """
    if experiment_folder is None:
        raise ValueError("training_snapshot: experiment_folder is required")
    if step is None:
        step = 0

    # --- Trajectories of states while policy is trained ---
    figstate = plt.figure()
    plt.plot(states[:, 0], states[:, 1], "o-")

    # Plot the obstacles (static circles)
    for (cx, cy), r in zip(positions, radii):
        circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    # optional window (if you pass CONSTRAINTS_X)
    if constraints_x is not None:
        # constraints_x can be scalar or vector; support both
        try:
            Xmax = float(constraints_x[0])
            Ymax = float(constraints_x[1])
            plt.xlim([-Xmax, 0])
            plt.ylim([-Ymax, 0])
        except Exception:
            pass

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"Trajectories of states while policy is trained$")
    plt.axis("equal")
    plt.grid()
    save_figures([(figstate, f"position_plotat_{step}.svg")], experiment_folder, "Learning")

    # --- Velocity plot ---
    figvelocity = plt.figure()
    if states.shape[1] >= 4:
        plt.plot(states[:, 2], "o-", label=r"Velocity x")
        plt.plot(states[:, 3], "o-", label=r"Velocity y")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Velocity Value")
    plt.title(r"Velocity Plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    save_figures([(figvelocity, f"figvelocity{step}.svg")], experiment_folder, "Learning")

    # --- TD plot (log scale) ---
    figtdtemp = plt.figure(figsize=(10, 5))
    indices = np.arange(len(td_values))
    plt.scatter(indices, td_values, label=r"TD")
    plt.yscale("log")
    plt.title(r"TD Over Training (Log Scale) - Colored by Proximity")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"TD")
    plt.legend()
    plt.grid(True)
    save_figures([(figtdtemp, f"TD_plotat_{step}.svg")], experiment_folder, "Learning")

    # --- Actions ---
    figactions = plt.figure()
    if actions.ndim == 2 and actions.shape[1] >= 2:
        plt.plot(actions[:, 0], "o-", label=r"Action 1")
        plt.plot(actions[:, 1], "o-", label=r"Action 2")
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"Action")
    plt.title(r"Actions")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    save_figures([(figactions, f"action_plotat_{step}.svg")], experiment_folder, "Learning")

    # --- h(x) plots for each obstacle ---
    if hx_list is not None and hx_list.size > 0:
        for oi in range(hx_list.shape[1]):
            fig_hi = plt.figure()
            plt.plot(hx_list[:, oi], "o", label=rf"$h_{{{oi+1}}}(x_k)$")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(rf"$h_{{{oi+1}}}(x_k)$")
            plt.title(rf"Obstacle {oi+1}: $h_{{{oi+1}}}(x_k)$ Over Time")
            plt.grid()
            save_figures([(fig_hi, f"hx_obstacle_plotat_{oi+1}_{step}.svg")], experiment_folder, "Learning")

    # --- Alphas from RNN ---
    if alphas is not None and np.size(alphas) > 0:
        fig_alpha = plt.figure()
        alphas_np = np.asarray(alphas)
        if alphas_np.ndim == 1:
            plt.plot(alphas_np, "o-", label=r"$\alpha(x_k)$")
        else:
            # common shape is (T, m) or (T,) list of lists
            if alphas_np.ndim >= 2:
                for j in range(alphas_np.shape[1]):
                    plt.plot(alphas_np[:, j], "o-", label=rf"$\alpha_{{{j+1}}}(x_k)$")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"$\alpha_i(x_k)$")
        plt.title(r"Neural-Network Outputs $\alpha_i(x_k)$")
        plt.grid()
        plt.legend(loc="upper right", fontsize="small")
        save_figures([(fig_alpha, f"alpha_plotat_{step}.svg")], experiment_folder, "Learning")


def plot_training_curves(
    params_history_P,
    sum_stage_cost_history,
    TD_history,
    experiment_folder,
    spectral_radii_hist=None,
    stage_cost_valid=None,
):
    """
    End-of-training plots + npz saving.

    params_history_P: (num_updates, ns, ns) or list of matrices
    sum_stage_cost_history: (num_episodes,) or list
    TD_history: (num_episodes,) or list
    spectral_radii_hist: (num_updates, num_recurrent_layers)
    stage_cost_valid: list of evaluation stage costs (optional)
    """
    params_history_P = np.asarray(params_history_P)
    TD_history = np.asarray(TD_history)
    sum_stage_cost_history = np.asarray(sum_stage_cost_history)

    # --- P diagonal plot ---
    figP = plt.figure(figsize=(10, 5))
    if params_history_P.ndim == 3 and params_history_P.shape[1] >= 4:
        plt.plot(params_history_P[:, 0, 0], label=r"$P_{1,1}$")
        plt.plot(params_history_P[:, 1, 1], label=r"$P_{2,2}$")
        plt.plot(params_history_P[:, 2, 2], label=r"$P_{3,3}$")
        plt.plot(params_history_P[:, 3, 3], label=r"$P_{4,4}$")
    plt.xlabel(r"Update Number", fontsize=20)
    plt.ylabel(r"Value", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    save_figures([(figP, "P.svg")], experiment_folder)

    # --- Stage cost over episodes (log) ---
    figstagecost = plt.figure()
    plt.plot(sum_stage_cost_history, "o", label=r"Stage Cost")
    plt.yscale("log")
    plt.xlabel(r"Episode Number", fontsize=20)
    plt.ylabel(r"Stage Cost", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    save_figures([(figstagecost, "stagecost.svg")], experiment_folder)

    # --- TD over episodes (log) ---
    figtd = plt.figure()
    plt.plot(TD_history, "o", label=r"TD")
    plt.yscale("log")
    plt.title(r"TD Over Training (Log Scale)")
    plt.xlabel(r"Episode Number")
    plt.ylabel(r"TD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figures([(figtd, "TD.svg")], experiment_folder)

    # --- Smoothed stage cost (mean/std band) ---
    cost = np.array(sum_stage_cost_history, dtype=float).reshape(-1)
    episodes = np.arange(len(cost))

    window = 100
    s = pd.Series(cost)
    running_mean = s.rolling(window, center=True, min_periods=1).mean().values
    running_std = s.rolling(window, center=True, min_periods=1).std().values

    figstagecost_nice = plt.figure(figsize=(10, 5))
    ax = figstagecost_nice.add_subplot(1, 1, 1)

    ax.plot(episodes, running_mean, "-", linewidth=2, label=rf"Stage Cost mean ({window}-ep)")
    ax.fill_between(
        episodes,
        running_mean - running_std,
        running_mean + running_std,
        alpha=0.3,
        label=rf"Stage Cost std ({window}-ep)",
    )

    if np.any(cost > 0):
        ax.set_yscale("log")

    ax.set_xlabel(r"Episode Number", fontsize=20)
    ax.set_ylabel(r"Stage Cost", fontsize=20)
    ax.tick_params(labelsize=12)
    ax.grid(True)
    ax.legend(fontsize=16)
    figstagecost_nice.tight_layout()
    save_figures([(figstagecost_nice, "stagecost_smoothed.svg")], experiment_folder)

    # --- spectral radius plot ---
    if spectral_radii_hist is not None:
        plot_spectral_radius(spectral_radii_hist, experiment_folder)

    # --- save training data ---
    npz_payload = {
        "episodes": episodes,
        "stage_cost": cost,
        "td": TD_history,
        "running_mean": running_mean,
        "running_std": running_std,
        "smoothing_window": np.array([window], dtype=int),
        "params_history_P": params_history_P,
    }

    if spectral_radii_hist is not None:
        npz_payload["spectral_radii_hist"] = np.asarray(spectral_radii_hist)

    if stage_cost_valid is not None:
        npz_payload["stage_cost_valid"] = np.asarray(stage_cost_valid)

    np.savez_compressed(os.path.join(experiment_folder, "training_data.npz"), **npz_payload)
    print(f"Saved training_data.npz in: {experiment_folder}")


def generate_experiment_notes(experiment_folder, params, params_innit, episode_duration, num_episodes, seed, alpha, sampling_time, gamma, decay_rate, decay_at_end, 
                              noise_scalingfactor, noise_variance, stage_cost_sum_before, stage_cost_sum_after, layers_list, replay_buffer, episode_updatefreq,
                              patience_threshold, lr_decay_factor, horizon, modes, mode_params, positions, radii, 
                              slack_penalty_MPC_L1, slack_penalty_MPC_L2, 
                              slack_penalty_RL_L1, slack_penalty_RL_L2, violation_penalty):
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
    Slack Penalty MPC L1: {slack_penalty_MPC_L1}
    Slack Penalty RL L1: {slack_penalty_RL_L1}
    Slack Penalty MPC L2: {slack_penalty_MPC_L2}
    Slack Penalty RL L2: {slack_penalty_RL_L2}
    Violation Penalty: {violation_penalty}

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
    Initialized params: {params_innit['rnn_params']}
    Learned rnn params: {params['rnn_params']}



    Additional Notes:
    -----------------
    - Off-policy training with initial parameters
    - Noise scaling based on distance to target
    - Decay rate applied to noise over iterations
    - Scaling adjused

    """
    save_notes(experiment_folder, notes)
        
