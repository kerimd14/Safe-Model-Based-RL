import os
import numpy as np

from npz_builder import NPZBuilder


def run_before_after(
    runner,
    params_before,
    params_after,
    experiment_folder,
    episode_duration,
    viz=None,     # your viz module/class
    tag_before="before",
    tag_after="after",
):
    os.makedirs(experiment_folder, exist_ok=True)

    # --- BEFORE ---
    res_before = runner.rollout(params_before, episode_duration=episode_duration)
    if viz is not None:
        viz.save_rollout_all(res_before, experiment_folder, tag=tag_before)  # you implement in viz

    # --- AFTER ---
    res_after = runner.rollout(params_after, episode_duration=episode_duration)
    if viz is not None:
        viz.save_rollout_all(res_after, experiment_folder, tag=tag_after)

    # --- save npz (both) ---
    data_dir = os.path.join(experiment_folder, "thesis_data_rnn")
    sim_data = NPZBuilder(data_dir, "simulation", float_dtype="float32")

    sim_data.add(
        states_before=res_before.states,
        actions_before=res_before.actions,
        stage_cost_before=res_before.stage_cost,
        hx_before=res_before.hx,
        alphas_before=res_before.alphas,
        obs_positions_before=res_before.obs_positions,
        plans_before=res_before.plans,
        slacks_eval_before=res_before.slacks_eval,
        lam_g_hist_before=res_before.lam_g_hist,

        states_after=res_after.states,
        actions_after=res_after.actions,
        stage_cost_after=res_after.stage_cost,
        hx_after=res_after.hx,
        alphas_after=res_after.alphas,
        obs_positions_after=res_after.obs_positions,
        plans_after=res_after.plans,
        slacks_eval_after=res_after.slacks_eval,
        lam_g_hist_after=res_after.lam_g_hist,
    )

    npz_path = sim_data.finalize(suffix="before_after")
    print(f"[saved] {npz_path}")

    return float(res_before.stage_cost.sum()), float(res_after.stage_cost.sum())
