# rl/evaluator.py
import os
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from config import CONSTRAINTS_X, CONSTRAINTS_U
from viz.animation import make_system_obstacle_animation_v3


class Evaluator:

    def __init__(self, agent, viz):
        """
        agent: RLAgent (has env, mpc, solvers, warmstarts, etc.)
        viz:   viz module/object (has save_figures)
        """
        self.agent = agent
        self.viz = viz

        # evaluation counter (used in filenames)
        self.eval_count = 1

        # keep a history of evaluation stage costs (optional)
        self.stage_cost_valid = []

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

        Returns:
            sum_stage_cost (float):
                Summed stage cost over the evaluation episode.
        """

        # Make sure evaluation folder exists
        target_folder = os.path.join(experiment_folder, "evaluation")
        os.makedirs(target_folder, exist_ok=True)

        # Warmstart hidden state (kept like your original logic)
        hidden_in = self.agent.rnn_warmstart(params)

        # Reset environment and obstacle motion
        state, _ = self.agent.env.reset(seed=self.agent.seed, options={})
        self.agent.obst_motion.reset()

        states_eval = [state]
        actions_eval = []
        stage_cost_eval = []

        xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)

        hx = [
            float(hf(cs.DM(state),
                     xpred_list[0:self.agent.mpc.rnn.obst.obstacle_num],
                     ypred_list[0:self.agent.mpc.rnn.obst.obstacle_num]))
            for hf in self.agent.h_func_list
        ]
        hx_list = [hx]

        # for RNN outputs
        alphas = []

        obs_positions = [self.agent.obst_motion.current_positions()]

        # Warmstart variables storage (evaluation uses VMPC)
        # (We keep only what evaluation touches; agent has a helper for this)
        self.agent.x_prev_VMPC = cs.DM()
        self.agent.lam_x_prev_VMPC = cs.DM()
        self.agent.lam_g_prev_VMPC = cs.DM()

        slacks_eval = []   # will become shape (T, m, N)

        m = self.agent.mpc.rnn.obst.obstacle_num
        N = self.agent.horizon

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

            action, _, hidden_in, alpha, plan_xy, _ = self.agent.v_mpc(
                params=params,
                x=state,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
                hidden_in=hidden_in
            )

            # Save slacks of last solved VMPC (agent stores self.S_VMPC)
            S_now_mN = unflatten_slack(self.agent.S_VMPC, m, N)   # shape (m, N)
            slacks_eval.append(S_now_mN)
            plans_eval.append(plan_xy)

            statsv = self.agent.solver_inst.stats()
            if statsv["success"] == False:
                print("V_MPC NOT SUCCEEDED in EVALUATION")

            alphas.append(alpha)

            action = cs.fmin(cs.fmax(cs.DM(action), -CONSTRAINTS_U), CONSTRAINTS_U)
            stage_cost_eval.append(self.agent.stage_cost(action, state, self.agent.S_VMPC, hx))

            # Environment step
            state, _, done, _, _ = self.agent.env.step(action)

            states_eval.append(state)
            actions_eval.append(action)

            hx = [
                float(hf(cs.DM(state),
                         xpred_list[0:self.agent.mpc.rnn.obst.obstacle_num],
                         ypred_list[0:self.agent.mpc.rnn.obst.obstacle_num]))
                for hf in self.agent.h_func_list
            ]
            hx_list.append(hx)

            # obstacle motion
            _ = self.agent.obst_motion.step()
            xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)
            obs_positions.append(self.agent.obst_motion.current_positions())

        # Convert to numpy
        states_eval = np.array(states_eval)
        actions_eval = np.array(actions_eval)
        stage_cost_eval = np.array(stage_cost_eval).reshape(-1)
        obs_positions = np.array(obs_positions)
        hx_list = np.vstack(hx_list)
        alphas = np.array(alphas)
        slacks_eval = np.stack(slacks_eval, axis=0)
        plans_eval = np.array(plans_eval)

        sum_stage_cost = float(np.sum(stage_cost_eval))
        print(f"Stage Cost: {sum_stage_cost}")

        # -----------------------
        # Plotting / Saving
        # -----------------------

        # States trajectory
        figstates = plt.figure()
        plt.plot(states_eval[:, 0], states_eval[:, 1], "o-")

        # Plot the obstacle
        for (cx, cy), r in zip(self.agent.mpc.rnn.obst.positions, self.agent.mpc.rnn.obst.radii):
            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
            plt.gca().add_patch(circle)

        plt.xlim([-CONSTRAINTS_X[0], 0])
        plt.ylim([-CONSTRAINTS_X[1], 0])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title(r"Trajectories")
        plt.axis("equal")
        plt.grid()

        self.viz.save_figures(
            [(figstates, f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,
            "Evaluation"
        )

        # Actions
        figactions = plt.figure()
        if actions_eval.ndim == 2 and actions_eval.shape[1] >= 2:
            plt.plot(actions_eval[:, 0], "o-", label=r"Action 1")
            plt.plot(actions_eval[:, 1], "o-", label=r"Action 2")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"Action")
        plt.title(r"Actions")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        self.viz.save_figures(
            [(figactions, f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,
            "Evaluation"
        )

        # Stage cost
        figstagecost = plt.figure()
        plt.plot(stage_cost_eval, "o-")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"Cost")
        plt.title(r"Stage Cost")
        plt.grid()
        plt.tight_layout()

        self.viz.save_figures(
            [(figstagecost, f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,
            "Evaluation"
        )

        # Velocity plot
        figsvelocity = plt.figure()
        if states_eval.shape[1] >= 4:
            plt.plot(states_eval[:, 2], "o-", label=r"Velocity x")
            plt.plot(states_eval[:, 3], "o-", label=r"Velocity y")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"Velocity Value")
        plt.title(r"Velocity Plot")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        self.viz.save_figures(
            [(figsvelocity, f"velocity_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,
            "Evaluation"
        )

        # h(x) per obstacle
        for oi in range(hx_list.shape[1]):
            fig_hi = plt.figure()
            plt.plot(hx_list[:, oi], "o", label=rf"$h_{{{oi+1}}}(x_k)$")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(rf"$h_{{{oi+1}}}(x_k)$")
            plt.title(rf"Obstacle {oi+1}: $h_{{{oi+1}}}(x_k)$ Over Time")
            plt.grid()

            self.viz.save_figures(
                [(fig_hi, f"hx_obstacle_{oi+1}_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                experiment_folder,
                "Evaluation"
            )

        # Alphas from RNN
        fig_alpha = plt.figure()
        if alphas.ndim == 1:
            plt.plot(alphas, "o-", label=r"$\alpha(x_k)$")
        else:
            # common: (T, m)
            if alphas.ndim >= 2:
                for j in range(alphas.shape[1]):
                    plt.plot(alphas[:, j], "o-", label=rf"$\alpha_{{{j+1}}}(x_k)$")
        plt.xlabel(r"Iteration $k$")
        plt.ylabel(r"$\alpha_i(x_k)$")
        plt.title(r"Neural-Network Outputs $\alpha_i(x_k)$")
        plt.grid()
        plt.legend(loc="upper right", fontsize="small")

        self.viz.save_figures(
            [(fig_alpha, f"alpha_{self.eval_count}_SC_{sum_stage_cost}.svg")],
            experiment_folder,
            "Evaluation"
        )

        # Slack plots
        T_, m_, N_ = slacks_eval.shape
        t_eval = np.arange(T_)

        for oi in range(m_):
            fig_slack_i = plt.figure(figsize=(10, 4))
            for j in range(N_):
                plt.plot(t_eval, slacks_eval[:, oi, j],
                         label=rf"horizon $j={j+1}$", marker="o", linewidth=1.2)
            plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(rf"Slack $S_{{{oi+1},j}}(k)$")
            plt.title(rf"Obstacle {oi+1}: slacks across prediction horizon")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=min(4, N_), fontsize="small")
            plt.tight_layout()

            self.viz.save_figures(
                [(fig_slack_i, f"slack_obs{oi+1}_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                experiment_folder,
                "Evaluation"
            )

        # -----------------------
        # Evaluation GIF
        # -----------------------

        T_pred = plans_eval.shape[0]
        out_gif = os.path.join(target_folder, f"system_and_obstaclewithobstpred_{self.eval_count}_SC_{sum_stage_cost}.gif")

        make_system_obstacle_animation_v3(
            states_eval=states_eval[:T_pred],
            pred_paths=plans_eval,
            obs_positions=obs_positions[:T_pred],
            radii=self.agent.mpc.rnn.obst.radii,
            constraints_x=CONSTRAINTS_X[0],
            out_path=out_gif,
            trail_len=self.agent.horizon,      # fade the tail to last H
            camera="follow",
            follow_width=1.0,
            follow_height=1.0,
        )

        # Update counters/history
        self.eval_count += 1
        self.stage_cost_valid.append(sum_stage_cost)

        return sum_stage_cost
