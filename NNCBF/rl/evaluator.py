# rl/evaluator_nn.py
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
                Dictionary of system and NN parameters.
            experiment_folder (str):
                Path to the experiment folder where results will be saved.
            episode_duration (int):
                Number of steps in the evaluation episode.
        """
        # Make sure evaluation folder exists
        target_folder = os.path.join(experiment_folder, "evaluation")
        os.makedirs(target_folder, exist_ok=True)

        # Reset environment & obstacle motion
        state, _ = self.agent.env.reset(seed=self.agent.seed, options={})
        self.agent.obst_motion.reset()

        # Prepare storage for rollout data
        states_eval = [state]
        actions_eval = []
        stage_cost_eval = []
        stage_cost_validation = []
        stage_cost_alt = []

        #Pre-compute initial obstacle predictions and initial nn hidden state
        xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)
        obs_positions = [self.agent.obst_motion.current_positions()]

        hx = [ float(hf(cs.DM(state), xpred_list[0:self.agent.mpc.nn.obst.obstacle_num], ypred_list[0:self.agent.mpc.nn.obst.obstacle_num]))
                     for hf in self.agent.h_funcs ]
        hx_list = [hx]

        #for NN outputs
        alphas = []

        # Initialize warmstart variables
        self.agent.x_prev_VMPC        = cs.DM()
        self.agent.lam_x_prev_VMPC    = cs.DM()
        self.agent.lam_g_prev_VMPC    = cs.DM()

        for i in range(episode_duration):
            action, _, alpha= self.agent.V_MPC(params=params, x=state,
                                         xpred_list=xpred_list,
                                         ypred_list=ypred_list)

            statsv = self.agent.solver_inst.stats()
            if statsv["success"] == False:
                print("V_MPC NOT SUCCEEDED in EVALUATION")
            alphas.append(alpha)

            action = cs.fmin(cs.fmax(cs.DM(action), -CONSTRAINTS_U), CONSTRAINTS_U)
            state, _, done, _, _ = self.agent.env.step(action)

            states_eval.append(state)
            actions_eval.append(action)
            stage_cost_eval.append(self.agent.stage_cost(action, state, self.agent.S_VMPC, hx))
            stage_cost_validation.append(self.agent.stage_cost_validation(action, state, hx))
            stage_cost_alt.append(self.agent.stage_cost_alt(action, state))

            hx = [ float(hf(cs.DM(state), xpred_list[0:self.agent.mpc.nn.obst.obstacle_num], ypred_list[0:self.agent.mpc.nn.obst.obstacle_num]))
                     for hf in self.agent.h_funcs ]
            hx_list.append(hx)


            # Advance obstacle motion and re-predict
            _ = self.agent.obst_motion.step()
            xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)
            obs_positions.append(self.agent.obst_motion.current_positions())

        states_eval = np.array(states_eval)
        actions_eval = np.array(actions_eval)
        stage_cost_eval = np.array(stage_cost_eval)
        stage_cost_eval = stage_cost_eval.reshape(-1)
        obs_positions = np.array(obs_positions)
        hx_list = np.vstack(hx_list)
        alphas = np.array(alphas)
        alphas = np.squeeze(alphas)

        sum_stage_cost = float(np.sum(stage_cost_eval))
        sum_stage_cost_alt = float(np.sum(stage_cost_alt))
        print(f"Stage Cost: {sum_stage_cost}")

        figstates=plt.figure()
        plt.plot(
            states_eval[:, 0], states_eval[:, 1],
            "o-"
        )

        # Plot the obstacle
        for (cx, cy), r in zip(self.agent.mpc.nn.obst.positions, self.agent.mpc.nn.obst.radii):
                    circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                    plt.gca().add_patch(circle)
        plt.gca().add_patch(circle)
        plt.xlim([-CONSTRAINTS_X[0], 0])
        plt.ylim([-CONSTRAINTS_X[1], 0])

        # Set labels and title
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("Trajectories")
        plt.legend()
        plt.axis("equal")
        plt.grid()
        self.viz.save_figures([(figstates,
        f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
        experiment_folder, "Evaluation")

        figactions=plt.figure()
        plt.plot(actions_eval[:, 0], "o-", label="Action 1")
        plt.plot(actions_eval[:, 1], "o-", label="Action 2")
        plt.xlabel("Iteration $k$")
        plt.ylabel("Action")
        plt.title("Actions")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        self.viz.save_figures([(figactions,
        f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
        experiment_folder,"Evaluation")


        figstagecost=plt.figure()
        plt.plot(stage_cost_eval, "o-")
        plt.xlabel("Iteration $k$")
        plt.ylabel("Cost")
        plt.title("Stage Cost")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        self.viz.save_figures([(figstagecost,
        f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
        experiment_folder, "Evaluation")

        figsvelocity=plt.figure()
        plt.plot(states_eval[:, 2], "o-", label="Velocity x")
        plt.plot(states_eval[:, 3], "o-", label="Velocity y")
        plt.xlabel("Iteration $k$")
        plt.ylabel("Velocity Value")
        plt.title("Velocity Plot")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        self.viz.save_figures([(figsvelocity,
        f"velocity_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
        experiment_folder, "Evaluation")

        for i in range(hx_list.shape[1]):
            fig_hi = plt.figure()
            plt.plot(hx_list[:,i], "o", label=rf"$h_{{{i+1}}}(x_k)$")
            plt.xlabel(r"Iteration $k$")
            plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
            plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
            plt.grid()
            self.viz.save_figures([(fig_hi,
                        f"hx_obstacle_{self.eval_count}_SC_{sum_stage_cost_alt}.svg")],
                        experiment_folder, "Evaluation")

        # Alphas from NN
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
        self.viz.save_figures([(fig_alpha,
                    f"alpha_{self.eval_count}_SC_{sum_stage_cost_alt}.svg")],
                    experiment_folder, "Evaluation")

        # T_pred = plans_eval.shape[0]
        out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost_alt}.gif")
        self.agent.make_system_obstacle_animation(
        states_eval,
        obs_positions,
        self.agent.mpc.nn.obst.radii,
        CONSTRAINTS_X[0],
        out_gif,
        )

        self.agent.update_learning_rate(sum_stage_cost, params)

        self.eval_count += 1

        # self.stage_cost_valid.append(np.sum(stage_cost_validation))
        self.stage_cost_valid.append(sum_stage_cost)

        return sum_stage_cost
