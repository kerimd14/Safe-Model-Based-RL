import os
from collections import deque
from typing import Optional, Any

import numpy as np
import casadi as cs

from config import CONSTRAINTS_U, CONSTRAINTS_X
from rl.lrscheduler import update_learning_rate


class Trainer:
    """
    Trainer orchestrates: rollout -> TD/grad collection -> periodic param updates -> plots -> evaluation.
    """

    def __init__(self, agent: Any, evaluator: Optional[Any] = None, viz: Optional[Any] = None, lr_scheduler: Optional[Any] = None):
        """
        agent:        RLAgent (core MPC/RL capabilities)
        evaluator:    Evaluator (evaluation episodes + evaluation plots/gifs)
        viz:          viz module (viz/viz.py) OR None
        lr_scheduler: LearningRateScheduler dataclass (holds alpha + patience state)
        """
        self.agent = agent
        self.evaluator = evaluator
        self.viz = viz
        self.lr_scheduler = lr_scheduler

        self.eval_count = 1
        self.stage_cost_valid = []

    def rl_trainingloop(
        self,
        episode_duration: int,
        num_episodes: int,
        replay_buffer: int,
        episode_updatefreq: int,
        experiment_folder: str,
    ):
        # folders early (viz saves into learning_process/evaluation)
        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "learning_process"), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "evaluation"), exist_ok=True)

        # init params
        params = self.agent.params_init

        # keep alpha inside scheduler as source of truth
        if self.lr_scheduler is not None and self.lr_scheduler.best_params is None:
            self.lr_scheduler.best_params = params.copy()

        # for plotting
        params_history_P = [self.agent.params_init["P"]]

        x, _ = self.agent.env.reset(seed=self.agent.seed, options={})
        self.agent.obst_motion.reset()

        # logging buffers
        stage_cost_history = []
        sum_stage_cost_history = []
        TD_history = []
        TD_temp = []
        TD_episode = []
        B_update_history = []
        grad_temp = []

        obs_positions = [self.agent.obst_motion.current_positions()]
        phi_list = []

        # replay buffer for gradient-like updates
        B_update_buffer = deque(maxlen=replay_buffer)

        # for plotting snapshots
        states = [x]
        actions = []

        xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)

        hx = [
            float(
                hf(
                    cs.DM(x),
                    xpred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                    ypred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                )
            )
            for hf in self.agent.h_func_list
        ]
        hx_list = [hx]

        alphas = []

        k = 0
        error_happened = False

        total_steps = episode_duration * num_episodes

        # NOTE: (starts at i=1)
        for i in range(1, total_steps):

            # exploration noise
            noise = self.agent.noise_scalingfactor * self.agent.noise_scale_by_distance(x[0], x[1])
            rand = noise * self.agent.np_random.normal(loc=0, scale=self.agent.noise_variance, size=(2, 1))

            # noisy V-MPC action (policy provider) 
            u, alpha, _ = self.agent.v_mpc_rand(
                params=params,
                x=x,
                rand=rand,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
            )
            u = cs.fmin(cs.fmax(cs.DM(u), -CONSTRAINTS_U), CONSTRAINTS_U)

            alphas.append(alpha)
            actions.append(u)

            statsvrand = self.agent.solver_inst_random.stats()
            if statsvrand["success"] is False:
                print("v_mpc_RANDOM NOT SUCCEEDED")
                error_happened = True

            # Q-MPC for TD target
            solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _, _, _ = self.agent.q_mpc(
                params=params,
                action=u,
                x=x,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
            )

            statsq = self.agent.solver_inst.stats()
            if statsq["success"] is False:
                print("q_mpc NOT SUCCEEDED")
                error_happened = True

            # slack vector
            S = solution[self.agent.na * (self.agent.horizon) + self.agent.ns * (self.agent.horizon + 1) :]

            # stage cost (RL)
            stage_cost = self.agent.stage_cost(action=u, state=x, S=self.agent.S_VMPC_rand, hx=hx)

            # environment step
            x, _, done, _, _ = self.agent.env.step(u)

            # store for plotting
            states.append(x)

            hx = [
                float(
                    hf(
                        cs.DM(x),
                        xpred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                        ypred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                    )
                )
                for hf in self.agent.h_func_list
            ]
            hx_list.append(hx)

            # deterministic V-MPC for TD
            _, Vcost, _, _, _, _ = self.agent.v_mpc(
                params=params,
                x=x,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
            )

            statsv = self.agent.solver_inst.stats()
            if statsv["success"] is False:
                print("v_mpc NOT SUCCEEDED")
                error_happened = True

            # TD update
            TD = stage_cost + self.agent.gamma * Vcost - Qcost

            # extract U and X for phi/grad
            U = solution[
                self.agent.ns * (self.agent.horizon + 1) : self.agent.na * (self.agent.horizon)
                + self.agent.ns * (self.agent.horizon + 1)
            ]
            X = solution[: self.agent.ns * (self.agent.horizon + 1)]

            # phi diagnostics
            params_nn = self.agent.mpc.nn.unpack_flat_parameters(params["nn_params"])
            phi = self.agent.phi_func(
                X,
                U,
                cs.DM(xpred_list),
                cs.DM(ypred_list),
                *params_nn,
            )
            phi_list.append(np.array(phi).reshape(self.agent.horizon, self.agent.mpc.m))

            # numeric jacobian of qlagrange
            qlagrange_numeric_jacob = self.agent.qlagrange_fn_jacob(
                params["A"],
                params["B"],
                params["b"],
                params["Q"],
                params["R"],
                params["P"],
                lam_lbx,
                lam_ubx,
                lagrange_mult_g,
                X,
                U,
                S,
                params["nn_params"],
                xpred_list,
                ypred_list,
            )

            # first order update
            B_update = TD * qlagrange_numeric_jacob
            grad_temp.append(qlagrange_numeric_jacob)
            B_update_buffer.append(B_update)

            stage_cost_history.append(stage_cost)

            if not error_happened:
                TD_episode.append(TD)
                TD_temp.append(TD)
            else:
                TD_temp.append(cs.DM(np.nan))
                error_happened = False

            # obstacle step
            _ = self.agent.obst_motion.step()
            obs_positions.append(self.agent.obst_motion.current_positions())
            xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)

            # end of episode block 
            
            if k == episode_duration - 1:
                current_episode = i // episode_duration 

                # parameter updates every episode_updatefreq episodes
                if (i - 1) % (episode_duration * episode_updatefreq) == 0:
                    B_update_avg = np.mean(B_update_buffer, axis=0)
                    B_update_history.append(B_update_avg)

                    # learning rate source of truth
                    if self.lr_scheduler is not None:
                        self.agent.alpha = self.lr_scheduler.alpha

                    params = self.agent.parameter_updates(params=params, B_update_avg=B_update_avg)

                    # keep P history
                    params_history_P.append(params["P"])

                    # noise decay
                    self.agent.noise_scalingfactor *= (1 - self.agent.decay_rate)
                    print(f"noise scaling: {self.agent.noise_scalingfactor}")

                # episode summaries
                sum_stage_cost_history.append(np.sum(stage_cost_history))
                TD_history.append(np.mean(TD_episode) if len(TD_episode) else np.nan)

                stage_cost_history = []
                TD_episode = []

                # periodic training plots/snapshots
                if (current_episode % 50) == 0 and self.viz is not None:
                    states_np = np.array(states)
                    actions_np = np.asarray(actions)
                    TD_temp_np = np.asarray(TD_temp)
                    obs_positions_np = np.array(obs_positions)
                    hx_list_np = np.vstack(hx_list)
                    alphas_np = np.array(alphas)



                    self.viz.training_snapshot(
                        states=states_np,
                        actions=actions_np,
                        td_values=TD_temp_np,
                        obs_positions=obs_positions_np,
                        hx_list=hx_list_np,
                        alphas=alphas_np,
                        positions=self.agent.mpc.nn.obst.positions,
                        radii=self.agent.mpc.nn.obst.radii,
                        constraints_x=CONSTRAINTS_X,
                        experiment_folder=experiment_folder,
                        step=i,
                    )

                # periodic evaluation 
                if (current_episode % 50) == 0 and self.evaluator is not None:
                    eval_cost = self.evaluator.evaluation_step(
                        params=params,
                        experiment_folder=experiment_folder,
                        episode_duration=episode_duration,
                    )
                    self.stage_cost_valid.append(eval_cost)

                    if self.lr_scheduler is not None:
                        params = update_learning_rate(
                            current_stage_cost=eval_cost,
                            params=params,
                            scheduler=self.lr_scheduler,
                        )
                        self.agent.alpha = self.lr_scheduler.alpha

                    self.eval_count += 1

                # reset episode
                x, _ = self.agent.env.reset(seed=self.agent.seed, options={})
                self.agent.obst_motion.reset()
                k = 0

                self.agent._reset_warmstart_caches()

                states = [x]
                TD_temp = []
                actions = []
                grad_temp = []
                obs_positions = [self.agent.obst_motion.current_positions()]

                xpred_list, ypred_list = self.agent.obst_motion.predict_states(self.agent.horizon)

                hx = [
                    float(
                        hf(
                            cs.DM(x),
                            xpred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                            ypred_list[0 : self.agent.mpc.nn.obst.obstacle_num],
                        )
                    )
                    for hf in self.agent.h_func_list
                ]
                hx_list = [hx]
                alphas = []

                print("reset")

            k += 1

            if i % 1000 == 0:
                print(f"{i}/{total_steps}")

        # end of training: final plots + saving
        params_history_P = np.asarray(params_history_P)
        TD_history = np.asarray(TD_history)
        sum_stage_cost_history = np.asarray(sum_stage_cost_history)

        if self.viz is not None:
            self.viz.plot_B_update(B_update_history, experiment_folder)
            self.viz.plot_training_curves(
                params_history_P=params_history_P,
                sum_stage_cost_history=sum_stage_cost_history,
                TD_history=TD_history,
                experiment_folder=experiment_folder,
                spectral_radii_hist=np.array(self.agent.spectral_radii_hist),
                stage_cost_valid=self.stage_cost_valid,
            )

        if self.lr_scheduler is not None and self.lr_scheduler.best_params is not None:
            return self.lr_scheduler.best_params
        return params
    
 