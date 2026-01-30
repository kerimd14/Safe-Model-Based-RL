import os
import copy
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import casadi as cs

from config import SEED, CONSTRAINTS_U, CONSTRAINTS_X
from mpc.mpc import MPC
from obstacles.obstacles_motion import ObstacleMotion


@dataclass
class WarmstartCache:
    x_prev: cs.DM = cs.DM()
    lam_x_prev: cs.DM = cs.DM()
    lam_g_prev: cs.DM = cs.DM()

    def reset(self):
        self.x_prev = cs.DM()
        self.lam_x_prev = cs.DM()
        self.lam_g_prev = cs.DM()


@dataclass
class RolloutResult:
    states: np.ndarray               # (T+1, ns)
    actions: np.ndarray              # (T, na)
    stage_cost: np.ndarray           # (T,)
    hx: np.ndarray                   # (T+1, m)
    alphas: np.ndarray               # (T, m) or (T,) depending on your net
    obs_positions: np.ndarray        # (T+1, m, 2)
    plans: np.ndarray                # (T, horizon+1, 2)
    slacks_eval: np.ndarray          # (T, m, horizon)
    lam_g_hist: np.ndarray           # (T, n_constraints)  (whatever your solver returns)
    g_resid_lst: np.ndarray          # (T, n_cbf_constraints) or similar


class PolicyRunner:
    """
    Run rollouts (before/after training) using deterministic MPC policy.

    This replaces:
      - run_simulation()
      - MPC_func()
      - flat_input_fn()
      - RNN_warmstart()
    in your old Functions.py style.
    """

    def __init__(
        self,
        env_cls,
        layers_list,
        horizon,
        positions,
        radii,
        modes,
        mode_params,
        slack_penalty_MPC_L1,
        slack_penalty_MPC_L2,
        seed: int = SEED,
    ):
        self.seed = seed

        # Environment
        self.env = env_cls()

        # MPC + solver
        self.mpc = MPC(
            layers_list=layers_list,
            horizon=horizon,
            positions=positions,
            radii=radii,
            slack_penalty_MPC_L1=slack_penalty_MPC_L1,
            slack_penalty_MPC_L2=slack_penalty_MPC_L2,
            mode_params=copy.deepcopy(mode_params),
            modes=modes,
        )
        self.solver_inst = self.mpc.MPC_solver()

        # Obstacles motion model
        self.obst_motion = ObstacleMotion(positions, modes, copy.deepcopy(mode_params))

        # helper functions
        self.h_func_list = self.mpc.rnn.obst.make_h_functions()
        self.get_hidden_func = self.mpc.rnn.make_rnn_step()      # one-step hidden update
        self.flat_input_fn = self.mpc.make_flat_input_fn()       # already fixed in your MPC

        # caches
        self.cache = WarmstartCache()

        # sizes
        self.ns = self.mpc.ns
        self.na = self.mpc.na
        self.m = self.mpc.rnn.obst.obstacle_num

    def _unflatten_slack(self, S_raw, m, N):
        """
        Turn S_flat (m*N,) or (m*N,1) from CasADi into (m, N).
        CasADi reshape is column-major (Fortran order): columns are horizon j.
        """
        S_raw = np.array(cs.DM(S_raw).full())
        if S_raw.shape == (m, N):
            return S_raw
        if S_raw.shape == (N, m):
            return S_raw.T
        flat = S_raw.reshape(-1)
        if flat.size == m * N:
            return flat.reshape(m, N, order="F")
        raise ValueError(f"Unexpected slack shape {S_raw.shape}, cannot make (m={m}, N={N}).")

    def warmstart_hidden(self, params, warmup_steps: int = 50):
        """
        Equivalent to your old RNN_warmstart(), but without rebuilding env/mpc every time.
        """
        self.cache.reset()
        state, _ = self.env.reset(seed=self.seed, options={})
        self.obst_motion.reset()

        hidden_in = [
            cs.DM.zeros(self.mpc.rnn.layers_list[i + 1], 1)
            for i in range(len(self.mpc.rnn.layers_list) - 2)
        ]

        for _ in range(warmup_steps):
            xpred_list, ypred_list = self.obst_motion.predict_states(self.mpc.horizon)

            _u, _V, hidden_in, _alpha, _plan_xy, _Xmat, _S = self.step_mpc(
                params=params,
                state=state,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
                hidden_in=hidden_in,
            )

            _ = self.obst_motion.step()

        return hidden_in

    def step_mpc(self, params, state, xpred_list, ypred_list, hidden_in):
        """
        One deterministic MPC solve, then update hidden state.

        Returns:
          u_opt, V_cost, hidden_t1, alpha_list, plan_xy, X_matrix, S_flat
        """
        # bounds
        X_lower_bound = -np.tile(CONSTRAINTS_X, self.mpc.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, self.mpc.horizon)
        U_lower_bound = -CONSTRAINTS_U * np.ones(self.na * self.mpc.horizon)
        U_upper_bound = CONSTRAINTS_U * np.ones(self.na * self.mpc.horizon)

        # constraint bounds
        state_const_lbg = np.zeros(self.ns * self.mpc.horizon)
        state_const_ubg = np.zeros(self.ns * self.mpc.horizon)
        cbf_const_lbg = -np.inf * np.ones(self.m * self.mpc.horizon)
        cbf_const_ubg = np.zeros(self.m * self.mpc.horizon)

        # decision vector bounds [x0; X(1..H); U(0..H-1); S]
        lbx = np.concatenate([
            np.asarray(state).flatten(),
            X_lower_bound,
            U_lower_bound,
            np.zeros(self.m * self.mpc.horizon),
        ])
        ubx = np.concatenate([
            np.asarray(state).flatten(),
            X_upper_bound,
            U_upper_bound,
            np.inf * np.ones(self.m * self.mpc.horizon),
        ])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        # flatten parameters for solver p
        A_flat = cs.reshape(params["A"], -1, 1)
        B_flat = cs.reshape(params["B"], -1, 1)
        P_diag = cs.diag(params["P"])
        Q_diag = cs.diag(params["Q"])
        R_diag = cs.diag(params["R"])

        sol = self.solver_inst(
            p=cs.vertcat(
                A_flat, B_flat, params["b"], params["V0"],
                P_diag, Q_diag, R_diag,
                params["rnn_params"],
                cs.DM(xpred_list), cs.DM(ypred_list),
                *hidden_in
            ),
            x0=self.cache.x_prev,
            lam_x0=self.cache.lam_x_prev,
            lam_g0=self.cache.lam_g_prev,
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg,
        )

        # warmstart caches
        self.cache.x_prev = sol["x"]
        self.cache.lam_x_prev = sol["lam_x"]
        self.cache.lam_g_prev = sol["lam_g"]

        # extract u0
        u_opt = sol["x"][self.ns * (self.mpc.horizon + 1): self.ns * (self.mpc.horizon + 1) + self.na]

        # extract X and U matrices
        Xmat = cs.reshape(sol["x"][:self.ns * (self.mpc.horizon + 1)], self.ns, self.mpc.horizon + 1)
        Umat = cs.reshape(
            sol["x"][self.ns * (self.mpc.horizon + 1): self.ns * (self.mpc.horizon + 1) + self.na * self.mpc.horizon],
            self.na,
            self.mpc.horizon
        )

        # flat input for hidden update
        flat_input = self.flat_input_fn(Xmat, cs.DM(xpred_list), cs.DM(ypred_list), Umat)
        x_t0 = flat_input[: self.mpc.rnn.layers_list[0]]
        x_t0 = self.mpc.rnn.normalization_z(x_t0)

        params_rnn_list = self.mpc.rnn.unpack_flat_parameters(params["rnn_params"])
        *hidden_t1, alpha_list = self.get_hidden_func(*hidden_in, x_t0, *params_rnn_list)

        # slack
        S_flat = sol["x"][self.na * self.mpc.horizon + self.ns * (self.mpc.horizon + 1):]

        # plan xy
        plan_xy = np.array(Xmat[:2, :]).T  # (horizon+1,2)

        return u_opt, sol["f"], hidden_t1, alpha_list, plan_xy, Xmat, S_flat

    def rollout(self, params, episode_duration: int, early_stop: bool = True):
        """
        Run a full episode and return RolloutResult.
        """
        self.cache.reset()
        hidden_in = self.warmstart_hidden(params)

        state, _ = self.env.reset(seed=self.seed, options={})
        self.obst_motion.reset()

        states = [np.array(state).copy()]
        actions = []
        stage_cost = []
        plans = []
        slacks_eval = []
        lam_g_hist = []
        g_resid_lst = []

        obs_positions = [self.obst_motion.current_positions()]

        xpred_list, ypred_list = self.obst_motion.predict_states(self.mpc.horizon)

        # hx list (include state0)
        hx0 = [
            float(hf(cs.DM(state),
                     xpred_list[0:self.m],
                     ypred_list[0:self.m]))
            for hf in self.h_func_list
        ]
        hx = [np.array(hx0)]

        alphas = []

        for _k in range(episode_duration):
            u, V, hidden_in, alpha, plan_xy, Xmat, S_flat = self.step_mpc(
                params=params,
                state=state,
                xpred_list=xpred_list,
                ypred_list=ypred_list,
                hidden_in=hidden_in,
            )

            u = cs.fmin(cs.fmax(cs.DM(u), -CONSTRAINTS_U), CONSTRAINTS_U)

            # store plan + slack
            plans.append(plan_xy)
            slacks_eval.append(self._unflatten_slack(S_flat, self.m, self.mpc.horizon))

            # store alpha
            alphas.append(np.array(alpha).reshape(-1))

            # constraints residuals (CBF part)
            g_resid = np.array(self.cache.lam_g_prev).flatten()
            lam_g_hist.append(g_resid)

            # step environment
            state, _, done, _, _ = self.env.step(u)
            states.append(np.array(state).copy())
            actions.append(np.array(u).reshape(-1))

            # update obstacle motion
            _ = self.obst_motion.step()
            obs_positions.append(self.obst_motion.current_positions())
            xpred_list, ypred_list = self.obst_motion.predict_states(self.mpc.horizon)

            # compute hx for this new state
            hx_k = [
                float(hf(cs.DM(state),
                         xpred_list[0:self.m],
                         ypred_list[0:self.m]))
                for hf in self.h_func_list
            ]
            hx.append(np.array(hx_k))

            # stage cost (use your old style)
            stage_cost.append(float(V))  # if you want EXACT stage cost, pass in your stage cost func instead

            if early_stop and (abs(state[0]) < 1e-3) and (abs(state[1]) < 1e-3):
                break

        # finalize arrays
        states = np.asarray(states)
        actions = np.asarray(actions)
        hx = np.vstack(hx)
        obs_positions = np.asarray(obs_positions)
        plans = np.asarray(plans)
        slacks_eval = np.asarray(slacks_eval)
        alphas = np.asarray(alphas)
        lam_g_hist = np.asarray(lam_g_hist)

        # g_resid_lst placeholder: if you want it, store solver g residuals directly (sol["g"])
        g_resid_lst = np.zeros((actions.shape[0], 1))

        stage_cost = np.asarray(stage_cost).reshape(-1)

        return RolloutResult(
            states=states,
            actions=actions,
            stage_cost=stage_cost,
            hx=hx,
            alphas=alphas,
            obs_positions=obs_positions,
            plans=plans,
            slacks_eval=slacks_eval,
            lam_g_hist=lam_g_hist,
            g_resid_lst=g_resid_lst,
        )
