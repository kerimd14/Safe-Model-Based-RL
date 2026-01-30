import numpy as np
import casadi as cs

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X
from rnn.rnn import RNN


class MPC:
    # constant of MPC class
    ns = NUM_STATES  # num of states
    na = NUM_INPUTS  # num of inputs

    def __init__(
        self,
        layers_list,
        horizon,
        positions,
        radii,
        slack_penalty_MPC_L1,
        slack_penalty_MPC_L2,
        mode_params,
        modes,
    ):
        """
        Initialize the MPC class with parameters.

        Initialize the MPC problem:
         - build discrete-time system matrices A, B
         - set up CasADi symbols for Q, R, P, V0
         - instantiate the RNN for the CBF
         - prepare CasADi decision variables X, U, S
         - build dynamics function f(x,u) and RNN forward pass
        """
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = horizon

        dt = SAMPLING_TIME

        # discrete‐time dynamics matrices
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])

        # fixed state‐cost weight
        self.Q = np.diag([10, 10, 10, 10])

        # symbolic parameters for MPC
        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.b_sym = cs.MX.sym("b", self.ns)

        self.P_diag = cs.MX.sym("P_diag", self.ns, 1)
        self.P_sym = cs.diag(self.P_diag)

        self.V_sym = cs.MX.sym("V0")

        self.Q_diag = cs.MX.sym("Q_diag", self.ns, 1)
        self.R_diag = cs.MX.sym("R_diag", self.na, 1)
        self.Q_sym = cs.diag(self.Q_diag)
        self.R_sym = cs.diag(self.R_diag)

        # instantiate the RNN
        self.rnn = RNN(layers_list, positions, radii, horizon, mode_params, modes)
        self.m = self.rnn.obst.obstacle_num  # number of obstacles

        # flattened predictions for each time step
        self.xpred_hor = cs.MX.sym("xpred_hor", self.m * (self.horizon + 1))
        self.ypred_hor = cs.MX.sym("ypred_hor", self.m * (self.horizon + 1))

        # weight on slack variables in CBF constraints
        self.slack_penalty_MPC_L1 = slack_penalty_MPC_L1
        self.slack_penalty_MPC_L2 = slack_penalty_MPC_L2

        # decision variables:
        #   X_sym: states over horizon+1
        #   U_sym: control inputs over horizon
        #   S_sym: slack for CBF constraints (m x horizon)
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon + 1)
        self.U_sym = cs.MX.sym("U", self.na, self.horizon)
        self.S_sym = cs.MX.sym("S", self.m, self.horizon)

        # states for one time step and inputs for one time step
        self.x_sym = cs.MX.sym("x", self.ns)
        self.u_sym = cs.MX.sym("u", self.na)

        # CasADi function for one‐step dynamics: f(x,u) = A x + B u
        x_new = self.A @ self.x_sym + self.B @ self.u_sym
        self.dynamics_f = cs.Function("f", [self.x_sym, self.u_sym], [x_new], ["x", "u"], ["ode"])

        # CBF h‐functions for each obstacle
        self.h_funcs = self.rnn.obst.make_h_functions()  # length‐m list[casadi.Function]

        # build flat‐input function for RNN
        self.flat_input_fn = self.make_flat_input_fn()
        flat_input = self.flat_input_fn(self.X_sym, self.xpred_hor, self.ypred_hor)

        # hidden‐state symbols and parameter symbols for the RNN
        self.hid_syms = self.rnn.hidden_sym_list           # [h0_layer0, h0_layer1, ...]
        param_syms = self.rnn.get_flat_parameters_list()   # [Wih0, bih0, Whh0, ..., WihL, bihL]

        # RNN forward‐pass over the horizon to get alphas
        self.rnn_fwd_func = self.rnn.forward_rnn()
        _, Y_stack = self.rnn_fwd_func(
            flat_input,
            *self.hid_syms,
            *param_syms
        )

        # Y_stack containts list of alphas: [alpha_0, alpha_1, ..., alpha_{m*horizon}]
        self.alpha_list = Y_stack

        # theta vector for qp updates 
        self.theta = cs.vertcat(self.rnn.get_flat_parameters())

    def state_const(self):
        """
        Build linear dynamics constraints:
          X_{k+1} - [A_sym X_k + B_sym U_k + b_sym] == 0, for k=0...horizon-1
        """
        state_const_list = []

        for k in range(self.horizon):
            state_const_list.append(
                self.X_sym[:, k + 1] - (self.A_sym @ self.X_sym[:, k] + self.B_sym @ self.U_sym[:, k] + self.b_sym)
            )

        self.state_const_list = cs.vertcat(*state_const_list)
        print(f"self.state_const_list shape: {self.state_const_list.shape}")
        return

    def make_flat_input_fn(self):
        """
        Returns a CasADi Function that maps:
           (X, xpred_hor, ypred_hor, U)
           --> flattened sequence: [x_t; h(x_t); obs_x_t; obs_y_t; u_t] for t=0...horizon-1

           i dont do make one flat input sized for the RNN but i make a couple of them because i need to feed it into the rnn_fwd
           i need to do this since i need to use RNN to calculate a number of these params
        """
        X = cs.MX.sym("X", self.ns, self.horizon + 1)

        # IMPORTANT: use local symbols as inputs (not self.xpred_hor/self.ypred_hor),
        # otherwise the function closes over self.* and becomes fragile.
        xpred_hor = cs.MX.sym("xpred_hor", self.m * (self.horizon + 1))
        ypred_hor = cs.MX.sym("ypred_hor", self.m * (self.horizon + 1))

        inter = []
        for t in range(self.horizon):
            x_t = X[:, t]

            obs_x = xpred_hor[t * self.m:(t + 1) * self.m]
            obs_y = ypred_hor[t * self.m:(t + 1) * self.m]

            cbf_t = [
                h_i(x_t, obs_x, obs_y)
                for h_i in self.h_funcs
            ]

            obs_x_list = cs.vertsplit(obs_x)  # [MX(1x1), ..., MX(1x1)]
            obs_y_list = cs.vertsplit(obs_y)

            inter.append(x_t)                 # nsx1
            inter.extend(cbf_t)               # m scalars
            inter.extend(obs_x_list)
            inter.extend(obs_y_list)     # na scalars

        flat_in = cs.vertcat(*inter)
        return cs.Function(
            "flat_input",
            [X, xpred_hor, ypred_hor],
            [flat_in],
            ["X", "xpred_hor", "ypred_hor"],
            ["flat_in"],
        )

    def cbf_const(self):
        """
        Build the slack-augmented CBF constraints phi_i + S >= 0
        for each i=1...m and each time k=0...h-1:
        phi_i = h_i(x_{k+1}) - h_i(x_k) + alpha_k,i * h_i(x_k)
        """
        cons = []
        m = len(self.h_funcs)

        for k in range(self.horizon):
            xk = self.X_sym[:, k]
            uk = self.U_sym[:, k]

            phi_k_list = []
            for i, h_i in enumerate(self.h_funcs):
                alpha_ki = self.alpha_list[k * m + i]

                h_x = h_i(
                    xk,
                    self.xpred_hor[k * self.m:(k + 1) * self.m],
                    self.ypred_hor[k * self.m:(k + 1) * self.m],
                )
                x_next = self.dynamics_f(xk, uk)
                h_xnext = h_i(
                    x_next,
                    self.xpred_hor[(k + 1) * self.m:(k + 2) * self.m],
                    self.ypred_hor[(k + 1) * self.m:(k + 2) * self.m],
                )

                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)  # mx1
            cons.append(phi_k + self.S_sym[:, k])

        self.cbf_const_list = cs.vertcat(*cons)

    def cbf_const_noslack(self):
        """
        Same as cbf_const but without slack variables.
        """
        cons = []
        m = len(self.h_funcs)

        for k in range(self.horizon):
            xk = self.X_sym[:, k]
            uk = self.U_sym[:, k]

            phi_k_list = []
            for i, h_i in enumerate(self.h_funcs):
                alpha_ki = self.alpha_list[k * m + i]

                h_x = h_i(
                    xk,
                    self.xpred_hor[k * self.m:(k + 1) * self.m],
                    self.ypred_hor[k * self.m:(k + 1) * self.m],
                )
                x_next = self.dynamics_f(xk, uk)
                h_xnext = h_i(
                    x_next,
                    self.xpred_hor[(k + 1) * self.m:(k + 2) * self.m],
                    self.ypred_hor[(k + 1) * self.m:(k + 2) * self.m],
                )

                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)
            cons.append(phi_k)

        self.cbf_const_list_noslack = cs.vertcat(*cons)

    def objective_method(self):
        """""
        Builds MPC stage cost and terminal cost
        stage cost: sum_k [ x.T @ Q @ x + u.T @ R @ u ] + slack penalties
        terminal cost: x.T @ P @ x
        """
        quad_cost = sum(
            self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k]
            + self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k]
            for k in range(self.horizon)
        )

        # CBF slack (or violation) over objects and time
        cbf_cost_L1 = self.slack_penalty_MPC_L1 * sum(
            self.S_sym[m, k]
            for m in range(self.m)
            for k in range(self.horizon)
        )

        cbf_cost_L2 = cs.DM(0.5) * self.slack_penalty_MPC_L2 * sum(
            self.S_sym[m, k] ** 2
            for m in range(self.m)
            for k in range(self.horizon)
        )

        stage_cost = quad_cost + cbf_cost_L1 + cbf_cost_L2
        print(f"Stage cost current: {stage_cost}")

        terminal_cost = cs.bilin(self.P_sym, self.X_sym[:, -1])
        self.objective = self.V_sym + terminal_cost + stage_cost
        return

    def objective_method_noslack(self):
        """""
        stage cost calculation
        """
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k]
             + self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k])
            for k in range(self.horizon)
        )

        terminal_cost = cs.bilin(self.P_sym, self.X_sym[:, -1])
        self.objective_noslack = self.V_sym + terminal_cost + stage_cost
        return

    def MPC_solver_noslack(self):
        """""
        Create and return a CasADi NLP solver for MPC without slack.
        MPC built according to V-value function setup
        """
        self.state_const()
        self.objective_method_noslack()
        self.cbf_const_noslack()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)

        A_sym_flat = cs.reshape(self.A_sym, -1, 1)
        B_sym_flat = cs.reshape(self.B_sym, -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat),
            "p": cs.vertcat(
                A_sym_flat, B_sym_flat, self.b_sym, self.V_sym,
                self.P_diag, self.Q_diag, self.R_diag,
                self.rnn.get_flat_parameters(),
                self.xpred_hor, self.ypred_hor,
                *self.hid_syms
            ),
            "f": self.objective_noslack,
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list_noslack),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "ipopt": {"max_iter": 2000, "print_level": 0, "warm_start_init_point": "yes"},
        }

        return cs.nlpsol("solver", "ipopt", nlp, opts)

    def MPC_solver(self):
        """""
        solves the MPC according to V-value function setup
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)

        A_sym_flat = cs.reshape(self.A_sym, -1, 1)
        B_sym_flat = cs.reshape(self.B_sym, -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(
                A_sym_flat, B_sym_flat, self.b_sym, self.V_sym,
                self.P_diag, self.Q_diag, self.R_diag,
                self.rnn.get_flat_parameters(),
                self.xpred_hor, self.ypred_hor,
                *self.hid_syms
            ),
            "f": self.objective,
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "ipopt": {"max_iter": 2000, "print_level": 0, "warm_start_init_point": "yes"},
        }

        return cs.nlpsol("solver", "ipopt", nlp, opts)

    def MPC_solver_rand(self):
        """""
        solves the MPC according to V-value function setup (exploration noise on first action)
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)

        A_sym_flat = cs.reshape(self.A_sym, -1, 1)
        B_sym_flat = cs.reshape(self.B_sym, -1, 1)

        rand_noise = cs.MX.sym("rand_noise", self.na)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(
                A_sym_flat, B_sym_flat, self.b_sym, self.V_sym,
                self.P_diag, self.Q_diag, self.R_diag,
                self.rnn.get_flat_parameters(),
                rand_noise,
                self.xpred_hor, self.ypred_hor,
                *self.hid_syms
            ),
            "f": self.objective + rand_noise.T @ self.U_sym[:, 0],
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "ipopt": {"max_iter": 2000, "print_level": 0, "warm_start_init_point": "yes"},
        }

        return cs.nlpsol("solver", "ipopt", nlp, opts)

    def generate_symbolic_mpcq_lagrange(self):
        """
        Construct a CasADi Function that computes the gradient of the MPC Lagrangian
        (w.r.t. theta = rnn parameters).
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)

        opt_solution = cs.vertcat(X_flat, U_flat, S_flat)

        # X_con + U_con + S_con  (same shape as decision vector)
        lagrange_mult_x_lb_sym = cs.MX.sym(
            "lagrange_mult_x_lb_sym",
            self.ns * (self.horizon + 1) + self.na * self.horizon + self.m * self.horizon
        )
        lagrange_mult_x_ub_sym = cs.MX.sym(
            "lagrange_mult_x_ub_sym",
            self.ns * (self.horizon + 1) + self.na * self.horizon + self.m * self.horizon
        )
        lagrange_mult_g_sym = cs.MX.sym(
            "lagrange_mult_g_sym",
            self.ns * self.horizon + self.m * self.horizon
        )

        # bounds 
        X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)

        # IMPORTANT:
        # fixing U[:,0] via lbx/ubx, and leaving the remaining horizon-1 free here
        U_lower_bound = -np.ones(self.na * (self.horizon - 1))
        U_upper_bound = np.ones(self.na * (self.horizon - 1))

        lbx = cs.vertcat(
            self.X_sym[:, 0],
            cs.DM(X_lower_bound),
            self.U_sym[:, 0],
            cs.DM(U_lower_bound),
            np.zeros(self.m * self.horizon),
        )
        ubx = cs.vertcat(
            self.X_sym[:, 0],
            cs.DM(X_upper_bound),
            self.U_sym[:, 0],
            cs.DM(U_upper_bound),
            np.inf * np.ones(self.m * self.horizon),
        )

        # construct lower bound here
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx)
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, -self.cbf_const_list)

        theta_vector = self.theta
        qlagrange = self.objective + lagrange1 + lagrange2 + lagrange3

        print("BEFORE THE JACOBIAN")
        qlagrange_sens = cs.gradient(qlagrange, theta_vector)
        print(f"Shape is ({qlagrange_sens.size1()}, {qlagrange_sens.size2()})")
        print("AFTER THE JACOBIAN")

        qlagrange_fn = cs.Function(
            "qlagrange_fn",
            [
                self.A_sym, self.B_sym, self.b_sym, self.Q_sym, self.R_sym,
                self.P_sym,
                lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, lagrange_mult_g_sym,
                X_flat, U_flat, S_flat,
                self.theta,
                self.xpred_hor, self.ypred_hor,
                *self.hid_syms,
            ],
            [qlagrange_sens]
        )
        return qlagrange_fn

    def qp_solver_fn(self):
        """
        Construct and return a small QP solver (OSQP) for constrained parameter updates.
        """
        n = int(self.theta.shape[0])  # dimension of theta
        Hessian = cs.DM.eye(n)
        p_gradient_sym = cs.MX.sym("gradient", self.theta.shape[0])

        delta_theta = cs.MX.sym("delta_theta", self.theta.shape[0])
        theta = cs.MX.sym("theta", self.theta.shape[0])

        qp = {
            "x": cs.vertcat(delta_theta),
            "p": cs.vertcat(theta, p_gradient_sym),
            "f": 0.5 * delta_theta.T @ Hessian @ delta_theta + p_gradient_sym.T @ (delta_theta),
            "g": theta + delta_theta,
        }

        opts = {
            "error_on_fail": False,
            "print_time": False,
            "verbose": False,
            "max_io": False,
            "osqp": {
                "eps_abs": 1e-9,
                "eps_rel": 1e-9,
                "max_iter": 10000,
                "eps_prim_inf": 1e-9,
                "eps_dual_inf": 1e-9,
                "polish": True,
                "scaling": 100,
                "verbose": False,
            },
        }

        return cs.qpsol("solver", "osqp", qp, opts)

    def make_phi_fn(self):
        """
        Return a CasADi Function phi(X, U, xpred_hor, ypred_hor, h0, params)
        which returns a list of mx1 vectors, one per k.

        (no slack added)
        """
        m = len(self.h_funcs)

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)

        cons = []
        for k in range(self.horizon):
            xk = self.X_sym[:, k]
            uk = self.U_sym[:, k]

            phi_k_list = []
            for i, h_i in enumerate(self.h_funcs):
                alpha_ki = self.alpha_list[k * m + i]

                h_x = h_i(
                    xk,
                    self.xpred_hor[k * self.m:(k + 1) * self.m],
                    self.ypred_hor[k * self.m:(k + 1) * self.m],
                )
                x_next = self.dynamics_f(xk, uk)
                h_xnext = h_i(
                    x_next,
                    self.xpred_hor[(k + 1) * self.m:(k + 2) * self.m],
                    self.ypred_hor[(k + 1) * self.m:(k + 2) * self.m],
                )

                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)
            cons.append(phi_k)

        return cs.Function(
            "phi_fn",
            [X_flat, U_flat, self.xpred_hor, self.ypred_hor, *self.hid_syms, *self.rnn.get_flat_parameters_list()],
            [*cons]
        )
