import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque


class env(gym.Env):

    def __init__(self, sampling_time:float):
        super().__init__()
        self.na = 2
        self.ns = 4
        self.umax = 1
        self.umin = -1

        self.observation_space = Box(-5, 5, (self.ns,), np.float64)
        self.action_space = Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = sampling_time

        self.A = np.array([
            [1, 0, self.dt, 0], 
            [0, 1, 0, self.dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])


    def reset(self, seed, options):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-5, -5, 0, 0])
        # assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
        return self.x, {}

    def step(
        self, action):
        x = self.x
        u = np.asarray(action).reshape(self.na)
        self.B = np.array([
            [0.5 * self.dt**2, 0], 
            [0, 0.5 * self.dt**2], 
            [self.dt, 0], 
            [0, self.dt]
        ])
        # assert self.action_space.contains(u), f"invalid action {u}"

        x_new = self.A @ self.x + self.B @ u
        # assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self.x = x_new
        return x_new, np.nan, False, False, {}


class MPC:
    # constant of MPC class
    ns = 4 # num of states
    na = 2 # num of inputs
    horizon = 1 # MPC horizon

    def __init__(self, dt):
        """
        Initialize the MPC class with parameters.
        """
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = MPC.horizon
        # self.x0 = cs.MX.sym("x0")

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

        self.Q = np.diag([10, 10, 10, 10])
        
        #dynamics
        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.b_sym = cs.MX.sym("b", self.ns)

        #MPC params
        # self.P_sym = cs.MX.sym("P", self.ns, self.ns)
        self.P_diag = cs.MX.sym("P_diag", self.ns, 1)
        self.P_sym = cs.diag(self.P_diag)


        self.Q_sym = cs.MX.sym("Q", self.ns, self.ns)
        self.R_sym = cs.MX.sym("R", self.na, self.na)
        self.V_sym = cs.MX.sym("V0")

        #weight on the slack variables
        self.weight_cbf = cs.DM([1e4])


        # decision variables
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        self.S_sym = cs.MX.sym("S", 1, self.horizon)

        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)
        self.alpha_sym = cs.MX.sym("alpha")

        # defining stuff for CBF
        x_new  = self.A @ self.x_sym + self.B @ self.u_sym 
        self.dynamics_f = cs.Function('f', [self.x_sym, self.u_sym], [x_new], ['x','u'], ['ode'])

        h = (self.x_sym[0]-(self.pos[0]))**2 + (self.x_sym[1]-(self.pos[1]))**2 - self.r**2 
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])



    def dcbf(self,
        h, x, u, dynamics, alphas,
    ) -> cs.Function:
        r"""Discrete-time Control Barrier Function (DCBF) for the given constraint ``h`` and
        system with dynamics ``dynamics``. This method constructs a DCBF for the constraint
        :math:`h(x) \geq 0` using the given system's dynamics :math:`x_{+} = f(x, u)`. Here,
        :math:`x_{+}` is the next state after applying control input :math:`u`, and
        :math:`f` is the dynamics function ``dynamics``.

        The method can also compute a High-Order (HO) DCBF by passing more than one class
        :math:`\mathcal{K}` functions ``alphas``.

        As per [1]_, the HO-DCBF :math:`\phi_m` of degree :math:`m` is recursively found as

        .. math::
            \phi_m(x_"V0") = \phi_{m-1}(x_{k+1}) - \phi_{m-1}(x_k) + \alpha_m(\phi_{m-1}(x_k))

        and should be imposed as the constraint :math:`\phi_m(x_k) \geq 0`.

        Parameters
        ----------
        h : callable
            The constraint function for which to build the DCBF. It must be of the signature
            :math:`x \rightarrow h(x)`.
        x : casadi SX or MX
            The state vector variable :math:`x`.
        u : casadi SX or MX
            The control input vector variable :math:`u`.
        dynamics : callable
            The dynamics function :math:`f` with signature :math:`x,u \rightarrow f(x, u)`.
        alphas : iterable of callables
            An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
            the HO-DCBF. The length of the iterable determines the degree of the HO-DCBF.

        Returns
        -------
        casadi SX or MX
            Returns the HO-DCBF function :math:`\phi_m` as a symbolic variable that is
            function of the provided ``x`` and ``u``.
        """
        x_next = dynamics(x, u)
        phi = h(x)
        for alpha in alphas:
            phi_next = cs.substitute(phi, x, x_next)
            phi = phi_next - phi + alpha(phi)
        return phi
    

    def state_const(self):

        """""
        used to construct state constraints 
        """

        state_const_list = []

        for k in range(self.horizon):

            state_const_list.append( self.X_sym[:,k+1] - ( self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym ) )

        self.state_const_list = cs.vertcat( *state_const_list )

        return 
    
    
    def cbf_func(self):

        # i have to normalize it here before it goes into the NN
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f, [lambda y: self.alpha_sym*y])

        return cs.Function('cbff', [self.x_sym, self.u_sym, self.alpha_sym], [cbf], ['x','u', 'alpha_sym'], ['cbff'])

    def cbf_const(self):

        """""
        used to construct cbf constraints 
        """

        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):

            cbf_const_list.append(cbf_func(self.X_sym[:,k], self.U_sym[:,k], self.alpha_sym) + self.S_sym[:,k]) 

        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        return 
    
    def objective_method(self):

        """""
        stage cost calculation
        """
        # why doesnt work? --> idk made it in line
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k] + 
            self.weight_cbf * self.S_sym[:,k])
            for k in range(self.horizon)
        )
        #slack penalty
        terminal_cost = cs.bilin((self.P_sym), self.X_sym[:, -1])


        self.objective = self.V_sym + terminal_cost + stage_cost

        return
    
    def MPC_solver(self):
        """""
        solves the MPC according to V-value function setup
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1) 
        S_flat = cs.reshape(self.S_sym, -1, 1)  
        U_flat = cs.reshape(self.U_sym, -1, 1)  

        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        P_sym_flat = cs.reshape(self.P_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, self.alpha_sym),
            "f": self.objective, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": False,
            "calc_lam_x": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "calc_lam_p": True,
            "fatrop": {"max_iter": 500, "print_level": 0},
        }

        MPC_solver = cs.nlpsol("solver", "fatrop", nlp, opts)

        return MPC_solver
    

    def MPC_solver_rand(self):
        """""
        solves the MPC according to V-value function setup
        """
        self.state_const()
        self.objective_method()
        self.cbf_const()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)    
        U_flat = cs.reshape(self.U_sym, -1, 1)


        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        P_sym_flat = cs.reshape(self.P_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        rand_noise = cs.MX.sym("rand_noise", 2)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, self.alpha_sym, rand_noise),
            "f": self.objective + rand_noise.T @ self.U_sym[:,0], 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": False,
            "calc_lam_x": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "calc_lam_p": False,
            "fatrop": {"max_iter": 500, "print_level": 0},
        }

        MPC_solver = cs.nlpsol("solver", "fatrop", nlp, opts)

        print("hey")

        return MPC_solver
    
    def generate_symbolic_mpcq_lagrange(self):
        """
        constructs MPC action state value function solver
        """
        self.state_const()
        self.objective_method()

        X_flat = cs.reshape(self.X_sym, -1, 1)  # Flatten 
        U_flat = cs.reshape(self.U_sym, -1, 1)
        S_flat = cs.reshape(self.S_sym, -1, 1)   

        opt_solution = cs.vertcat(X_flat, U_flat, S_flat)

      
        # X_con + U_con + S_con 
        lagrange_mult_x_lb_sym = cs.MX.sym("lagrange_mult_x_lb_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + 1 * (self.horizon))
        lagrange_mult_x_ub_sym = cs.MX.sym("lagrange_mult_x_ub_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + 1 * (self.horizon))
        lagrange_mult_g_sym = cs.MX.sym("lagrange_mult_g_sym", 1*self.ns*(self.horizon) + self.horizon)

        # construct 
        X_lower_bound = -5 * np.array([1, 1, 1, 1])
        X_upper_bound = 5 * np.array([1, 1, 1, 1])

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        # X_con + U_con + S_con + Sx_con
        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), np.zeros(self.horizon)) 
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), np.inf*np.ones(self.horizon))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, self.cbf_const_list) # opposite signs

 
        theta_vector = cs.vertcat(self.P_diag, self.alpha_sym)

        self.theta = theta_vector

        qlagrange = self.objective + lagrange1 + lagrange2 + lagrange3

        #computing derivative of lagrangian for A
        _, qlagrange_sens = cs.hessian(qlagrange, theta_vector)

        #transpose it becase cs.hessian gives it differently than cs.jacobian
        qlagrange_sens = qlagrange_sens

        qlagrange_fn = cs.Function(
            "qlagrange_fn",
            [
                self.A_sym, self.B_sym, self.b_sym, self.Q_sym, self.R_sym,
                self.P_sym, lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.alpha_sym
            ],
            [qlagrange_sens],
            [
                'A_sym', 'B_sym', 'b_sym', 'Q_sym', 'R_sym', 'P_sym', 'lagrange_mult_x_lb_sym', 
                'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'alpha'
            ],
            ['qlagrange_sens']
        )

        return qlagrange_fn
    
    def qp_solver_fn(self):
            #implementing optimiazation for one time step

            Hessian_sym = cs.MX.sym("Hessian", self.theta.shape[0]*self.theta.shape[0])

            # p_fradient_sym for SGD is -alpha*gradient
            p_gradient_sym = cs.MX.sym("gradient", self.theta.shape[0])

            delta_theta = cs.MX.sym("delta_theta", self.theta.shape[0])
            theta = cs.MX.sym("delta_theta", self.theta.shape[0])

            lambda_reg = 1e-6
 
            qp = {
                "x": cs.vertcat(delta_theta),
                "p": cs.vertcat(theta, Hessian_sym, p_gradient_sym),
                "f": 0.5*delta_theta.T @ Hessian_sym.reshape((self.theta.shape[0], self.theta.shape[0])) @ delta_theta +  p_gradient_sym.T @ (delta_theta) + lambda_reg/2 * delta_theta.T @ delta_theta, 
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

            return cs.qpsol('solver','osqp', qp, opts)

####################### RL #######################


class RLclass:

        def __init__(self, params_innit, seed, alpha, sampling_time, gamma, decay_rate, 
                     noise_scalingfactor, noise_variance, patience_threshold, lr_decay_factor):
            self.seed = seed

            # enviroment class
            self.env = env(sampling_time=sampling_time)

            # mpc class
            self.mpc = MPC(sampling_time)
            # self.x0, _ = self.env.reset(seed=seed, options={})
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit

            #learning rate
            self.alpha = alpha
            
            # bounds
            #state bounded between 5 and -5
            self.X_lower_bound = -5 * np.array([1, 1, 1, 1])#-1e6 * 5 * np.ones(mpc.ns * (mpc.horizon))
            self.X_upper_bound = 5 * np.array([1, 1, 1, 1])#1e6 * 5 * np.ones(mpc.ns  * (mpc.horizon))

            #state bound between 0 and 0, to make sure Ax +Bu = 0
            self.state_const_lbg = np.zeros(1*self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1*self.ns  * (self.horizon))

            #cbf constraint bounded between -inf and zero --> it is inverted
            # to make sure the constraints stay the same for all of them
            self.cbf_const_lbg = -np.inf * np.ones(1*(self.horizon))
            self.cbf_const_ubg = np.zeros(1*(self.horizon))
            # gamma
            self.gamma = gamma

            # to generate randomness
            self.np_random = np.random.default_rng(seed=self.seed)
            self.noise_scalingfactor = noise_scalingfactor
            self.noise_variance = noise_variance

            #solver instance
            self.solver_inst = self.mpc.MPC_solver()

            #get parameter sensitivites
            self.qlagrange_fn_jacob = self.mpc.generate_symbolic_mpcq_lagrange()

            #decay_rate 
            self.decay_rate = decay_rate

            #randomness
            self.solver_inst_random =self.mpc.MPC_solver_rand()

            #qp solver
            self.qp_solver = self.mpc.qp_solver_fn()

            # learning update initialization
            self.best_stage_cost = np.inf
            self.patience_threshold = patience_threshold
            self.lr_decay_factor = lr_decay_factor

            self.pos = self.mpc.pos
            self.r = self.mpc.r

            #ADAM
            self.exp_avg = np.zeros(5)
            self.exp_avg_sq = np.zeros(5)
            self.adam_iter = 1
        
        
        def save_figures(self, figures, experiment_folder, save_in_subfolder=False):
            
            save_choice = True  
            
            if save_choice:
                
                if save_in_subfolder == "Learning":
                    target_folder = os.path.join(experiment_folder, "learning_process")
                elif save_in_subfolder == "Evaluation":
                    target_folder = os.path.join(experiment_folder, "evaluation")
                else:
                    target_folder = experiment_folder

                os.makedirs(target_folder, exist_ok=True)
                
                for fig, filename in figures:
                    file_path = os.path.join(target_folder, filename)
                    fig.savefig(file_path)
                    print(f"Figure saved as: {file_path}")
            else:
                print("Figure not saved")

        
        def ADAM(self, iteration, gradient, exp_avg, exp_avg_sq,
            learning_rate, beta1, beta2, eps = 1e-8): 
            """Computes the update's change according to Adam algorithm."""
            gradient = np.asarray(gradient).flatten()

            exp_avg = beta1 * exp_avg + (1 - beta1) * gradient
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * np.square(gradient)

            bias_correction1 = 1 - beta1**iteration                
            bias_correction2 = 1 - beta2**iteration

            step_size = learning_rate / bias_correction1
            bias_correction2_sqrt = np.sqrt(bias_correction2)

            denom = np.sqrt(exp_avg_sq) / (bias_correction2_sqrt + eps)
            
            dtheta = -step_size * (exp_avg / denom)
            return dtheta, exp_avg, exp_avg_sq
        

        def update_learning_rate(self, current_stage_cost):
            """
            Update the learning rate based on the current stage cost metric.
            """

            if current_stage_cost < self.best_stage_cost:
                self.best_stage_cost = current_stage_cost
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience_threshold:
                old_alpha = self.alpha
                self.alpha *= self.lr_decay_factor  # decay 
                print(f"Learning rate decreased from {old_alpha} to {self.alpha} due to no stage cost improvement.")
                self.current_patience = 0  # reset 


        # def noise_scale_by_distance(x, y, max_radius=3):
        #     # i might remove this because it doesnt allow for exploration of the last states which is important
        #     dist = np.sqrt(x**2 + y**2)
        #     if dist >= max_radius:
        #         return 1
        #     else:
        #         return (dist / max_radius)

        def cholesky_added_multiple_identity(self,
            A, beta: float = 1e-3, maxiter: int = 1000
        ):
            r"""Lower Cholesky factorization with added multiple of the identity matrix to ensure
            positive-definitiveness from Algorithm 3.3 in :cite:`nocedal_numerical_2006`.

            The basic idea is to add a multiple of the identity matrix to the original matrix
            unitl the factorization is successful, i.e., find :math:`\tau \ge 0` such that
            :math:`L L^\top = A + \tau I` is successful.

            Parameters
            ----------
            A : array of double
                The 2D matrix to compute the cholesky factorization of.
            beta : float, optional
                Initial tolerance of the algorithm, by default ``1e-3``.
            maxiter : int, optional
                Maximum iterations of the algorithm, by default ``1000``.

            Returns
            -------
            array of double
                The lower cholesky factorization of the modified ``A`` (with the addition of
                identity matrices to ensure that it is positive-definite).

            Raises
            ------
            ValueError
                If the factorization is unsuccessful for the maximum number of iterations.
            """
            a_min = np.diag(A).min()
            tau = 0 if a_min > 0 else -a_min + beta
            identity = np.eye(A.shape[0])
            for _ in range(maxiter):
                try:
                    return np.linalg.cholesky(A + tau * identity)
                except np.linalg.LinAlgError:
                    tau = max(1.05 * tau, beta)
            print("Warning: Cholesky decomposition failed after maximum iterations; returning zero matrix.")
            return np.zeros((A.shape[0], A.shape[0]))

        def V_MPC(self, params, x):
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["alpha"]),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract solution from solver 
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]


            return u_opt, solution["f"]
        
        def V_MPC_rand(self, params, x, rand):
            """
            same function as V_MPC but now with randomness
            """
            
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.horizon)])
            

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)


            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["alpha"], rand),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]

            return u_opt

        def Q_MPC(self, params, action, x):

            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon-1))
            U_upper_bound = np.ones(self.na * (self.horizon-1))

            #here since its Q value we also give action
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, np.asarray(action).flatten(), U_lower_bound,  np.zeros(self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound, np.asarray(action).flatten(), U_upper_bound, np.inf*np.ones(self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["alpha"]),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            lagrange_mult_g = solution["lam_g"]
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]

            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p
            
        def stage_cost(self, action, x):
            """Computes the stage cost :math:`L(s,a)`.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])

            state = x

            hx = (state[0]-(self.pos[0]))**2 + (state[1]-(self.pos[1]))**2 - self.r**2 

            # if 4e3*np.abs(min(hx, 0)) > 0:
            #     print(f"hx stage cost: {4e3*np.abs(min(hx, 0))}")
            #     print(f"hx: {hx}")


            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + 1e5*np.abs(min(hx, 0))
            )
        
        def evaluation_step(self, params, experiment_folder, episode_duration):
                
                state, _ = self.env.reset(seed=self.seed, options={})

                states_eval = []
                actions_eval = []
                stage_cost_eval = []

                for i in range(episode_duration):
                    action, _ = self.V_MPC(params=params, x=state)

                    action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                    state, _, done, _, _ = self.env.step(action)
                    states_eval.append(state)
                    actions_eval.append(action)

                    stage_cost_eval.append(self.stage_cost(action, state))

                states_eval = np.array(states_eval)
                actions_eval = np.array(actions_eval)
                stage_cost_eval = np.array(stage_cost_eval)
                stage_cost_eval = stage_cost_eval.reshape(-1) 

                figstates=plt.figure()
                plt.plot(
                    states_eval[:, 0], states_eval[:, 1],
                    "o-"
                )

                # Plot the obstacle
                circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
                plt.gca().add_patch(circle)
                plt.xlim([-5, 0])
                plt.ylim([-5, 0])

                # Set labels and title
                plt.xlabel("$x$ (m)")
                plt.ylabel("$y$ (m)")
                plt.title("Trajectories")
                plt.legend()
                plt.axis("equal")
                plt.grid()

                figactions=plt.figure()
                plt.plot(actions_eval[:, 0], "o-", label="Action 1")
                plt.plot(actions_eval[:, 1], "o-", label="Action 2")
                plt.xlabel("Time (s)")
                plt.ylabel("Action")
                plt.title("Actions")
                plt.legend()
                plt.grid()
                plt.tight_layout()


                figstagecost=plt.figure()
                plt.plot(stage_cost_eval, "o-")
                plt.xlabel("Time (s)")
                plt.ylabel("Cost")
                plt.title("Stage Cost")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                
                figsvelocity=plt.figure()
                plt.plot(states_eval[:, 2], "o-", label="Velocity x")
                plt.plot(states_eval[:, 3], "o-", label="Velocity y")    
                plt.xlabel("Time (s)")
                plt.ylabel("Velocity Value")
                plt.title("Velocity Plot")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                sum_stage_cost = sum(stage_cost_eval)
                
                figs = [
                                (figstates, f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost:.2f}.png"),
                                (figactions, f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost:.2f}.png"),
                                (figstagecost, f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost:.2f}.png"),
                                (figsvelocity, f"velocity_MPCeval_{self.eval_count}_SC_{sum_stage_cost:.2f}.png")
                ]

                self.save_figures(figs, experiment_folder, "Evaluation")
                plt.close("all")

                self.update_learning_rate(np.sum(stage_cost_eval))

                self.eval_count += 1

                return 
        
        def parameter_updates(self, params, B_update_avg):

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            theta_vector_num = cs.vertcat(P_diag, params["alpha"])


            identity = np.eye(theta_vector_num.shape[0])

            alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-1), self.alpha*1e-1)
            
            # self.alpha is the inital ADAM thing

            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 

            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, dtheta)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")


            # lbx = cs.vertcat(-np.inf*np.ones(4), -0.01*np.ones(theta_vector_num.shape[0]-4))
            # ubx = cs.vertcat(np.inf*np.ones(4), 0.01*np.ones(theta_vector_num.shape[0]-4)

            # constrained update qp update
            solution = self.qp_solver(
                    p = cs.vertcat(theta_vector_num, identity.flatten(), dtheta),
                    lbg = cs.vertcat(np.zeros(theta_vector_num.shape[0]-1), np.array([1e-6])),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0]-1), np.array([1])),
                    # ubx = ubx,
                    # lbx = lbx
                )
            stats = self.qp_solver.stats()

            if stats["success"] == False:
                print("QP NOT SUCCEEDED")
                theta_vector_num = theta_vector_num
            else:
                theta_vector_num = theta_vector_num + solution["x"]

            print(f"theta_vector_num: {theta_vector_num}")

            P_diag_shape = self.ns*1
            #constructing the diagonal posdef P matrix 
            P_posdef = cs.diag(theta_vector_num[:P_diag_shape])

            params["P"] = P_posdef
            params["alpha"] = theta_vector_num[-1]

            

            return params
        # def parameter_updates(self, params, B_update_avg):

        #     """
        #     function responsible for carryin out parameter updates after each episode
        #     """
        #     P_diag = cs.diag(params["P"])

        #     #vector of parameters which are differenitated with respect to
        #     theta_vector_num = cs.vertcat(P_diag, params["alpha"])


        #     identity = np.eye(theta_vector_num.shape[0])

        #     # print(f"before updates : {theta_vector_num}")

        #     # alpha_vec is resposible for the updates
        #     # alpha_vec = cs.vertcat(self.alpha*np.ones(2), self.alpha, self.alpha, self.alpha*1e-2)
        #     alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-1), self.alpha*1e-3)

        #     # uncostrained update to compare to the qp update
        #     y = np.linalg.solve(identity, B_update_avg)
        #     theta_vector_num_toprint = theta_vector_num - (alpha_vec * y)#self.alpha * y
        #     print(f"theta_vector_num no qp: {theta_vector_num_toprint}")


        #     # lbx = cs.vertcat(-np.inf*np.ones(4), -0.01*np.ones(theta_vector_num.shape[0]-4))
        #     # ubx = cs.vertcat(np.inf*np.ones(4), 0.01*np.ones(theta_vector_num.shape[0]-4))

            
        #     # constrained update qp update
        #     solution = self.qp_solver(
        #             p = cs.vertcat(theta_vector_num, identity.flatten(), alpha_vec*B_update_avg),
        #             lbg = cs.vertcat(np.zeros(theta_vector_num.shape[0]-1), np.array([1e-6])),
        #             ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0]-1), np.array([1])),
        #             # ubx = ubx,
        #             # lbx = lbx
        #         )
        #     stats = self.qp_solver.stats()

        #     if stats["success"] == False:
        #         print("QP NOT SUCCEEDED")
        #         theta_vector_num = theta_vector_num
        #     else:
        #         theta_vector_num = theta_vector_num + solution["x"]

        #     print(f"theta_vector_num: {theta_vector_num}")

        #     P_diag_shape = self.ns*1
        #     #constructing the diagonal posdef P matrix 
        #     P_posdef = cs.diag(theta_vector_num[:P_diag_shape])

        #     params["P"] = P_posdef
        #     params["alpha"] = theta_vector_num[-1]

            

        #     return params
        
        

        def rl_trainingloop(self, episode_duration, num_episodes, replay_buffer, episode_updatefreq, experiment_folder):
    
            #to store for plotting
            params_history_P = [self.params_innit["P"]]

            #for the for loop
            params = self.params_innit
            
            x, _ = self.env.reset(seed=self.seed, options={})

            stage_cost_history = []
            sum_stage_cost_history = []
            TD_history = []
            TD_episode = []
            TD_temp = []

            # deque update buffer for the first ord update
            B_update_buffer = deque(maxlen=replay_buffer)
            
            states = [(x)]
            
            actions = []

            #intialize
            k = 0
            self.error_happened = False
            self.eval_count = 1
            # try:
            for i in range(1,episode_duration*num_episodes):
                
                rand = self.noise_scalingfactor*self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u = self.V_MPC_rand(params=params, x=x, rand = rand)
                u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)

                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    self.error_happened = True

                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _ = self.Q_MPC(params=params, action=u, x=x)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    self.error_happened = True

                stage_cost = self.stage_cost(action=u,x=x)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)

                _, Vcost = self.V_MPC(params=params, x=x)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED")
                    self.error_happened = True


                #time penalty
                TD = (stage_cost) + self.gamma*Vcost - Qcost
    

                U = solution[self.ns * (self.horizon+1):self.na * (self.horizon) + self.ns * (self.horizon+1)] 
                X = solution[:self.ns * (self.horizon+1)] 
                S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1): :self.na * (self.horizon) + self.ns * (self.horizon+1) +self.horizon]
            
                #parameter update: A
                qlagrange_numeric_jacob=  self.qlagrange_fn_jacob(
                    A_sym=params["A"],
                    B_sym=params["B"],
                    b_sym=params["b"],
                    Q_sym = params["Q"],
                    R_sym = params["R"],
                    P_sym = params["P"],
                    lagrange_mult_x_lb_sym=lam_lbx,
                    lagrange_mult_x_ub_sym=lam_ubx,
                    lagrange_mult_g_sym=lagrange_mult_g,
                    X=X, U=U, S=S, alpha=params["alpha"]
                )['qlagrange_sens']

                # second order update
                B_update = -TD*qlagrange_numeric_jacob
                B_update_buffer.append(B_update)
                
            
                stage_cost_history.append(stage_cost)
                if self.error_happened == False:
                    TD_episode.append(TD)
                    TD_temp.append(TD)
                else:
                    TD_temp.append(np.nan)
                    self.error_happened = False

                # if i == 790000:
                #     self.alpha = self.alpha*1e-2
                
                # # reset state to learn from somehwhere else:
                if (k == episode_duration):
                  
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:

                        self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
                        print (f"updatedddddd")
                        B_update_avg = np.mean(B_update_buffer, 0)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.sum(TD_episode))

                    stage_cost_history = []
                    TD_episode = []

                    x, _ = self.env.reset(seed=self.seed, options={})
                    k=0

                    print("reset")


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 20) == 0:
                        states = np.array(states)
                        actions = np.asarray(actions)
                        TD_temp = np.array(TD_temp)

                        figstate=plt.figure()
                        plt.plot(
                            states[:, 0], states[:, 1],
                            "o-"
                        )

                        # Plot the obstacle
                        circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
                        plt.xlim([-5, 0])
                        plt.ylim([-5, 0])

                        # Set labels and title
                        plt.xlabel("$x$ (m)")
                        plt.ylabel("$y$ (m)")
                        plt.title("Trajectories of states while policy is trained$")
                        plt.legend()
                        plt.axis("equal")
                        plt.grid()


                        figvelocity=plt.figure()
                        plt.plot(states[:, 2], "o-", label="Velocity x")
                        plt.plot(states[:, 3], "o-", label="Velocity y")    
                        plt.xlabel("Time (s)")
                        plt.ylabel("Velocity Value")
                        plt.title("Velocity Plot")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        # 

                        positions = states[1:, :2]  # assuming first 2 entries are x and y
                        circle_center = np.array([-2, -2.25])
                        circle_radius = 1.7
                        distances = np.sqrt((positions[:, 0] - circle_center[0])**2 +
                                            (positions[:, 1] - circle_center[1])**2)
                        mask_near = distances <= circle_radius
                        mask_far = distances > circle_radius
                        indices = np.arange(len(TD_temp))

                        figtdtemp = plt.figure(figsize=(10, 5))
                        plt.scatter(indices[mask_near], TD_temp[mask_near], c='red', label='Near Circle')
                        plt.scatter(indices[mask_far], TD_temp[mask_far], c='blue', label='Far from Circle')
                        plt.yscale('log')
                        plt.title("TD Over Training (Log Scale) - Colored by Proximity")
                        plt.xlabel("Training Step")
                        plt.ylabel("TD")
                        plt.legend()
                        plt.grid(True)

                        figactions=plt.figure()
                        plt.plot(actions[:, 0], "o-", label="Action 1")
                        plt.plot(actions[:, 1], "o-", label="Action 2")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Action")
                        plt.title("Actions")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()

                        figures_training = [
                            (figstate, f"position_plotat_{i}"),
                            (figvelocity, f"velocity_plotat_{i}"),
                            (figtdtemp, f"TD_plotat_{i}"),
                            (figactions, f"action_plotat_{i}")
                            ]
                        self.save_figures(figures_training, experiment_folder, "Learning")

                    states = [(x)]
                    TD_temp = []
                    actions = []
                # k counter    
                k+=1
                
                #counter
                if i % 1000 == 0:
                    print(f"{i}/{episode_duration*num_episodes}")  

            #show trajectories
            
            # plt.close()


            params_history_P = np.asarray(params_history_P)

            TD_history = np.asarray(TD_history)
            sum_stage_cost_history = np.asarray(sum_stage_cost_history)
        

            figP = plt.figure(figsize=(10, 5))
            for i in range(params_history_P.shape[1]):
                # for j in range(params_history_P.shape[2]):
                    plt.plot(params_history_P[:, i, i], label=f"P[{i},{i}]")
            plt.title("Parameter: P")
            plt.xlabel("Training Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            # plt.close()
            #

            figstagecost = plt.figure(figsize=(10, 5))
            plt.plot(sum_stage_cost_history, 'o', label="Stage Cost")
            plt.yscale('log') 
            plt.title("Stage Cost Over Training (Log Scale)")
            plt.xlabel("Training Step")
            plt.ylabel("Stage Cost")
            plt.legend()
            plt.grid(True)
            #

            figtd = plt.figure(figsize=(10, 5))
            plt.plot(TD_history, 'o', label="TD")
            plt.yscale('log')
            plt.title("TD Over Training (Log Scale)")
            plt.xlabel("Training Step")
            plt.ylabel("TD")
            plt.legend()
            plt.grid(True)
            

            figures_to_save = [
                (figP, "P"),
                (figstagecost, "stagecost"),
                (figtd, "TD")

            ]

            self.save_figures(figures_to_save, experiment_folder)
            return params
        
        