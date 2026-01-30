import numpy as np
import casadi as cs
import copy
from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X
from nn.nn import NN


class MPC:
    # constant of MPC class
    ns = NUM_STATES # num of states
    na = NUM_INPUTS # num of inputs

    def __init__(self, layers_list, horizon, positions, radii, slack_penalty, mode_params, modes):
        """
        Initialize the MPC class with parameters.
        
        Initialize the MPC problem:
         - build discrete-time system matrices A, B
         - set up CasADi symbols for Q, R, P, V0
         - instantiate the nn for the CBF
         - prepare CasADi decision variables X, U, S
         - build dynamics function f(x,u) and nn forward pass
        
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

        self.Q_sym = cs.MX.sym("Q", self.ns, self.ns)
        self.R_sym = cs.MX.sym("R", self.na, self.na)
        self.V_sym = cs.MX.sym("V0")
        
        # instantiate the Neural Network
        self.nn = NN(layers_list, positions, radii, copy.deepcopy(mode_params), modes=modes)
        self.m = self.nn.obst.obstacle_num  # number of obstacles
        self.xpred_list = cs.MX.sym("xpred_list", self.m)  # velocity in x direction
        self.ypred_list = cs.MX.sym("ypred_list", self.m) # velocity in y direction
        self.xpred_hor = cs.MX.sym("xpred_hor", self.m * (self.horizon+1))
        self.ypred_hor = cs.MX.sym("ypred_hor", self.m * (self.horizon+1))
        self.h_funcs = self.nn.obst.make_h_functions()  # length‐m Python list of casadi.Function
        
        #weight on the slack variables
        self.weight_cbf = cs.DM([slack_penalty])


        # decision variables
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        
        # slack variable matrix with positions and horizon
        self.S_sym = cs.MX.sym("S", self.m, self.horizon)

        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)


        # defining stuff for CBF
        x_new  = self.A @ self.x_sym + self.B @ self.u_sym 
        self.dynamics_f = cs.Function('f', [self.x_sym, self.u_sym], [x_new], ['x','u'], ['ode'])

        self.alpha_nn = self.nn.get_alpha_nn()

    def state_const(self):

        """
        Build linear dynamics constraints:
          X_{k+1} - [A_sym X_k + B_sym U_k + b_sym] == 0, for k=0…horizon
        """

        state_const_list = []

        for k in range(self.horizon):

            state_const_list.append( self.X_sym[:,k+1] - ( self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym ) )

        self.state_const_list = cs.vertcat( *state_const_list )
        
        print(f"self.state_const_list shape: {self.state_const_list.shape}")

        return 
    

    def cbf_const(self):
        """
        Build the slack-augmented CBF constraints phi_i + S >= 0
        for each i=1...m and each time k=0...h-1:
        phi_i = h_i(x_{k+1}) - h_i(x_k) + alpha_k*_i(x_k)
        """
        cons = []
        

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            uk = self.U_sym[:, k]      # 2×1

            phi_k_list = []
            
            nn_in_list = [h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m]) for h_i in self.h_funcs ]
            alpha_list = self.alpha_nn(cs.vertcat(xk, *nn_in_list, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m]))
            for i, h_i in enumerate(self.h_funcs):
                

                alpha_ki = alpha_list[i]       

                h_x     = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])                          # h(x_k)
                x_next  = self.dynamics_f(xk, uk)          # f(x_k, u_k)
                h_xnext = h_i(x_next, self.xpred_hor[(k+1)*self.m:(k+2)*self.m], self.ypred_hor[(k+1)*self.m:(k+2)*self.m])                      # h(x_{k+1})

                # φ_i = h(x_{k+1}) − h(x_k) + α⋅h(x_k)
                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            # now add slack
            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k + self.S_sym[:, k])         # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list = cs.vertcat(*cons)
        
    # def cbf_const_noslack(self):
    #     """
    #     Same as cbf_const, but omit slack.  We simply stack every φ_k(x,u) over k=0..horizon-1.
    #     Result is an (m*horizon)×1 vector of pure CBF‐constraints.
    #     """
    #     cbf_fn = self.cbf_func()
    #     cons = []

    #     for k in range(self.horizon):
    #             xk    = self.X_sym[:, k]
    #             uk    = self.U_sym[:, k]
    #             phi_k = cbf_fn(xk, uk, self.nn.get_flat_parameters(), self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])  # m×1
    #             print(f"{self.xpred_hor[k*self.m].shape} and {self.ypred_hor[k*self.m].shape}")
    #             cons.append(phi_k)

    #     self.cbf_const_list_noslack = cs.vertcat(*cons)
    #     print(f"self.m shape :  {self.m}")
    #     print(f"cbf_const_list_noslack shape: {self.cbf_const_list_noslack.shape}")
    #     return
    
    def cbf_const_noslack(self):
        """
        Same as cbf_const but without slack variables.
        """
        cons = []
        m = len(self.h_funcs)

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            uk = self.U_sym[:, k]      # 2×1
            phi_k_list = []
            
            nn_in_list = [h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m]) for h_i in self.h_funcs ]
            alpha_list = self.alpha_nn(cs.vertcat(xk, *nn_in_list, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m]))
            for i, h_i in enumerate(self.h_funcs):
                
                
                alpha_ki = alpha_list[i]      # MX-scalar

                h_x     = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])                          # h(x_k)
                x_next  = self.dynamics_f(xk, uk)          # f(x_k, u_k)
                h_xnext = h_i(x_next, self.xpred_hor[(k+1)*self.m:(k+2)*self.m], self.ypred_hor[(k+1)*self.m:(k+2)*self.m])                       # h(x_{k+1})

                # φ_i = h(x_{k+1}) − h(x_k) + α⋅h(x_k)
                phi_i = h_xnext - h_x + alpha_ki * h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k)         # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list_noslack = cs.vertcat(*cons)
    
    def objective_method(self):

        """"
        Builds MPC stage cost and terminal cost
        stage cost: sum_{k,i} [ x.T @ Q @ x + u.T @ R @ u + weight_cbf * S_i]
        terminal cost: x.T @ P @ x
        """
        quad_cost = sum(
        self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k]
         + self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k]
        for k in range(self.horizon)
        )

        # CBF slack (or violation) over objects and time
        cbf_cost = (self.weight_cbf/(self.m + self.horizon) ) * sum(
        self.S_sym[m, k]
        for m in range(self.m)
        for k in range(self.horizon)
        ) 
        
        stage_cost = quad_cost + cbf_cost
        #slack penalty
        terminal_cost = cs.bilin((self.P_sym), self.X_sym[:, -1])


        self.objective = self.V_sym + terminal_cost + stage_cost

        return
    
    def objective_method_noslack(self):

        """""
        stage cost calculation
        """
        # why doesnt work? --> idk made it in line
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k]) 
            for k in range(self.horizon)
        )
        #slack penalty
        terminal_cost = cs.bilin((self.P_sym), self.X_sym[:, -1])


        self.objective_noslack = self.V_sym + terminal_cost + stage_cost

        return
    
    def MPC_solver_noslack(self):
        """"
        Create and return a CasADi NLP solver for MPC without slack.
        MPC built according to V-value function setup
        """
        self.state_const()
        self.objective_method_noslack()
        self.cbf_const_noslack()

        # Flatten matrices to put in as vector
        X_flat = cs.reshape(self.X_sym, -1, 1) 
        U_flat = cs.reshape(self.U_sym, -1, 1)  

        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, 
                            R_sym_flat, self.nn.get_flat_parameters(), self.xpred_hor, self.ypred_hor),
            "f": self.objective_noslack, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list_noslack),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 500, "print_level": 0, "warm_start_init_point": True},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

        return MPC_solver
    
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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, 
                            R_sym_flat, self.nn.get_flat_parameters(), self.xpred_hor, self.ypred_hor),
            "f": self.objective, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 500, "print_level": 0, "warm_start_init_point": True},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, 
                            self.nn.get_flat_parameters(), rand_noise, self.xpred_hor, self.ypred_hor),
            "f": self.objective + rand_noise.T @ self.U_sym[:,0], 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            # (Replace MX with SX expressions in problem formulation)
            # Automatically expand and simplify symbolic expressions before solving
            "expand": True,
            # 	print information about execution time
            "print_time": False,
            # 	Ensure that primal-dual solution is consistent with the bounds (aka you make sure bound 0 is 0 and not 1e-8)
            "bound_consistency":True,
            # Calculate Lagrange multipliers
            "calc_lam_x": True,
            "calc_lam_p": True,
            "calc_multipliers": True,
            # When errors occur during evaluation of f,g,...,stop the iterations
            "eval_errors_fatal": True,
            # Throw exceptions when function evaluation fails (default true).
            "error_on_fail": False,
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 500, "print_level": 0, "warm_start_init_point": True},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

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
        lagrange_mult_x_lb_sym = cs.MX.sym("lagrange_mult_x_lb_sym", self.ns * (self.horizon+1) + 
                                           self.na * (self.horizon) + self.nn.obst.obstacle_num * (self.horizon))
        lagrange_mult_x_ub_sym = cs.MX.sym("lagrange_mult_x_ub_sym", self.ns * (self.horizon+1) + 
                                           self.na * (self.horizon) + self.nn.obst.obstacle_num  * (self.horizon))
        lagrange_mult_g_sym = cs.MX.sym("lagrange_mult_g_sym", 1*self.ns*(self.horizon) + 
                                        self.nn.obst.obstacle_num*self.horizon)

        
        X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
        X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        # X_con + U_con + S_con + Sx_con
        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), 
                         np.zeros(self.nn.obst.obstacle_num *self.horizon)) 
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), 
                         np.inf*np.ones(self.nn.obst.obstacle_num *self.horizon))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, -self.cbf_const_list) # opposite signs

 
        # theta_vector = cs.vertcat(self.P_diag, self.nn.get_flat_parameters())
        theta_vector = cs.vertcat(self.nn.get_flat_parameters())
        self.theta = theta_vector

        qlagrange = self.objective + lagrange1 + lagrange2 + lagrange3

        #computing derivative of lagrangian for A
        qlagrange_sens = cs.gradient(qlagrange, theta_vector)

        #transpose it becase cs.hessian gives it differently than cs.jacobian
        qlagrange_sens = qlagrange_sens

        qlagrange_fn = cs.Function(
            "qlagrange_fn",
            [
                self.A_sym, self.B_sym, self.b_sym, self.Q_sym, self.R_sym,
                self.P_sym, lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.nn.get_flat_parameters(),
                self.xpred_hor, self.ypred_hor
            ],
            [qlagrange_sens],
            [
                'A_sym', 'B_sym', 'b_sym', 'Q_sym', 'R_sym','P_sym', 'lagrange_mult_x_lb_sym', 
                'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'inputs_NN',
                'xpred_hor', 'ypred_hor'
            ],
            ['qlagrange_sens']
        )

        return qlagrange_fn
    
    def qp_solver_fn(self):
        """
        Constructs QP solve for parameter updates
        """

        # Hessian_sym = cs.MX.sym("Hessian", self.theta.shape[0]*self.theta.shape[0]) # making this hessian take a lot 
        n = int(self.theta.shape[0])  # dimension of theta
        Hessian = cs.DM.eye(n)
        p_gradient_sym = cs.MX.sym("gradient", self.theta.shape[0])

        delta_theta = cs.MX.sym("delta_theta", self.theta.shape[0])
        theta = cs.MX.sym("delta_theta", self.theta.shape[0])

        lambda_reg = 1e-6

        qp = {
            "x": cs.vertcat(delta_theta),
            "p": cs.vertcat(theta, p_gradient_sym),
            "f": 0.5*delta_theta.T @ Hessian.reshape((self.theta.shape[0], self.theta.shape[0])) @ delta_theta +  p_gradient_sym.T @ (delta_theta) + lambda_reg/2 * delta_theta.T @ delta_theta, 
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