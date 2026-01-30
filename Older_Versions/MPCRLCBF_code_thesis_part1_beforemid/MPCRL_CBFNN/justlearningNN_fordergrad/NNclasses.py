import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque


class NN:

    def __init__(self, layers_size):
        """
        layers_size = list of layer sizes, including input and output sizes, for example: [5, 7, 7, 1]
        hidden_layers = number of hidden layers
        """
        self.layers_size = layers_size

        # list of weights and biases and activations
        self.weights = []  
        self.biases  = [] 
        self.activations = []


        self.build_network()
        self.np_random = np.random.default_rng(seed=69)

    def relu(self, x):
        #relu activation, used for all layers except the output layer
        return cs.fmax(x, 0)
    
    def tanh(self, x):
        #tanh activation, used for all layers except the output layer
        return cs.tanh(x)

    def leaky_relu(self, x, alpha=0.01):
        # For x >= 0, returns x; for x < 0, returns alpha*x.
        return cs.fmax(x, 0) + alpha * cs.fmin(x, 0)

    def sigmoid(self, x):
        #sigmoid activation, used for the last output layer
        return 1 / (1 + cs.exp(-x))
    
    def shifted_sigmoid(self, x, epsilon=1e-6):
        return epsilon + (1 - epsilon) * (1 / (1 + cs.exp(-x)))
    
    def build_network(self):
        """
        build the stuff needed for network
        """
        for i in range(len(self.layers_size) - 1):
            W = cs.MX.sym(f"W{i}", self.layers_size[i+1], self.layers_size[i])
            b = cs.MX.sym(f"b{i}", self.layers_size[i+1], 1)
            self.weights.append(W)
            self.biases.append(b)

            if i == len(self.layers_size) - 2:
                self.activations.append(self.shifted_sigmoid)
            else:
                self.activations.append(self.leaky_relu)

    def forward(self, x):
        """
        memeber function to perform the forward pass
        """
        # x = self.normalization_z(x)
        a = x
        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            a = self.activations[i](z)
        return a

    def create_forward_function(self):
        """
        making casadi function for the forward pass (in other words making the NN function)
        """
        # the input the class kappa function needs to take in
        x = cs.MX.sym('x', self.layers_size[0], 1)

        y = self.forward(x)

        inputs = [x] + self.weights + self.biases

        return cs.Function('NN', inputs, [y])
    
    def get_flat_parameters(self):
        weight_list = [cs.reshape(W, -1, 1) for W in self.weights]
        bias_list = [cs.reshape(b, -1, 1) for b in self.biases]
        return cs.vertcat(*(weight_list + bias_list))
    
    def get_alpha_nn(self):
        nn_fn = self.create_forward_function()
        return lambda x: nn_fn(x, *self.weights, *self.biases)
    
    # def normalization_z(self, x):
    #     """
    #     normalization of inputs for neural network
    #     """


    #     p_min, p_max = -5, 0
    #     mu_p = (p_min + p_max) / 2.0  # -2.5
    #     sigma_p = (p_max - p_min) / np.sqrt(12)  # std of uniform distribution on [0,5]
    #     x_norm_p = (x[0:2] - mu_p) / sigma_p

      
    #     # v_min, v_max = -5, 5
    #     mu_v =  0.00833056#(v_min + v_max) / 2.0  
    #     sigma_v = 0.10660607#(v_max - v_min) / np.sqrt(12)  # std of uniform distribution on [-5,5]
    #     x_norm_v = (x[2:4] - mu_v) / sigma_v  # std of uniform distribution
        

    #     # Soft saturation using tanh(z/a) --> source
    #     # x_sat = cs.tanh(x_norm)


    #     h_min = 0#-2.25
    #     h_max = 11.8125
    #     mu_h = (h_min + h_max) / 2.0
    #     sigma_h = (h_max - h_min) / np.sqrt(12)

    #     pos = cs.DM([-2, -2.25])
    #     r = cs.DM(1.5)

    #     h = (x[0] - (pos[0]))**2 + (x[1] - (pos[1]))**2 - r**2
    #     h_norm = (h - mu_h) / sigma_h

    #     x_norm = cs.vertcat(x_norm_p, x_norm_v)
    #     # h_sat = cs.tanh(h_norm)

    #     return x_norm, h_norm

    def normalization_z(self, x):
        """
        Normalizes based on opt trajct
        Mean states: [-0.0305078  -0.02024065  0.00833056  0.00833056]
        Std states: [0.33782308 0.27604534 0.09290382 0.10660607]
        Mean h: 6.7885445905109325
        Std h: 0.4382974988082045   
        """
        mu_states = cs.DM([-0.97504422, -0.64636289, 0.05090653, 0.05091322])# cs.DM([-1.57845433, -1.04727382,  0.43094213,  0.43102892])#cs.DM([-0.97504422, -0.64636289, 0.05090653, 0.05091322])
        sigma_states = cs.DM([1.39463336, 1.18101312, 0.0922252, 0.09315792])# cs.DM([1.8605361,  1.42324245, 0.51423965, 0.63705421])#cs.DM([1.39463336, 1.18101312, 0.0922252, 0.09315792])
        
        x_norm = (x - mu_states) / sigma_states
        pos = cs.DM([-2, -2.25])
        r = cs.DM(1.5)
        h = (x[0] - pos[0])**2 + (x[1] - pos[1])**2 - r**2

        # mu_h = 4.422387613414234 # 6.7885445905109325
        # sigma_h =   2.7606664837421535 # 0.4382974988082045
        h_min = 0#-2.25
        h_max = 11.8125
        mu_h = (h_min + h_max) / 2.0
        sigma_h = (h_max - h_min) / np.sqrt(12)
        h_norm = (h - mu_h) / sigma_h

        return x, h_norm

    
    def numerical_forward(self):
        """
        memeber function to perform the forward pass
        """
        # a = x
        # for i in range(len(self.weights)):
        #     z = self.weights[i] @ a + self.biases[i]
        #     a = self.activations[i](z)
        # return a
        x = cs.MX.sym('x', self.layers_size[0]-1, 1)


        x, h_norm = self.normalization_z(x)

        input = cs.vertcat(x, h_norm)

        y = self.forward(input)

       

        return cs.Function('NN', [x, self.get_flat_parameters()],[y])
    
    
    # def normalization_minmax(self, x):
    #     """
    #     normalization of inputs for neural network
    #     """
    #     pos = cs.DM([-2, -2.25])

    #     x_min = -5
    #     x_max = 5
    #     # Normalize state x to [-1,1]
    #     x_norm = 2 * (x - x_min) / (x_max - x_min) - 1.0

        
    #     h_min = -2.25
    #     h_max = 99.3125

    #     h = (x[0] - pos[0])**2 + (x[1] - pos[1])**2 - self.r**2

    #     h_norm = 2 * (h - h_min) / (h_max - h_min) - 1.0

    #     return x_norm, h_norm
    
    # def initialize_parameters(self):
    #     weight_values = []
    #     bias_values = []
    #     for i in range(len(self.layers_size) - 1):
    #         input_size = self.layers_size[i]
    #         output_size = self.layers_size[i+1]

    #         W_val = self.np_random.standard_normal((output_size, input_size))
    #         b_val = self.np_random.standard_normal((output_size, 1))

    #         weight_values.append(W_val.reshape(-1))
    #         bias_values.append(b_val.reshape(-1))

    #     flat_params = np.concatenate(weight_values + bias_values, axis=0)
    #     flat_params = cs.DM(flat_params)  
    #     return flat_params, weight_values, bias_values

    def initialize_parameters(self):
        """
        initialization for the neural network (he normal for relu)
        """
        weight_values = []
        bias_values = []
        for i in range(len(self.layers_size) - 1):
            fan_in = self.layers_size[i] #5 input dim # 7 input dim #7 input dim
            fan_out = self.layers_size[i + 1] #7 output dim # 7 output dim # 1 output dim

            if i < len(self.layers_size) - 2:
                # W_val = self.np_random.normal(
                #     loc=0.0, 
                #     scale=np.sqrt(2.0 / fan_in),
                #     size=(fan_out, fan_in)
                # )
                # bound = sqrt(3) * sqrt(2/fan_in) = sqrt(6/fan_in)
                bound = np.sqrt(6.0 / fan_in)
                W_val = self.np_random.uniform(low=-bound, high=bound, size=(fan_out, fan_in))
            else:
                # W_val = self.np_random.normal(
                #     loc=0.0, 
                #     scale=1,
                #     size=(fan_out, fan_in)
                # )
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                W_val = self.np_random.uniform(low=-bound, high=bound, size=(fan_out, fan_in))
            
            # biases = zero
            b_val = np.zeros((fan_out, 1))

            weight_values.append(W_val.reshape(-1))
            bias_values.append(b_val.reshape(-1))

        flat_params = np.concatenate(weight_values + bias_values, axis=0)
        flat_params = cs.DM(flat_params)
        return flat_params, weight_values, bias_values




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

    def __init__(self, dt, layers_list):
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
        self.weight_cbf = cs.DM([1e8])


        # decision variables
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        self.S_sym = cs.MX.sym("S", 1, self.horizon)

        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)


        # defining stuff for CBF
        x_new  = self.A @ self.x_sym + self.B @ self.u_sym 
        self.dynamics_f = cs.Function('f', [self.x_sym, self.u_sym], [x_new], ['x','u'], ['ode'])

        h = (self.x_sym[0]-(self.pos[0]))**2 + (self.x_sym[1]-(self.pos[1]))**2 - self.r**2 
        h_min = 0#-2.25
        h_max = 11.8125
        mu_h = (h_min + h_max) / 2.0
        sigma_h = (h_max - h_min) / np.sqrt(12)
        # mu_h = 6.7885445905109325# 4.692387613414234
        # sigma_h =   0.4382974988082045   #2.7606664837421535

        h = (self.x_sym[0] - (self.pos[0]))**2 + (self.x_sym[1] - (self.pos[1]))**2 - self.r**2
        h_norm = (h - mu_h) / sigma_h
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])
        self.h_func_norm = cs.Function('h', [self.x_sym], [h_norm], ['x'], ['cbf'])

        
        # intilization of the Neural Network
        self.nn = NN(layers_list)
        # self.nn_fn = self.nn.create_forward_function()

        self.alpha_nn = self.nn.get_alpha_nn()

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

        References
        ----------
        .. [1] Yuhang Xiong, Di-Hua Zhai, Mahdi Tavakoli, Yuanqing Xia. Discrete-time
        control barrier function: High-order case and adaptive case. *IEEE Transactions
        on Cybernetics*, 53(5), 3231-3239, 2022.

        Examples
        --------
        >>> import casadi as cs
        >>> A = cs.SX.sym("A", 2, 2)
        >>> B = cs.SX.sym("B", 2, 1)
        >>> x = cs.SX.sym("x", A.shape[0], 1)
        >>> u = cs.SX.sym("u", B.shape[1], 1)
        >>> dynamics = lambda x, u: A @ x + B @ u
        >>> M = cs.SX.sym("M")
        >>> c = cs.SX.sym("c")
        >>> gamma = cs.SX.sym("gamma")
        >>> alphas = [lambda z: gamma * z]
        >>> h = lambda x: M - c * x[0]  # >= 0
        >>> cbf = dcbf(h, x, u, dynamics, alphas)
        >>> print(cbf)
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
        x_norm, _ = self.nn.normalization_z(self.x_sym)
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f, [lambda y: self.alpha_nn(cs.vertcat(self.x_sym, self.h_func_norm(self.x_sym)))*y])

        return cs.Function('cbff', [self.x_sym, self.u_sym, self.nn.get_flat_parameters()], [cbf], ['x','u', 'alpha params'], ['cbff'])

    def cbf_const(self):

        """""
        used to construct cbf constraints 
        """

        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):

            cbf_const_list.append(cbf_func(self.X_sym[:,k], self.U_sym[:,k], self.nn.get_flat_parameters()) + self.S_sym[:,k]) 

        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        print(f"here is the self.cbf_constlist; {self.cbf_const_list}")
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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, self.nn.get_flat_parameters()),
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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, self.nn.get_flat_parameters(), rand_noise),
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
        X_lower_bound = -5 * np.array([1, 1, 1, 1])#-1e6 * 5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.array([1, 1, 1, 1])#1e6 * 5 * np.ones(mpc.ns  * (mpc.horizon))

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        # X_con + U_con + S_con + Sx_con
        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), np.zeros(self.horizon)) 
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), np.inf*np.ones(self.horizon))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, self.cbf_const_list) # opposite signs

 
        theta_vector = cs.vertcat(self.V_sym, self.P_diag, self.nn.get_flat_parameters())

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
                self.V_sym, self.P_sym, lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.nn.get_flat_parameters()
            ],
            [qlagrange_sens],
            [
                'A_sym', 'B_sym', 'b_sym', 'Q_sym', 'R_sym','V_sym', 'P_sym', 'lagrange_mult_x_lb_sym', 
                'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'inputs_NN'
            ],
            ['qlagrange_sens']
        )

        # qlagrange_fn_hessian = cs.Function(
        #     "qlagrange_fn_hessian",
        #     [
        #         self.A_sym, self.B_sym, self.b_sym, self.Q_sym, self.R_sym,
        #         self.V_sym, self.P_sym, lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
        #         lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.nn.get_flat_parameters()
        #     ],

        #     [qlagrange_hessian],
        #     [
        #         'A_sym', 'B_sym', 'b_sym', 'Q_sym', 'R_sym','V_sym', 'P_sym', 'lagrange_mult_x_lb_sym', 
        #         'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'inputs_NN'
        #     ],
        #     ['qlagrange_hessian']
        # )


        return qlagrange_fn, _
    
    def qp_solver_fn(self):
            #implementing optimiazation for one time step

            Hessian_sym = cs.MX.sym("Hessian", self.theta.shape[0]*self.theta.shape[0])
            gradient_sym = cs.MX.sym("gradient", self.theta.shape[0])

            delta_theta = cs.MX.sym("delta_theta", self.theta.shape[0])
            theta = cs.MX.sym("delta_theta", self.theta.shape[0])

            # alpha_sym = cs.MX.sym("alpha")

            alpha_vec = cs.MX.sym("alpha_vec", self.theta.shape[0])

            lambda_reg = 1e-6
 
            qp = {
                "x": cs.vertcat(delta_theta),
                "p": cs.vertcat(theta, Hessian_sym, gradient_sym, alpha_vec),
                "f": 0.5*delta_theta.T @ Hessian_sym.reshape((self.theta.shape[0], self.theta.shape[0])) @ delta_theta +  gradient_sym.T @ (alpha_vec * delta_theta) + lambda_reg/2 * delta_theta.T @ delta_theta, 
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

        def __init__(self, params_innit, seed, alpha, sampling_time, gamma, decay_rate, layers_list, noise_scalingfactor, noise_variance, patience_threshold, lr_decay_factor):
            self.seed = seed

            # enviroment class
            self.env = env(sampling_time=sampling_time)

            # mpc class
            self.mpc = MPC(sampling_time, layers_list)
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
            self.qlagrange_fn_jacob, _ = self.mpc.generate_symbolic_mpcq_lagrange()

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

        
        # def save_figures(self, figures, experiment_folder):

        #     save_choice = True#input("Save the figure? (y/n): ")
        #     if save_choice ==  True:#"y":
        #         os.makedirs(experiment_folder, exist_ok=True) # make directory ( exist_ok makes sure it doenst throw exception when it alreadt exists)
        #         for fig, filename in figures: 
        #             file_path = os.path.join(experiment_folder, filename) # add the file to directory
        #             fig.savefig(file_path)
        #             print(f"Figure saved as: {file_path}")
        #     else:
        #         print("Figure not saved")
        
        
        def save_figures(self, figures, experiment_folder, save_in_subfolder=False):
            
            save_choice = True  
            
            if save_choice:
                
                if save_in_subfolder:
                    target_folder = os.path.join(experiment_folder, "learning_process")
                else:
                    target_folder = experiment_folder

                os.makedirs(target_folder, exist_ok=True)
                
                for fig, filename in figures:
                    file_path = os.path.join(target_folder, filename)
                    fig.savefig(file_path)
                    print(f"Figure saved as: {file_path}")
            else:
                print("Figure not saved")


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

            self.V_MPCsolution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat,  params["nn_params"]),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract solution from solver 
            u_opt = self.V_MPCsolution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]


            return u_opt, self.V_MPCsolution["f"]
        
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
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)


            self.V_MPC_randsolution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["nn_params"], rand),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            u_opt = self.V_MPC_randsolution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]

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
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            self.Q_MPCsolution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["nn_params"]),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            u_opt = self.Q_MPCsolution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            lagrange_mult_g = self.Q_MPCsolution["lam_g"]
            lam_lbx = -cs.fmin(self.Q_MPCsolution["lam_x"], 0)
            lam_ubx = cs.fmax(self.Q_MPCsolution["lam_x"], 0)
            lam_p = self.Q_MPCsolution["lam_p"]

            return self.Q_MPCsolution["x"], self.Q_MPCsolution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p
            
        def stage_cost(self, action, x):
            """Computes the stage cost :math:`L(s,a)`.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 3, 3])
            Rstage = np.diag([1, 1])

            state = x
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action
            )
            
        

        def parameter_updates(self, params, B_update_avg):

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            theta_vector_num = cs.vertcat(params["V0"], P_diag, params["nn_params"])


            identity = np.eye(theta_vector_num.shape[0])

            # print(f"before updates : {theta_vector_num}")

            # alpha_vec is resposible for the updates
            alpha_vec = cs.vertcat(self.alpha*np.ones(3), self.alpha, self.alpha, self.alpha*np.ones(theta_vector_num.shape[0]-5))
            # alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-2), self.alpha,self.alpha*1e-5)

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, B_update_avg)
            theta_vector_num_toprint = theta_vector_num - (alpha_vec * y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")

            # lbx = cs.vertcat(-np.inf*np.ones(5), 0.1*np.abs(theta_vector_num[5:]))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.1*np.abs(theta_vector_num[5:]))

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.001*np.ones(theta_vector_num.shape[0]-5))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.001*np.ones(theta_vector_num.shape[0]-5))
         
            
            # constrained update qp update
            self.QPsolution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, identity.flatten(), B_update_avg, alpha_vec),
                    lbg=cs.vertcat(np.zeros(5), -np.inf*np.ones(theta_vector_num.shape[0]-5)),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    # ubx = ubx,
                    # lbx = lbx
                )
            stats = self.qp_solver.stats()
            if stats["success"] == False:
                print("QP NOT SUCCEEDED")
                print("Returned solution x:", self.QPsolution["x"])
                print("Objective function value f:", self.QPsolution["f"])
                theta_vector_num = theta_vector_num
            else:
                theta_vector_num = theta_vector_num + self.QPsolution["x"]


            print(f"theta_vector_num: {theta_vector_num}")

            V0_shape = 1
            P_diag_shape = self.ns*1
            #constructing the diagonal posdef P matrix 
            P_posdef = cs.diag(theta_vector_num[V0_shape:V0_shape+P_diag_shape])

            params["V0"] = theta_vector_num[:V0_shape]
            params["P"] = P_posdef
            params["nn_params"] = theta_vector_num[V0_shape+P_diag_shape:]

            return params
        

        def rl_trainingloop(self, episode_duration, num_episodes, replay_buffer, episode_updatefreq, experiment_folder):
    
            #to store for plotting
            params_history_V0 = [self.params_innit["V0"]]
            params_history_P = [self.params_innit["P"]]

            #for the for loop
            params = self.params_innit
            
            x, _ = self.env.reset(seed=self.seed, options={})

            stage_cost_history = []
            sum_stage_cost_history = []
            TD_history = []
            TD_temp = []
            TD_episode = []

            # A_update_lst = []
            # B_update_lst = []
            # A_update_lst = []
            # B_update_lst = []
            # chagning lst to buffer for replay
            B_update_buffer = deque(maxlen=replay_buffer)
            

            states = [(x)]

            actions = []

            self.fwd_func = self.mpc.nn.numerical_forward()

            #intialize
            k = 0
            self.error_happened = False
            # try:
            for i in range(1,episode_duration*num_episodes):

                
                rand = self.noise_scalingfactor*self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u = self.V_MPC_rand(params=params, x=x, rand = rand)
                u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)

                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    print("Returned solution x:", self.V_MPC_randsolution["x"])
                    print("Objective function value f:", self.V_MPC_randsolution["f"])
                    self.error_happened = True
     
                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _ = self.Q_MPC(params=params, action=u, x=x)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    print("Returned solution x:", self.Q_MPCsolution["x"])
                    print("Objective function value f:", self.Q_MPCsolution["f"])
                    self.error_happened = True

                stage_cost = self.stage_cost(action=u,x=x)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)

                #calculate V value

                # print(f"x_2: {x}")
                # print(f"params_3: {params}")

                _, Vcost = self.V_MPC(params=params, x=x)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED")
                    print("Returned solution x:", self.V_MPCsolution["x"])
                    print("Objective function value f:", self.V_MPCsolution["f"])
                    self.error_happened = True


                #time penalty
                #TD update time_penalty*k
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
                    V_sym=params["V0"],
                    P_sym = params["P"],
                    lagrange_mult_x_lb_sym=lam_lbx,
                    lagrange_mult_x_ub_sym=lam_ubx,
                    lagrange_mult_g_sym=lagrange_mult_g,
                    X=X, U=U, S=S, inputs_NN=params["nn_params"]
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

                # # reset state to learn from somehwhere else:
                if (k == episode_duration): 
                    print(f"k: {k}")
                    print(f"i: {i}")
                    
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:
                        print (f"updatedddddd")
                        B_update_avg = np.mean(B_update_buffer, 0)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_V0.append(params["V0"])
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.sum(TD_episode))

                    self.update_learning_rate(np.sum(stage_cost_history))

                    stage_cost_history = []
                    TD_episode = []

                    x, _ = self.env.reset(seed=self.seed, options={})
                    k=0

                    print("reset")


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 20) == 0:
                        states = np.array(states)
                        actions = np.array(actions)
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
                        # plt.show()

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
                        self.save_figures(figures_training, experiment_folder, True)

            
                    states = [(x)]
                    TD_temp = []
                    actions = []
                # k counter    
                k+=1
                
                #counter
                if i % 1000 == 0:
                    print(f"{i}/{episode_duration*num_episodes}")  

            #show trajectories
            # plt.show()
            # plt.close()


            params_history_V0 = np.asarray(params_history_V0)
            params_history_P = np.asarray(params_history_P)

            TD_history = np.asarray(TD_history)
            sum_stage_cost_history = np.asarray(sum_stage_cost_history)
       
            figV0 = plt.figure(figsize=(10, 5))
            plt.plot(params_history_V0[:, 0], label="V0")
            plt.title("Parameter: V0")
            plt.xlabel("Training Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            # plt.close()
            #plt.show()

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
            #plt.show()

            figstagecost = plt.figure(figsize=(10, 5))
            plt.plot(sum_stage_cost_history, 'o', label="Stage Cost")
            plt.yscale('log') 
            plt.title("Stage Cost Over Training (Log Scale)")
            plt.xlabel("Training Step")
            plt.ylabel("Stage Cost")
            plt.legend()
            plt.grid(True)
            #plt.show()

            figtd = plt.figure(figsize=(10, 5))
            plt.plot(TD_history[:,0], 'o', label="TD")
            plt.yscale('log')
            plt.title("TD Over Training (Log Scale)")
            plt.xlabel("Training Step")
            plt.ylabel("TD")
            plt.legend()
            plt.grid(True)
            #plt.show()

            figures_to_save = [
                (figV0, "V0"),
                (figP, "P"),
                (figstagecost, "stagecost"),
                (figtd, "TD")

            ]

            self.save_figures(figures_to_save, experiment_folder)
            return params
        
        