
import gymnasium as gym 
import numpy as np
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr


class NN:

    def __init__(self, layers_size):
        """
        layers_size = list of layer sizes, including input and output sizes: [4, 7, 7, 1]
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

    def sigmoid(self, x):
        #sigmoid activation, used for the last output layer
        return 1 / (1 + cs.exp(-x))
    
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
                self.activations.append(self.sigmoid)
            else:
                self.activations.append(self.relu)

    def forward(self, x):
        """
        memeber function to perform the forward pass
        """
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
    
    def initialize_parameters(self):
        weight_values = []
        bias_values = []
        for i in range(len(self.layers_size) - 1):
            input_size = self.layers_size[i]
            output_size = self.layers_size[i+1]

            W_val = self.np_random.standard_normal((output_size, input_size))
            b_val = self.np_random.standard_normal((output_size, 1))

            weight_values.append(W_val.reshape(-1))
            bias_values.append(b_val.reshape(-1))

        flat_params = np.concatenate(weight_values + bias_values, axis=0)
        flat_params = cs.DM(flat_params)  
        return flat_params, weight_values, bias_values


    

# Test the network
    
layer_sizes = [4, 7, 7, 1]
    
h = (0-(-2))**2 + (0.15-(-2.25))**2 - 1.5**2 
nn_model = NN(layer_sizes)
nn_fn = nn_model.create_forward_function()
# test
#x_val = np.array([0, 0.15, 0, 0, h])
W0_val = np.random.randn(7, 4)
b0_val = np.random.randn(7, 1)
W1_val = np.random.randn(7, 7)
b1_val = np.random.randn(7, 1)
W2_val = np.random.randn(1, 7)
b2_val = np.random.randn(1, 1)


x_val = np.array([0, 0.15, 0, 0])


list, weight_values, bias_values = nn_model.initialize_parameters()

y_val = nn_fn(x_val, weight_values[0].reshape(7,4), weight_values[1].reshape(7,7), weight_values[2], bias_values[0], bias_values[1], bias_values[2])
print("Network output:", y_val)




##############################################



"""""
Kerim Dzhumageldyev

2D double integrator implementation with CBF decay



"""""

#HORIZON ISSUES

import gymnasium as gym 
import numpy as np
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr

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
        
        #dynamics
        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.b_sym = cs.MX.sym("b", self.ns)

        #MPC params
        self.P_sym = cs.MX.sym("P", self.ns, self.ns)
        self.Q_sym = cs.MX.sym("Q", self.ns, self.ns)
        self.R_sym = cs.MX.sym("R", self.na, self.na)
        self.V_sym = cs.MX.sym("V0")
      

        #weight on the slac"V0" variables
        self.weight = cs.DM([1e2 ,1e2 ,1e2 ,1e2])

        # decision variables
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        self.S_sym = cs.MX.sym("S", self.ns, self.horizon)

        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)


        # defining stuff for CBF
        x_new  = self.A @ self.x_sym + self.B @ self.u_sym 
        self.dynamics_f = cs.Function('f', [self.x_sym, self.u_sym], [x_new], ['x','u'], ['ode'])

        h = (self.x_sym[0]-(self.pos[0]))**2 + (self.x_sym[1]-(self.pos[1]))**2 - self.r**2 
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])


        # intilization of the Neural Network
        self.nn = NN([1, 7, 7, 1])
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
    
    # def h(self):
    #     h = (self.x_sym[0]-(self.pos[0]))**2 + (self.x_sym[1]-(self.pos[1]))**2 - self.r**2 
    #     return cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])

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
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f, [lambda y: self.alpha_nn(self.h_func(self.x_sym))*y])
        return cs.Function('cbff', [self.x_sym, self.u_sym, self.nn.get_flat_parameters()], [cbf], ['x','u', 'alpha params'], ['cbff'])

    def cbf_const(self):

        """""
        used to construct cbf constraints 
        """

        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):
            cbf_const_list.append(cbf_func(self.X_sym[:,k], self.U_sym[:,k], self.nn.get_flat_parameters()))   
        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        return 
    
    def objective_method(self):

        """""
        stage cost calculation
        """
        # why doesnt work? --> idk made it in line
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k] + self.weight.T @ self.S_sym[:, k])
            for k in range(self.horizon)
        )

        terminal_cost = cs.bilin(self.P_sym, self.X_sym[:, -1])

  

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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, P_sym_flat, Q_sym_flat, R_sym_flat, self.nn.get_flat_parameters()),
            "f": self.objective, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }
        # print(f"cbf const list: {cs.simplify(self.cbf_const_list)}")
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": False,
            "calc_lam_x": True,
            "eval_errors_fatal": True,
            "calc_lam_p": False,
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

        rand_noise = cs.MX.sym("rand_noise", 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, S_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, P_sym_flat, Q_sym_flat, R_sym_flat, self.nn.get_flat_parameters(), rand_noise),
            "f": self.objective + rand_noise, 
            "g": cs.vertcat(self.state_const_list, -self.cbf_const_list),
        }

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": False,
            "calc_lam_x": True,
            "eval_errors_fatal": True,
            "calc_lam_p": False,
            "fatrop": {"max_iter": 500, "print_level": 0},
        }

        MPC_solver = cs.nlpsol("solver", "fatrop", nlp, opts)


        return MPC_solver
    
    def generate_symbolic_mpcq_lagrange(self):
        """
        constructs MPC action state value function solver
        """
        self.state_const()
        self.objective_method()

        X_flat = cs.reshape(self.X_sym, -1, 1)  # Flatten 
        S_flat = cs.reshape(self.S_sym, -1, 1) # Flatten
        U_flat = cs.reshape(self.U_sym, -1, 1)
        opt_solution = cs.vertcat(X_flat, U_flat, S_flat)

        lagrange_mult_x_lb_sym = cs.MX.sym("lagrange_mult_x_lb_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + self.ns * (self.horizon))
        lagrange_mult_x_ub_sym = cs.MX.sym("lagrange_mult_x_ub_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + self.ns * (self.horizon))
        lagrange_mult_g_sym = cs.MX.sym("lagrange_mult_g_sym", 1*self.ns*(self.horizon) + self.horizon)

        # fix lower bound
        # construct 
        X_lower_bound = -np.inf* np.ones(self.ns * (self.horizon))
        X_upper_bound = np.inf* np.ones(self.ns * (self.horizon)) 

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        S_lower_bound = np.zeros(self.ns * (self.horizon))
        S_upper_bound = np.inf * np.ones(self.ns * (self.horizon))  

        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), cs.DM(S_lower_bound))  # what action is it grabbing
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), cs.DM(S_upper_bound))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, self.cbf_const_list) # opposite signs

        #flatten matrices
        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)
        P_sym_flat = cs.reshape(self.P_sym , -1, 1)
        Q_sym_flat = cs.reshape(self.Q_sym , -1, 1)
        R_sym_flat = cs.reshape(self.R_sym , -1, 1)

        theta_vector = cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, P_sym_flat, Q_sym_flat, R_sym_flat, self.nn.get_flat_parameters())

        qlagrange = self.objective + lagrange1 + lagrange2 + lagrange3

        #computing derivative of lagrangian for A
        qlagrange_hessian, qlagrange_sens = cs.hessian(qlagrange, theta_vector)

        #transpose it becase cs.hessian gives it differently than cs.jacobian
        qlagrange_sens = qlagrange_sens

        qlagrange_fn = cs.Function(
            "qlagrange_fn",
            [
                self.A_sym, self.B_sym, self.b_sym, self.V_sym, self.P_sym, self.Q_sym, self.R_sym,
                lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.nn.get_flat_parameters()
            ],
            [qlagrange_sens],
            [
                'A_sym', 'B_sym', 'b_sym', 'V_sym', 'P_sym', 'Q_sym', 'R_sym', 'lagrange_mult_x_lb_sym', 
                'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'inputs_NN'
            ],
            ['qlagrange_sens']
        )

        qlagrange_fn_hessian = cs.Function(
            "qlagrange_fn_hessian",
            [
                self.A_sym, self.B_sym, self.b_sym, self.V_sym, self.P_sym, self.Q_sym, self.R_sym,
                lagrange_mult_x_lb_sym, lagrange_mult_x_ub_sym, 
                lagrange_mult_g_sym, X_flat, U_flat, S_flat, self.nn.get_flat_parameters()
            ],

            [qlagrange_hessian],
            [
                'A_sym', 'B_sym', 'b_sym', 'V_sym', 'P_sym', 'Q_sym', 'R_sym', 'lagrange_mult_x_lb_sym', 
                'lagrange_mult_x_ub_sym', 'lagrange_mult_g_sym', 'X', 'U', 'S', 'inputs_NN'
            ],
            ['qlagrange_hessian']
        )

        return qlagrange_fn, qlagrange_fn_hessian

    

def MPC_func(x, mpc, params):
        dt = 0.2

        solver_inst = mpc.MPC_solver() 

        
        # bounds
        X_lower_bound = -5 * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = 5 * np.ones(mpc.ns  * (mpc.horizon))

        S_lower_bound = np.zeros(mpc.ns * (mpc.horizon))
        S_upper_bound = np.inf * np.ones(mpc.ns * (mpc.horizon)) 

        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon)) 

        state_const_lbg = np.zeros(1*mpc.ns * (mpc.horizon))
        state_const_ubg = np.zeros(1*mpc.ns  * (mpc.horizon))

        cbf_const_lbg = -np.inf * np.ones(1*(mpc.horizon))
        cbf_const_ubg = np.zeros(1*(mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), X_lower_bound, U_lower_bound, S_lower_bound])  
        ubx = np.concatenate([np.array(x).flatten(), X_upper_bound, U_upper_bound, S_upper_bound])

        lbg = np.concatenate([state_const_lbg, cbf_const_lbg])  
        ubg = np.concatenate([state_const_ubg, cbf_const_ubg])

        #params of MPC
        P = params["P"]
        Q = params["Q"]
        R = params["R"]
        b = params["b"]
        V = params["V0"]


        # A = np.array([
        #     [1, 0, dt, 0], 
        #     [0, 1, 0, dt], 
        #     [0, 0, 1, 0], 
        #     [0, 0, 0, 1]
        # ])

        # B = np.array([
        #     [0.5 * dt**2, 0], 
        #     [0, 0.5 * dt**2], 
        #     [dt, 0], 
        #     [0, dt]
        # ])
        

        #print(nn_params)
        #flatten
        A_flat = cs.reshape(params["A"] , -1, 1)
        B_flat = cs.reshape(params["B"] , -1, 1)
        P_flat = cs.reshape(P , -1, 1)
        Q_flat = cs.reshape(Q , -1, 1)
        R_flat = cs.reshape(R , -1, 1)

        solution = solver_inst(p = cs.vertcat(A_flat, B_flat, b, V, P_flat, Q_flat, R_flat, params["nn_params"]),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )


        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"]

def run_simulation(params, env):

    env = env(sampling_time=0.2)


   
    state, _ = env.reset(seed=42, options={})
    states = [state[:2]]
    actions = []
    mpc = MPC(0.2)

    for _ in range(200):
        action, _ = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state[:2])
        actions.append(action)

        # if (1.5)**2 >= ((state[0]+2)**2 + (state[1]+2.25)**2):
        #     break

    states = np.array(states)
    plt.plot(
        states[:, 0], states[:, 1],
        "o-", label = "Trajectory"
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
    plt.show()




def calculate_trajectory_length(states):
    # compute eucldian distance and then sum
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)



def run_simulation(params, env):

    env = env(sampling_time=0.2)


   
    state, _ = env.reset(seed=42, options={})
    states = [state[:2]]
    actions = []
    mpc = MPC(0.2)

    for _ in range(600):
        action, _ = MPC_func(state, mpc, params)

        action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
        state, _, done, _, _ = env.step(action)
        states.append(state[:2])
        actions.append(action)

        # if (1.5)**2 >= ((state[0]+2)**2 + (state[1]+2.25)**2):
        #     break
        if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
            break

    states = np.array(states)
    plt.plot(
        states[:, 0], states[:, 1],
        "o-", label = "Trajectory"
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
    plt.show()

    trajectory_length = calculate_trajectory_length(states)
    print(f"Total trajectory length: {trajectory_length:.3f} units")





####################### RL #######################
best_stage_cost = np.inf

class RLclass:
        # learning rate
        alpha = 2e-4

        def __init__(self, params_innit):
            seed = 69

            # enviroment class
            self.env = env(sampling_time=0.2)

            # mpc class
            self.mpc = MPC(0.2)
            self.x0, _ = self.env.reset(seed=seed, options={})
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit
            self.alpha = RLclass.alpha
            
            # bounds
            self.X_lower_bound = -5 * np.ones(self.ns * (self.horizon))
            self.X_upper_bound = 5 * np.ones(self.ns  * (self.horizon))

            self.S_lower_bound = np.zeros(self.ns * (self.horizon))
            self.S_upper_bound = np.inf * np.ones(self.ns * (self.horizon)) 

            self.state_const_lbg = np.zeros(1*self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1*self.ns  * (self.horizon))

            self.cbf_const_lbg = -np.inf * np.ones(1*(self.horizon))
            self.cbf_const_ubg = np.zeros(1*(self.horizon))

            # gamma
            self.gamma = 0.95

            self.np_random = np.random.default_rng(seed=seed)

            #solver instance
            self.solver_inst = self.mpc.MPC_solver()

            #get parameter sensitivites
            self.qlagrange_fn_jacob, self.qlagrange_fn_hessian  = self.mpc.generate_symbolic_mpcq_lagrange()

            #randomness
            self.solver_inst_random =self.mpc.MPC_solver_rand()

            


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
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound, self.S_lower_bound])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound, self.S_upper_bound])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_flat = cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            #process nn_params to be flattened or maybe dont because they already are flattened

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_flat, Q_flat, R_flat, params["nn_params"] ),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]


            return u_opt, solution["f"]
        
        def V_MPC_rand(self, params, x):
            rand =5*self.np_random.normal(loc=0, scale=3)
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound, self.S_lower_bound])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound, self.S_upper_bound])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_flat = cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)


            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_flat, Q_flat, R_flat, params["nn_params"], rand),
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
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, np.asarray(action).flatten(), U_lower_bound, self.S_lower_bound])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound, np.asarray(action).flatten(), U_upper_bound, self.S_upper_bound])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_flat = cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_flat, Q_flat, R_flat, params["nn_params"]),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            

            lagrange_mult_g= solution["lam_g"]
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]

            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p
            
        def stage_cost(self, action, x):
            """Computes the stage cost :math:`L(s,a)`.
            """
            #punishing distance from the center (big)
            Qstage = np.diag([100, 100, 1, 1])
            #punishing action (small)
            Rstage = np.diag([0.1, 0.1])
            #punishing safety violation (big)
            lambda_cbf = 1e3
            state = x
            return 0.5*(
                state.T @ Qstage @ state
                + action.T @ Rstage @ action
                #safety violation+
                #+ lambda_cbf * max(0,  1.5 ** 2 - ( (state[0] - (-2)) ** 2 + (state[1] - (-2.25)) ** 2 ))
            )
            
        

        def parameter_updates(self, params, A_update_avg, B_update_avg):

            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_flat = cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            theta_vector_num = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_flat, Q_flat, R_flat, params["nn_params"])

            L  = self.cholesky_added_multiple_identity(A_update_avg)

            A_update_chom = L @ L.T
            
            y = np.linalg.solve(A_update_chom, B_update_avg)


            theta_vector_num = theta_vector_num - self.alpha * y
            
            Avec_shape = self.ns*self.ns
            Bvec_shape = self.ns*self.na
            bvec_shape = self.ns
            V0_shape = 1
            P_shape = self.ns*self.ns
            Q_shape = self.ns*self.ns
            R_shape = self.na*self.na

            params["A"] = theta_vector_num[: Avec_shape].reshape((self.ns, self.ns))
            params["B"] = theta_vector_num[Avec_shape : Avec_shape+Bvec_shape].reshape((self.ns, self.na))
            params["b"] = theta_vector_num[Avec_shape+Bvec_shape : Avec_shape+Bvec_shape+bvec_shape]
            params["V0"] = theta_vector_num[Avec_shape+Bvec_shape+bvec_shape : Avec_shape+Bvec_shape+bvec_shape+V0_shape]
            params["P"] = theta_vector_num[Avec_shape+Bvec_shape+bvec_shape+V0_shape:Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape].reshape((self.ns, self.ns))
            params["Q"] = theta_vector_num[Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape:Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape+Q_shape].reshape((self.ns, self.ns))
            params["R"] = theta_vector_num[Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape+Q_shape:Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape+Q_shape+R_shape].reshape((self.na, self.na))
            params["nn_params"] = theta_vector_num[Avec_shape+Bvec_shape+bvec_shape+V0_shape+P_shape+Q_shape+R_shape:]

            return params
        

        def rl_trainingloop(self, training_duration, update_stepcnt):
    
            #to store for plotting
            params_history_A = [self.params_innit["A"]]
            params_history_B = [self.params_innit["B"]]
            params_history_b = [self.params_innit["b"]]
            params_history_V0 = [self.params_innit["V0"]]
            params_history_P = [self.params_innit["P"]]
            params_history_Q = [self.params_innit["Q"]]
            params_history_R = [self.params_innit["R"]]
            

            # #initial action
            # u_mpcV, _ = self.V_MPC(params=self.params_innit, x=self.x0)


            #for the for loop
            params = self.params_innit
            
            x = self.x0
            # u = u_mpcV
            


            stage_cost_history = []
            TD_history = []

            A_update_lst = []
            B_update_lst = []
            
            u_history = []
            x_history = [np.asarray(x).flatten()]
            # u_history = [np.asarray(u).flatten()]
            #intialize
            k = 0
            #step_cntr = 1e7
            try:
                for i in range(1,training_duration):

                    u = self.V_MPC_rand(params=params, x=x)
                
                    u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)

                    #calculate Q value
                    solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _ = self.Q_MPC(params=params, action=u, x=x)

                    stage_cost = self.stage_cost(action=u,x=x)
                    
                    # enviroment update step
                    x, _, done, _, _ = self.env.step(u)

                    #calculate V value
                    _, Vcost = self.V_MPC(params=params, x=x)


                    #time penalty
                    time_penalty = 0.1
                    #TD update
                    TD = (stage_cost + time_penalty*k) + self.gamma*Vcost - Qcost

                    U = solution[self.ns * (self.horizon+1):self.na * (self.horizon) + self.ns * (self.horizon+1)] 
                    X = solution[:self.ns * (self.horizon+1)] 
                    S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1):]
                    
                    #parameter update: A
                    qlagrange_numeric_jacob=  self.qlagrange_fn_jacob(
                        A_sym=params["A"],
                        B_sym=params["B"],
                        b_sym=params["b"],
                        V_sym=params["V0"],
                        P_sym = params["P"],
                        Q_sym = params["Q"],
                        R_sym = params["R"],
                        lagrange_mult_x_lb_sym=lam_lbx,
                        lagrange_mult_x_ub_sym=lam_ubx,
                        lagrange_mult_g_sym=lagrange_mult_g,
                        X=X, U=U, S=S, inputs_NN=params["nn_params"]
                    )['qlagrange_sens']

                    qlagrange_numeric_hess=  self.qlagrange_fn_hessian(
                        A_sym=params["A"],
                        B_sym=params["B"],
                        b_sym=params["b"],
                        V_sym=params["V0"],
                        P_sym = params["P"],
                        Q_sym = params["Q"],
                        R_sym = params["R"],
                        lagrange_mult_x_lb_sym=lam_lbx,
                        lagrange_mult_x_ub_sym=lam_ubx,
                        lagrange_mult_g_sym=lagrange_mult_g,
                        X=X, U=U, S=S, inputs_NN=params["nn_params"]
                    )['qlagrange_hessian']

                    outer_product = qlagrange_numeric_jacob @ qlagrange_numeric_jacob.T

                    # second order update
                    B_update = -TD*qlagrange_numeric_jacob
                    B_update_lst.append(B_update)

                    if qlagrange_numeric_hess.nnz() > 0:
                        A_update = -TD*qlagrange_numeric_hess + outer_product
                        A_update_lst.append(A_update) 
                        
                    else:
                        A_update = outer_product
                        A_update_lst.append(A_update)

                    # replay buffer
                    if i % update_stepcnt == 0:
                        A_update_avg = np.mean(A_update_lst, 0)
                        B_update_avg = np.mean(B_update_lst, 0)

                        # print(f"A_update_avg: {A_update_avg}")
                        # print(f"B_update_avg: {B_update_avg}")

                        params =self.parameter_updates(params = params, A_update_avg = A_update_avg, B_update_avg = B_update_avg)

                        params_history_A.append(params["A"])
                        params_history_B.append(params["B"])
                        params_history_b.append(params["b"])
                        params_history_V0.append(params["V0"])
                        params_history_P.append(params["P"])
                        params_history_Q.append(params["Q"])
                        params_history_R.append(params["R"])
                        

                        A_update_lst = []
                        B_update_lst = []

                
                    stage_cost_history.append(cs.DM(stage_cost + time_penalty*k))
                    TD_history.append(TD)
                    x_history.append(np.asarray(x).flatten())
                    u_history.append(np.asarray(u).flatten())
                    
                    if i % 1000 == 0:
                        print(f"{i}/{training_duration}")

                    # reset state to learn from somehwhere else:
                    if ((1e-2 > np.abs(x[0])) and (1e-2 > np.abs(x[1]))) or (np.abs(x[0]) > 6) or (np.abs(x[1]) > 6): #or step_cntr == k:
                        x = self.x0
                        k=0
                        step_cntr = k + 100
                        while True:
                            x[0] = self.np_random.uniform(-5, -4)
                            x[1] = self.np_random.uniform(-5, -2)
                            # Check if the state is outside the forbidden circle
                            if (x[0] + 2)**2 + (x[1] + 2.25)**2 > 1.5**2:
                                break
                        # if stage_cost_history[-1] < best_stage_cost:
                        #     params_best = params
                        #     best_stage_cost = stage_cost_history[-1]

                        print("reset")
                    k+=1  
            except Exception as e:
                    print("whoops something wrong", e)
            finally:    

                params_history_A = np.asarray(params_history_A)
                params_history_B = np.asarray(params_history_B)
                params_history_b = np.asarray(params_history_b)
                params_history_V0 = np.asarray(params_history_V0)
                params_history_P = np.asarray(params_history_P)
                params_history_Q = np.asarray(params_history_Q)
                params_history_R = np.asarray(params_history_R)
                
                TD_history = np.asarray(TD_history)
                stage_cost_history = np.asarray(stage_cost_history)
                x_history = np.asarray(x_history)
                u_history = np.asarray(u_history)
            
                print(stage_cost_history)
                #plotting
                plt.figure(figsize=(10, 5))
                for i in range(params_history_A.shape[1]):
                    for j in range(params_history_A.shape[2]):
                        plt.plot(params_history_A[:, i, j], label=f"A[{i},{j}]")
                plt.title("Parameter: A")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                for i in range(params_history_B.shape[1]):
                    for j in range(params_history_B.shape[2]):
                        plt.plot(params_history_B[:, i, j], label=f"B[{i},{j}]")
                plt.title("Parameter: B")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                for i in range(params_history_b.shape[1]):
                    plt.plot(params_history_b[:, i], label=f"b[{i}]")
                plt.title("Parameter: b")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                plt.plot(params_history_V0[:, 0], label="V0")
                plt.title("Parameter: V0")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                for i in range(params_history_P.shape[1]):
                    for j in range(params_history_P.shape[2]):
                        plt.plot(params_history_P[:, i, j], label=f"P[{i},{j}]")
                plt.title("Parameter: P")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()


                plt.figure(figsize=(10, 5))
                for i in range(params_history_Q.shape[1]):
                    for j in range(params_history_Q.shape[2]):
                        plt.plot(params_history_Q[:, i, j], label=f"Q[{i},{j}]")
                plt.title("Parameter: Q")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                for i in range(params_history_R.shape[1]):
                    for j in range(params_history_R.shape[2]):
                        plt.plot(params_history_R[:, i, j], label=f"R[{i},{j}]")
                plt.title("Parameter: R")
                plt.xlabel("Training Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                # plt.close()
                #plt.show()

                plt.figure(figsize=(10, 5))
                plt.plot(stage_cost_history[:,0], 'o', label="Stage Cost")
                plt.yscale('log') 
                plt.title("Stage Cost Over Training (Log Scale)")
                plt.xlabel("Training Step")
                plt.ylabel("Stage Cost")
                plt.legend()
                plt.grid(True)
                #plt.show()
                print(stage_cost_history[-1])

                plt.figure(figsize=(10, 5))
                plt.plot(TD_history[:,0], 'o', label="TD")
                plt.yscale('log')
                plt.title("TD Over Training (Log Scale)")
                plt.xlabel("Training Step")
                plt.ylabel("TD")
                plt.legend()
                plt.grid(True)
                plt.show()
                # real ssytem state and action plotting

                # x_bound_1 = [-5, 5]  
                # x_bound_2 = [-5, 5]
                # u_bounds = [-1, 1]  


                # plt.figure(figsize=(10, 5))
                # plt.plot(x_history[:, 0], label="s₁")
                # plt.hlines(x_bound_1, 0, len(x_history), colors='red', label="Bounds")
                # plt.title("State s₁")
                # plt.grid()
                # plt.legend()
                # plt.xlabel("Training Step")
                # #plt.show()

                # # State 2
                # plt.figure(figsize=(10, 5))
                # plt.plot(x_history[:, 1], label="s₂")
                # plt.hlines(x_bound_2, 0, len(x_history), colors='red', label="Bounds")
                # plt.title("State s₂")
                # plt.grid()
                # plt.legend()
                # plt.xlabel("Training Step")
                # #plt.show()

                # print(f"Concluded with resturns:{stage_cost_history.sum()}")
                # # Control input
                # plt.figure(figsize=(10, 5))
                # plt.plot(u_history[:, 0], label="a")
                # plt.hlines(u_bounds, 0, len(u_history), colors='red', label="Bounds")
                # plt.title("Control Input a")
                # plt.grid()
                # plt.legend()
                # plt.xlabel("Training Step")
                # plt.show()


            return params
        


training_duration =  200000
update_stepcnt = 1
dt = 0.2

params_innit = {
    # State matrices
    "A": cs.DM([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ]) ,
    "B": cs.DM([
            [0.5 * dt**2, 0], 
            [0, 0.5 * dt**2], 
            [dt, 0], 
            [0, dt]
        ]),
    # Learned parameters
    "b":  cs.DM([0, 0, 0, 0]),
    "V0": cs.DM(0.0),
    "P" : 100*np.identity(4),
    "Q" : 10*np.identity(4),
    "R" : np.identity(2),
}

# params_innit["nn_params"] = list

# run_simulation(params_innit, env)

# rl = RLclass(params_innit=params_innit)
# params = rl.rl_trainingloop(training_duration=training_duration, update_stepcnt=update_stepcnt)

# print(params)
# run_simulation(params, env)


