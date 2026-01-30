# no warmstart
        
import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
import copy
from gymnasium.spaces import Box
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque
import pandas as pd

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED, CONSTRAINTS_U



class Obstacles:

    def __init__(self, positions, radii):

        """
        positions: list of positions of the obstacles
        radius: list of radius of the obstacles
        """
        self.positions = positions
        self.radii = radii
        self.obstacle_num = len(positions)
        self.dt = SAMPLING_TIME

    def h_obsfunc(self, x, xpred_list, ypred_list):
        """
        resturns list of CBF funcs ( h(x) ) but in a numerical format
        """
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        h_list = []
        for (r, x_pred, y_pred )in zip(self.radii, xpred_list, ypred_list):
            h_list.append((x[0] - x_pred)**2 + (x[1] - y_pred)**2 - r**2)
        return h_list

    def make_h_functions(self):
        """
        resturns list of CBF funcs ( h(x) ) but for in a function format
        """
        funcs = []
        x = cs.MX.sym("x", 4)
        
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        
        xpred_list = cs.MX.sym("vx", self.obstacle_num)  # velocity in x direction
        ypred_list= cs.MX.sym("vy", self.obstacle_num) # velocity in y direction

        for  idx, (r) in enumerate(self.radii):
            # build the i-th obstacle expression
            xpred_i    = xpred_list[idx]
            ypred_i    = ypred_list[idx]
            
            hi_expr = (x[0] - xpred_i)**2 + (x[1] - ypred_i)**2 - r**2

            hi_fun = cs.Function(f"h{idx+1}", [x, xpred_list, ypred_list], [hi_expr])
            funcs.append(hi_fun)

        return funcs
    
    

class ObstacleMotion:
    def __init__(self,
                 positions: list,
                 modes: list[str],
                 mode_params: list[dict]):
        """
        positions    : list of obstacle centers (for bookkeeping)
        modes        : list of length m, each in {"static","random","sinusoid","step_bounce","orbit"}
        mode_params  : list of length m, each a dict with keys needed by that mode:
                       - for "step_bounce": {"bounds":(xmin,xmax),"speed":v, "dir":±1}
                       - for "orbit":       {"omega":angular_speed, "center":(cx0,cy0)}
                       - for "sinusoid":    {"amp":A,"freq":f,"phase":ϕ}
                       - for "random":      {"sigma":σ}
        """
        assert len(positions) == len(modes) == len(mode_params)
        self.m           = len(positions)
        self.modes       = modes
        self.mode_params = mode_params

        # state for velocity models
        self.vx = np.zeros(self.m)
        self.vy = np.zeros(self.m)
        self.dt = SAMPLING_TIME
        self.t  = 0.0
        
        self.cx = np.array([p[0] for p in positions], dtype=float)
        self.cy = np.array([p[1] for p in positions], dtype=float)
        
        self._dispatch = {
                "static"  : self._step_static,
                "random"  : self._step_random_walk,
                "sinusoid": self._step_sinusoid,
                "step_bounce": self._step_bounce,
                "orbit"      : self._step_orbit,
            }
        
        self._init_state = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t":  self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }


    def step(self):
        """Advance one dt, update & return (vx, vy) arrays of shape (m,)."""
        for i, mode in enumerate(self.modes):
            vx_i, vy_i = self._dispatch[mode](i)
            self.vx[i], self.vy[i] = vx_i, vy_i

        self.cx += - self.vx * self.dt
        self.cy += - self.vy * self.dt

        self.t += self.dt
        return self.cx.copy(), self.cy.copy()

    def _step_static(self, i: int):
        """Obstacle i stays still."""
        return 0.0, 0.0

    def _step_random_walk(self, i: int):
        """Gaussian random‐walk for obstacle i."""
        sigma = self.mode_params[i].get("sigma", 0.1)
        vx_new = self.vx[i] + np.random.randn() * sigma
        vy_new = self.vy[i] + np.random.randn() * sigma
        return vx_new, vy_new

    def _step_sinusoid(self, i: int):
        """Sinusoidal motion for obstacle i."""
        mp   = self.mode_params[i]
        amp  = mp.get("amp", 1.0)
        freq = mp.get("freq", 0.5)
        phase= mp.get("phase", 0.0)
        phi    = 2 * np.pi * freq * self.t + phase
        vx_i = amp * np.sin(phi)
        vy_i = amp * np.cos(phi)
        return vx_i, vy_i 
    
    def _step_bounce(self, i):
        mp     = self.mode_params[i]
        xmin, xmax = mp["bounds"]         
        speed = mp["speed"]                
        # read+update your current direction in-place:
        dir   = mp.get("dir", -1)          
        next_x = self.cx[i] - dir*speed*self.dt
        if next_x < xmin or next_x > xmax:
            dir *= -1
        mp["dir"] = dir                    # save flipped direction
        return dir*speed, 0.0
    
    def _step_orbit(self, i):
        """
        Rotate around a center point at angular rate omega.
        Velocity v = ω × r, where r is the distance to the center.
        """
        mp     = self.mode_params[i]
        omega  = mp.get("omega", 1.0)            # rad/s
        cx0, cy0 = mp.get("center", (0.0, 0.0))  # pivot
        dx = self.cx[i] - cx0
        dy = self.cy[i] - cy0
        vx_i = -omega * dy
        vy_i =  omega * dx
        return vx_i, vy_i
    
    def current_positions(self):
        return list(zip(self.cx, self.cy))
    
    def predict_states(self, N: int):
        """
        Return (N, m) arrays of vx and vy, by simulating N steps ahead
        and then rolling everything back.
        """
        prediction = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t":  self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }

        # simulate N steps
        x_pred = np.zeros((N, self.m))
        y_pred = np.zeros((N, self.m))
        for k in range(N):
            x_k, y_k = self.step()
            x_pred[k, :] = x_k
            y_pred[k, :] = y_k
                            
        # restore from backup
        self.cx          = prediction["cx"]
        self.cy          = prediction["cy"]
        self.vx          = prediction["vx"]
        self.vy          = prediction["vy"]
        self.t           = prediction["t"]
        self.mode_params = prediction["mode_params"]

        return x_pred.flatten(), y_pred.flatten()
    
    def reset(self):
        """
        Restore intial state of the obstacles.
        """
        st = self._init_state
        self.cx          = st["cx"].copy()
        self.cy          = st["cy"].copy()
        self.vx          = st["vx"].copy()
        self.vy          = st["vy"].copy()
        self.t           = st["t"]
        self.mode_params = copy.deepcopy(st["mode_params"])
        return self.current_positions()
    
    
class NN:

    def __init__(self, layers_size, positions, radii):
        """
        layers_size = list of layer sizes, including input and output sizes, for example: [5, 7, 7, 1]
        hidden_layers = number of hidden layers
        """
        
        self.obst = Obstacles(positions, radii)
        
        self.layers_size = layers_size

        # list of weights and biases and activations
        self.weights = []  
        self.biases  = [] 
        self.activations = []


        self.build_network()
        self.np_random = np.random.default_rng(seed=SEED)

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
    
    def normalization_z(self, nn_input):
        
        """
        Normalizes based on opt trajct

        """
        
        mu_states = cs.DM([-0.97504422, -0.64636289, 0.05090653, 0.05091322])
        sigma_states = cs.DM([1.39463336, 1.18101312, 0.0922252, 0.09315792])

        mu_h = 26.0464150055885 #* np.ones(self.obst.h_obsfunc(nn_input[:4]).shape[0]) # 26.0464150055885
        sigma_h = 11.6279296875#* np.ones(self.obst.h_obsfunc(nn_input[:4]).shape[0])
        

        x_norm = (nn_input[:4] - mu_states) / sigma_states
        
        #
        h_norm = nn_input[4:]#(nn_input[4:] - mu_h) / (sigma_h)

        return cs.vertcat(x_norm, h_norm)

    
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

    def forward(self, input_nn):
        """
        memeber function to perform the forward pass
        """
        normalized_input_nn = self.normalization_z(input_nn)
        a = normalized_input_nn
        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            a = self.activations[i](z)
        return a

    def create_forward_function(self):
        """
        making casadi function for the forward pass (in other words making the NN function)
        """
        # the input the cl  ass kappa function needs to take in
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


    def numerical_forward(self):
        """
        memeber function to perform the forward pass
        """
        # a = x
        # for i in range(len(self.weights)):
        #     z = self.weights[i] @ a + self.biases[i]
        #     a = self.activations[i](z)
        # return a
        x = cs.MX.sym('x', NUM_STATES, 1)
        h_func_list = cs.MX.sym('h_func', self.obst.obstacle_num, 1)
      
        input = cs.vertcat(x, h_func_list)

        y = self.forward(input)

       

        return cs.Function('NN', [x, h_func_list, self.get_flat_parameters()],[y])
    

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
                
                bound_low = np.sqrt(6.0 / fan_in)
                bound_high = np.sqrt(6.0 / fan_out)
                W_val = self.np_random.uniform(low=-bound_low, high=bound_high, size=(fan_out, fan_in))
                
            else:
                
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

    def __init__(self):
        super().__init__()
        self.na = NUM_INPUTS
        self.ns = NUM_STATES
        self.umax = CONSTRAINTS_U
        self.umin = -CONSTRAINTS_U

        self.observation_space = Box(-CONSTRAINTS_X, CONSTRAINTS_X, (self.ns,), np.float64)
        self.action_space = Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = SAMPLING_TIME

        self.A = np.array([
            [1, 0, self.dt, 0], 
            [0, 1, 0, self.dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])


    def reset(self, seed, options):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-CONSTRAINTS_X, -CONSTRAINTS_X, 0, 0])
        # assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
        return self.x, {}

    def step(
        self, action):
        # x = self.x
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
    ns = NUM_STATES # num of states
    na = NUM_INPUTS # num of inputs

    def __init__(self, layers_list, horizon, positions, radii):
        """
        Initialize the MPC class with parameters.
        """
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = horizon
        # self.x0 = cs.MX.sym("x0")
        dt = SAMPLING_TIME

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
        
        # intilization of the Neural Network
        self.nn = NN(layers_list, positions, radii)
        self.m = self.nn.obst.obstacle_num  # number of obstacles
        self.xpred_list = cs.MX.sym("xpred_list", self.m)  # velocity in x direction
        self.ypred_list = cs.MX.sym("ypred_list", self.m) # velocity in y direction
        self.xpred_hor = cs.MX.sym("xpred_hor", self.m * self.horizon)
        self.ypred_hor = cs.MX.sym("ypred_hor", self.m * self.horizon)
        
        print(self.xpred_hor.shape)

        #weight on the slack variables
        # self.weight_cbf = cs.DM([1e8])
        self.weight_cbf = cs.DM([2e7])


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
        phi = h(x, self.xpred_list, self.ypred_list)  # initial phi is h(x)
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
        
        print(f"self.state_const_list shape: {self.state_const_list.shape}")

        return 
    
    
    def cbf_func(self):
        """
        Build a single casadi.Function “cbff” whose output is an m×1 vector,
        where m = number of obstacles.  Internally, we loop over each scalar
        h_i and call dcbf(...) on it, then vert‐cat all φ_i into one vector.
        """
        # 1) Grab the list of individual CasADi Functions [h1(x₂); …; hₘ(x₂)]:
        h_funcs = self.nn.obst.make_h_functions()  # length‐m Python list of casadi.Function

        # 2) For each obstacle‐function h_i, compute φ_i(x,u) via dcbf:
        phi_list = []
        
        nn_in_list = [h_i(self.x_sym, self.xpred_list, self.ypred_list) for h_i in h_funcs ]
        alpha_list = self.alpha_nn(cs.vertcat(self.x_sym, *nn_in_list)) 
        
        for idx, h_i in enumerate(h_funcs):
            # dcbf(h_i, x_sym, u_sym, dynamics_f, [alpha_fn]) returns a scalar MX
            
            alpha_fn = lambda y: alpha_list[idx] * y
            
            phi_i = self.dcbf(h_i, self.x_sym, self.u_sym, self.dynamics_f, [alpha_fn])
            phi_list.append(phi_i)

        # 3) Vertically stack all φ_i into a single (m×1) MX:
        phi_vec = cs.vertcat(*phi_list)

        # 4) Wrap into one casadi.Function.  Its inputs are (x, u, all‐NN‐weights‐and‐biases),
        #    and its single output is that m×1 vector “phi_vec.”
        return cs.Function(
            "cbff",
            [self.x_sym, self.u_sym, self.nn.get_flat_parameters(), self.xpred_list, self.ypred_list],
            [phi_vec],
            ["x", "u", "alpha_params", "vx_list", "vy_list"],
            ["phi_vec"],
            )


    def cbf_const(self):
        """
        Build the full CBF‐constraint vector over horizon × m obstacles, with slack.
        Each column k gives [φ₁(x_k,u_k); …; φₘ(x_k,u_k)] + [s₁ₖ; …; sₘₖ].
        Stacked vertically, this becomes an (m*horizon)×1 vector.
        """
        cbf_fn = self.cbf_func()      # the Function built above
        # m = len(self.obst.positions)  # number of obstacles
        cons = []

        for k in range(self.horizon):
                xk      = self.X_sym[:, k]        # 4×1
                uk      = self.U_sym[:, k]        # 2×1
                phi_k   = cbf_fn(xk, uk, self.nn.get_flat_parameters(), self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])  # m×1
                slack_k = self.S_sym[:, k]        # m×1
                cons.append(phi_k + slack_k)      # m×1

        # Stack all horizon‐columns into one big (m*horizon)×1 vector:
        self.cbf_const_list = cs.vertcat(*cons)
        return


    def cbf_const_noslack(self):
        """
        Same as cbf_const, but omit slack.  We simply stack every φ_k(x,u) over k=0..horizon-1.
        Result is an (m*horizon)×1 vector of pure CBF‐constraints.
        """
        cbf_fn = self.cbf_func()
        cons = []

        for k in range(self.horizon):
                xk    = self.X_sym[:, k]
                uk    = self.U_sym[:, k]
                phi_k = cbf_fn(xk, uk, self.nn.get_flat_parameters(), self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])  # m×1
                print(f"{self.xpred_hor[k*self.m].shape} and {self.ypred_hor[k*self.m].shape}")
                cons.append(phi_k)

        self.cbf_const_list_noslack = cs.vertcat(*cons)
        print(f"self.m shape :  {self.m}")
        print(f"cbf_const_list_noslack shape: {self.cbf_const_list_noslack.shape}")
        return
    
    def objective_method(self):

        """""
        stage cost calculation
        """
        # why doesnt work? --> idk made it in line
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q_sym @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R_sym @ self.U_sym[:, k] + 
            self.weight_cbf * self.S_sym[m,k])
            for m in range (self.m) for k in range(self.horizon)
        )
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
        """""
        solves the MPC according to V-value function setup
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
            "expand": True,
            "print_time": False,
            "bound_consistency":True,
            "calc_lam_x": True,
            "eval_errors_fatal": True,
            "error_on_fail": False,
            "calc_lam_p": True,
            "fatrop": {"max_iter": 500, "print_level": 0},
        }

        MPC_solver = cs.nlpsol("solver", "fatrop", nlp, opts)

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
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.b_sym, self.V_sym, self.P_diag, Q_sym_flat, R_sym_flat, 
                            self.nn.get_flat_parameters(), rand_noise, self.xpred_hor, self.ypred_hor),
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
        lagrange_mult_x_lb_sym = cs.MX.sym("lagrange_mult_x_lb_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + self.nn.obst.obstacle_num * (self.horizon))
        lagrange_mult_x_ub_sym = cs.MX.sym("lagrange_mult_x_ub_sym", self.ns * (self.horizon+1) + self.na * (self.horizon) + self.nn.obst.obstacle_num  * (self.horizon))
        lagrange_mult_g_sym = cs.MX.sym("lagrange_mult_g_sym", 1*self.ns*(self.horizon) + self.nn.obst.obstacle_num*self.horizon)

        # construct 
        X_lower_bound = -CONSTRAINTS_X * np.ones(self.ns * (self.horizon))#-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
        X_upper_bound = CONSTRAINTS_X * np.ones(self.ns * (self.horizon)) #1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))

        U_lower_bound = -np.ones(self.na * (self.horizon-1))
        U_upper_bound = np.ones(self.na * (self.horizon-1))  

        # X_con + U_con + S_con + Sx_con
        lbx = cs.vertcat(self.X_sym[:,0], cs.DM(X_lower_bound), self.U_sym[:,0], cs.DM(U_lower_bound), np.zeros(self.nn.obst.obstacle_num *self.horizon)) 
        ubx = cs.vertcat(self.X_sym[:,0], cs.DM(X_upper_bound), self.U_sym[:,0], cs.DM(U_upper_bound), np.inf*np.ones(self.nn.obst.obstacle_num *self.horizon))

        # construct lower bound here 
        lagrange1 = lagrange_mult_x_lb_sym.T @ (opt_solution - lbx) #positive @ negative
        lagrange2 = lagrange_mult_x_ub_sym.T @ (ubx - opt_solution)  # positive @ negative                                
        lagrange3 = lagrange_mult_g_sym.T @ cs.vertcat(self.state_const_list, self.cbf_const_list) # opposite signs

 
        theta_vector = cs.vertcat(self.P_diag, self.nn.get_flat_parameters())

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
        """
        Constructs QP solve for parameter updates
        """

        Hessian_sym = cs.MX.sym("Hessian", self.theta.shape[0]*self.theta.shape[0])
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

        def __init__(self, params_innit, seed, alpha, gamma, decay_rate, layers_list, noise_scalingfactor, 
                     noise_variance, patience_threshold, lr_decay_factor, horizon, positions, radii, modes, mode_params):
            self.seed = seed

            # enviroment class
            self.env = env()

            # mpc class
            self.mpc = MPC(layers_list, horizon, positions, radii)
            self.obst_motion = ObstacleMotion(positions, modes, mode_params)
            # self.x0, _ = self.env.reset(seed=seed, options={})
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit

            #learning rate
            self.alpha = alpha
            
            # bounds
            #state bounded between CONSTRAINTS_X and -CONSTRAINTS_X
            self.X_lower_bound = -CONSTRAINTS_X * np.ones(self.ns * (self.horizon)) #-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
            self.X_upper_bound = CONSTRAINTS_X * np.ones(self.ns * (self.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))

            #state bound between 0 and 0, to make sure Ax +Bu = 0
            self.state_const_lbg = np.zeros(1*self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1*self.ns  * (self.horizon))

            #cbf constraint bounded between -inf and zero --> it is inverted
            # to make sure the constraints stay the same for all of them
            self.cbf_const_lbg = -np.inf * np.ones(self.mpc.nn.obst.obstacle_num*(self.horizon))
            self.cbf_const_ubg = np.zeros(self.mpc.nn.obst.obstacle_num*(self.horizon))
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
            
            #ADAM
            theta_vector_num = cs.vertcat(cs.diag(self.params_innit["P"]), self.params_innit["nn_params"])
            self.exp_avg = np.zeros(theta_vector_num.shape[0])
            self.exp_avg_sq = np.zeros(theta_vector_num.shape[0])
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

        def plot_B_update(self, B_update_history, experiment_folder):
            B_update = np.asarray(B_update_history)
            B_update = B_update.squeeze(-1)
 
            

            # build labels for first five
            labels = [f'P[{i},{i}]' for i in range(4)]
            print(f"labels: {labels}")

            nn_B_update = B_update[:, 4:]
            # take mean across rows (7,205) --> (7,)
            mean_mag = np.mean(np.abs(nn_B_update), axis=1)


            fig_p = plt.figure()
            for idx, lbl in enumerate(labels):
                plt.plot(B_update[:, idx], "o-", label=lbl)
            plt.xlabel('Update iteration')
            plt.ylabel('B_update')
            plt.title('P parameter B_update over training')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(fig_p, 'P_B_update_over_time')], experiment_folder)
            plt.close(fig_p)

            # 2. Plot just the NN mean
            fig_nn = plt.figure()
            plt.plot(mean_mag, "o-", label='mean |NN_B_update|')
            plt.xlabel('Update iteration')
            plt.ylabel('Mean absolute B_update')
            plt.title('NN mean acoss NN params B_update magnitude over training')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(fig_nn, 'NN_mean_B_update_over_time')], experiment_folder)
            plt.close(fig_nn)
        
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

        def update_learning_rate(self, current_stage_cost, params):
            """
            Update the learning rate based on the current stage cost metric.
            """

            if current_stage_cost < self.best_stage_cost:
                best_params = params.copy() 
                self.best_stage_cost = current_stage_cost
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience_threshold:
                old_alpha = self.alpha
                self.alpha *= self.lr_decay_factor  # decay 
                print(f"Learning rate decreased from {old_alpha} to {self.alpha} due to no stage cost improvement.")
                self.current_patience = 0  # reset 
                params = best_params  # revert to best params

            return params

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

        def V_MPC(self, params, x, xpred_list, ypred_list):
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat,  params["nn_params"], xpred_list, ypred_list),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract solution from solver 
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]


            return u_opt, solution["f"]
        
        def V_MPC_rand(self, params, x, rand, xpred_list, ypred_list):
            """
            same function as V_MPC but now with randomness
            """
            
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])
            

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)


            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["nn_params"], rand, xpred_list, ypred_list),
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]

            return u_opt

        def Q_MPC(self, params, action, x, xpred_list, ypred_list):

            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon-1))
            U_upper_bound = np.ones(self.na * (self.horizon-1))

            #here since its Q value we also give action
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, np.asarray(action).flatten(), U_lower_bound,  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound, np.asarray(action).flatten(), U_upper_bound, np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["nn_params"], xpred_list, ypred_list),
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
            
        def stage_cost(self, action, x, S):
            """Computes the stage cost :math:`L(s,a)`.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])

            state = x

            # print(f"S: {S}")


            # hx = (state[0]-(self.pos[0]))**2 + (state[1]-(self.pos[1]))**2 - self.r**2 

            # x_next  = self.params_innit["A"] @ x + self.params_innit["B"] @ action

            # hx_next = (x_next[0]-(self.pos[0]))**2 + (x_next[1]-(self.pos[1]))**2 - self.r**2


            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action #+ 1e3 * S
            )
        
        def evaluation_step(self, S, params, experiment_folder, episode_duration):
                
                state, _ = self.env.reset(seed=self.seed, options={})
                self.obst_motion.reset()
                
                states_eval = []
                actions_eval = []
                stage_cost_eval = []
                
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

                for i in range(episode_duration):
                    action, _ = self.V_MPC(params=params, x=state, xpred_list=xpred_list, ypred_list=ypred_list)

                    action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                    state, _, done, _, _ = self.env.step(action)
                    states_eval.append(state)
                    actions_eval.append(action)

                    stage_cost_eval.append(self.stage_cost(action, state, S))
                    xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

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
                for (cx, cy), r in zip(self.mpc.nn.obst.positions, self.mpc.nn.obst.radii):
                            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                            plt.gca().add_patch(circle)
                plt.gca().add_patch(circle)
                plt.xlim([-CONSTRAINTS_X, 0])
                plt.ylim([-CONSTRAINTS_X, 0])

                # Set labels and title
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                plt.title("Trajectories")
                plt.legend()
                plt.axis("equal")
                plt.grid()

                figactions=plt.figure()
                plt.plot(actions_eval[:, 0], "o-", label="Action 1")
                plt.plot(actions_eval[:, 1], "o-", label="Action 2")
                plt.xlabel("Iteration $k$")
                plt.ylabel("Action")
                plt.title("Actions")
                plt.legend()
                plt.grid()
                plt.tight_layout()


                figstagecost=plt.figure()
                plt.plot(stage_cost_eval, "o-")
                plt.xlabel("Iteration $k$")
                plt.ylabel("Cost")
                plt.title("Stage Cost")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                
                figsvelocity=plt.figure()
                plt.plot(states_eval[:, 2], "o-", label="Velocity x")
                plt.plot(states_eval[:, 3], "o-", label="Velocity y")    
                plt.xlabel("Iteration $k$")
                plt.ylabel("Velocity Value")
                plt.title("Velocity Plot")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                
                sum_stage_cost = np.sum(stage_cost_eval)
                print(f"Stage Cost: {sum_stage_cost}")
                
                figs = [
                                (figstates, f"states_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.png"),
                                (figactions, f"actions_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.png"),
                                (figstagecost, f"stagecost_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.png"),
                                (figsvelocity, f"velocity_MPCeval_{self.eval_count}_S_{sum_stage_cost}.png")
                ]


                self.save_figures(figs, experiment_folder, "Evaluation")
                plt.close("all")

                self.update_learning_rate(sum_stage_cost, params)

                self.eval_count += 1

                return 
        
        def parameter_updates(self, params, B_update_avg):

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            theta_vector_num = cs.vertcat(P_diag, params["nn_params"])


            identity = np.eye(theta_vector_num.shape[0])

            # print(f"before updates : {theta_vector_num}")

            # alpha_vec is resposible for the updates
            alpha_vec = cs.vertcat(self.alpha*np.ones(3), self.alpha, self.alpha, self.alpha*np.ones(theta_vector_num.shape[0]-5)*1e-2)
            # alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-2), self.alpha,self.alpha*1e-5)
            
            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 

            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, dtheta)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.01*np.abs(theta_vector_num[5:]))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.01*np.abs(theta_vector_num[5:]))

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.0001*np.ones(theta_vector_num.shape[0]-5))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.0001*np.ones(theta_vector_num.shape[0]-5))
            
            # constrained update qp update
            solution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, identity.flatten(), dtheta),
                    lbg=cs.vertcat(np.zeros(4), -np.inf*np.ones(theta_vector_num.shape[0]-4)),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
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
            params["nn_params"] = theta_vector_num[P_diag_shape:]       

            return params
        

        def rl_trainingloop(self, episode_duration, num_episodes, replay_buffer, episode_updatefreq, experiment_folder):
    
            #to store for plotting
            params_history_P = [self.params_innit["P"]]

            #for the for loop
            params = self.params_innit
            
            x, _ = self.env.reset(seed=self.seed, options={})

            stage_cost_history = []
            sum_stage_cost_history = []
            TD_history = []
            TD_temp = []
            TD_episode = []
            B_update_history = []
            grad_temp = []

      
            B_update_buffer = deque(maxlen=replay_buffer)
            

            states = [(x)]

            actions = []
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

            #intialize
            k = 0
            self.error_happened = False
            self.eval_count = 1

            for i in range(1,episode_duration*num_episodes):  
                
                rand = self.noise_scalingfactor*self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u = self.V_MPC_rand(params=params, x=x, rand = rand, xpred_list=xpred_list, ypred_list=ypred_list)
                u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)

                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    self.error_happened = True

     
                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _ = self.Q_MPC(params=params, action=u, x=x, xpred_list=xpred_list, ypred_list=ypred_list)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    self.error_happened = True


                S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1): :self.na * (self.horizon) + self.ns * (self.horizon+1) +self.horizon]
                stage_cost = self.stage_cost(action=u,x=x, S=S)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)

                #calculate V value

                # print(f"x_2: {x}")
                # print(f"params_3: {params}")

                _, Vcost = self.V_MPC(params=params, x=x, xpred_list=xpred_list, ypred_list=ypred_list)

                statsv = self.solver_inst.stats()
                if statsv["success"] == False:
                    print("V_MPC NOT SUCCEEDED")
                    self.error_happened = True

                # TD update
                TD = (stage_cost) + self.gamma*Vcost - Qcost

                U = solution[self.ns * (self.horizon+1):self.na * (self.horizon) + self.ns * (self.horizon+1)] 
                X = solution[:self.ns * (self.horizon+1)] 
            
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
                    X=X, U=U, S=S, inputs_NN=params["nn_params"],
                    xpred_hor=xpred_list, ypred_hor=ypred_list
                )['qlagrange_sens']

                # first order update
                B_update = -TD*qlagrange_numeric_jacob
                grad_temp.append(qlagrange_numeric_jacob)
                B_update_buffer.append(B_update)
                        
                stage_cost_history.append(stage_cost)
                if self.error_happened == False:
                    TD_episode.append(TD)
                    TD_temp.append(TD)
                else:
                    TD_temp.append(cs.DM(np.nan))
                    self.error_happened = False
                    
                _ = self.obst_motion.step()
        
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)

                if (k == episode_duration):                     
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:
                        self.evaluation_step(S=S, params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
                        print (f"updatedddddd")
                        B_update_avg = np.mean(B_update_buffer, 0)
                        B_update_history.append(B_update_avg)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.mean(TD_episode))

                    stage_cost_history = []
                    TD_episode = []

                    # reset the environment and the obstacle motion
                    x, _ = self.env.reset(seed=self.seed, options={})
                    self.obst_motion.reset()
                    k=0

                    print("reset")


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 20) == 0:
                        states = np.array(states)
                        actions = np.asarray(actions)
                        TD_temp = np.asarray(TD_temp) 

                        figstate=plt.figure()
                        plt.plot(
                            states[:, 0], states[:, 1],
                            "o-"
                        )

                        # Plot the obstacle
                        for (cx, cy), r in zip(self.mpc.nn.obst.positions, self.mpc.nn.obst.radii):
                            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                            plt.gca().add_patch(circle)
                        plt.xlim([-CONSTRAINTS_X, 0])
                        plt.ylim([-CONSTRAINTS_X, 0])

                        # Set labels and title
                        plt.xlabel("$x$")
                        plt.ylabel("$y$")
                        plt.title("Trajectories of states while policy is trained$")
                        plt.legend()
                        plt.axis("equal")
                        plt.grid()


                        figvelocity=plt.figure()
                        plt.plot(states[:, 2], "o-", label="Velocity x")
                        plt.plot(states[:, 3], "o-", label="Velocity y")    
                        plt.xlabel("Iteration $k$")
                        plt.ylabel("Velocity Value")
                        plt.title("Velocity Plot")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        # plt.show()

                        # Plot TD
                        indices = np.arange(len(TD_temp))
                        figtdtemp = plt.figure(figsize=(10, 5))
                        plt.scatter(indices,TD_temp, label='TD')
                        plt.yscale('log')
                        plt.title("TD Over Training (Log Scale) - Colored by Proximity")
                        plt.xlabel("Iteration $k$")
                        plt.ylabel("TD")
                        plt.legend()
                        plt.grid(True)

                        figactions=plt.figure()
                        plt.plot(actions[:, 0], "o-", label="Action 1")
                        plt.plot(actions[:, 1], "o-", label="Action 2")
                        plt.xlabel("Iteration $k$")
                        plt.ylabel("Action")
                        plt.title("Actions")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()

                        gradst = np.asarray(grad_temp)
                        gradst = gradst.squeeze(-1)


                        labels = [f'P[{i},{i}]' for i in range(4)]
                        nn_grads = gradst[:, 4:]
                        # take mean across rows (7,205) --> (7,)
                        mean_mag = np.mean(np.abs(nn_grads), axis=1)

                        P_figgrad = plt.figure()
                        
                        for idx, lbl in enumerate(labels):
                                plt.plot(gradst[:, idx], "o-", label=lbl)
                        plt.xlabel('Iteration $k$')
                        plt.ylabel('P gradient')
                        plt.title('P parameter gradients over training')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()



                        NN_figgrad = plt.figure()
                        plt.plot(mean_mag, "o-", label='mean |NN grad|')

                        plt.xlabel('Iteration $k$')
                        plt.ylabel('NN Mean absolute gradient')
                        plt.title('NN mean acoss NN params gradient magnitude over training')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        # plt.show()

                        figures_training = [
                            (figstate, f"position_plotat_{i}"),
                            (figvelocity, f"velocity_plotat_{i}"),
                            (figtdtemp, f"TD_plotat_{i}"),
                            (figactions, f"action_plotat_{i}"),
                            (P_figgrad, f"P_grad_plotat_{i}"),
                            (NN_figgrad, f"NN_grad_plotat_{i}"),
                            ]
                        self.save_figures(figures_training, experiment_folder, "Learning")
                        plt.close("all")

            
                    states = [(x)]
                    TD_temp = []
                    actions = []
                    grad_temp = []
                # k counter    
                k+=1
                
                #counter
                if i % 1000 == 0:
                    print(f"{i}/{episode_duration*num_episodes}")  

            #show trajectories
            # plt.show()
            # plt.close()

            params_history_P = np.asarray(params_history_P)

            TD_history = np.asarray(TD_history)
            sum_stage_cost_history = np.asarray(sum_stage_cost_history)

            self.plot_B_update(B_update_history, experiment_folder)
       

            figP = plt.figure(figsize=(10, 5))
            plt.plot(params_history_P[:, 0, 0], label=r"$P[1,1]$")
            plt.plot(params_history_P[:, 1, 1], label=r"$P[2,2]$")
            plt.plot(params_history_P[:, 2, 2], label=r"$P[3,3]$")
            plt.plot(params_history_P[:, 3, 3], label=r"$P[4,4]$")
            # plt.title("Parameter: P",        fontsize=24)
            plt.xlabel("Update Number",     fontsize=20)
            plt.ylabel("Value",             fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid()
            plt.tight_layout()

            figstagecost = plt.figure()
            plt.plot(sum_stage_cost_history, 'o', label="Stage Cost")
            plt.yscale('log')
            # plt.title("Stage Cost Over Training (Log Scale)", fontsize=24)
            plt.xlabel("Episode Number",                    fontsize=20)
            plt.ylabel("Stage Cost",                        fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            

            figtd = plt.figure()
            plt.plot(TD_history, 'o', label="TD")
            plt.yscale('log')
            plt.title("TD Over Training (Log Scale)")
            plt.xlabel("Episode Number")
            plt.ylabel("TD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            cost = np.array(sum_stage_cost_history)
            episodes = np.arange(len(cost))

            # choose window size
            window = 100

            # wrap in a pandas Series so rolling().mean()/.std() produce full‐length outputs
            s = pd.Series(cost)
            running_mean = s.rolling(window, center=True, min_periods=1).mean().values
            running_std  = s.rolling(window, center=True, min_periods=1).std().values

            # plt
            figstagecost_nice = plt.figure(figsize=(10,5))
            ax = figstagecost_nice.add_subplot(1,1,1)

            # running mean
            ax.plot(episodes, running_mean, '-', linewidth=2, label=f"Stage Cost mean ({window}-ep)")

            # ±1σ band
            ax.fill_between(episodes,
                            running_mean - running_std,
                            running_mean + running_std,
                            alpha=0.3,
                            label=f"Stage Cost std ({window}-ep)")

            if np.any(cost > 0):
                ax.set_yscale('log')

            ax.set_xlabel("Episode Number", fontsize=20)
            ax.set_ylabel("Stage Cost",     fontsize=20)
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.legend(fontsize=16)
            figstagecost_nice.tight_layout()


            figures_to_save = [
                (figP, "P"),
                (figstagecost, "stagecost"),
                (figstagecost_nice, "stagecost_smoothed"),
                (figtd, "TD")

            ]
            self.save_figures(figures_to_save, experiment_folder)
            plt.close("all")
            
            return params
        
        