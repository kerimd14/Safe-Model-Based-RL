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
import matplotlib.animation as animation

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
        
        x_pred[0, :] = self.cx
        y_pred[0, :] = self.cy
        
        for k in range(1,N):
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
        self.radii = radii
        self.positions = positions


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
        Normalizes based on maximum and minimum values of the states and h(x) values.

        """
  
        # x_norm = (nn_input[:4] - mu_states) / sigma_states
        
        x_min  = cs.DM([-CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0]]) # minimum values of the states
        x_max = cs.DM([0, 0, 0, 0]) # maximum values of the states
        
        x_norm = (nn_input[:4]-x_min)/(x_max-x_min) # normalize the states based on the maximum values
        
        h_max_list = []
        for (px, py), r in zip(self.positions, self.radii):
            dx = x_min[0] - px
            dy = x_min[1] - py
            h_max_i = dx**2 + dy**2 - r**2
            h_max_list.append(h_max_i)
        
        h_raw = nn_input[4:]             
        h_norm_list = []
        
        h_raw = cs.reshape(h_raw, -1, 1)  # reshape to (m, 1) where m is the number of obstacles

        h_raw_split = cs.vertsplit(h_raw)  # returns a list of (1x1) MX elements

        for i, h_i in enumerate(h_raw_split):
            h_max_i = h_max_list[i]
            h_norm_i = h_i / h_max_i
            h_norm_i = cs.fmin(cs.fmax(h_norm_i, 0), 1)
            h_norm_list.append(h_norm_i)
            
        h_norm = cs.vertcat(*h_norm_list)
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
                W_val = 0.1*self.np_random.uniform(low=-bound_low, high=bound_high, size=(fan_out, fan_in))
                
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

        self.observation_space = Box(-CONSTRAINTS_X[0], CONSTRAINTS_X[1], (self.ns,), np.float64)
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
        self.x = np.asarray([-CONSTRAINTS_X[0], -CONSTRAINTS_X[1], 0, 0])
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
        self.weight_cbf = cs.DM([5e5])


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

        """""
        used to construct state constraints 
        """

        state_const_list = []

        for k in range(self.horizon):

            state_const_list.append( self.X_sym[:,k+1] - ( self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym ) )

        self.state_const_list = cs.vertcat( *state_const_list )
        
        print(f"self.state_const_list shape: {self.state_const_list.shape}")

        return 
    
    

    def const(self):
        """
        Build the full slack‐augmented CBF constraint vector
        of size (m*horizon)×1, by inlining the DCBF formula
        with a Python loop — no tmp_substitute, no symbolic k.
        """
        cons = []
        h_funcs = self.nn.obst.make_h_functions()

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            phi_k_list = []
            for i, h_i in enumerate(h_funcs):    # MX-scalar

                h_x     = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])                                  # h(x_{k+1})

                phi_i = h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k)         # m×1    # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list = cs.vertcat(*cons)
   
    def const_noslack(self):
        """
        Build the full slack‐augmented CBF constraint vector
        of size (m*horizon)×1, by inlining the DCBF formula
        with a Python loop — no tmp_substitute, no symbolic k.
        """
        cons = []
        
        h_funcs = self.nn.obst.make_h_functions()

        for k in range(self.horizon):
            xk = self.X_sym[:, k]      # 4×1
            phi_k_list = []
            for i, h_i in enumerate(h_funcs):    # MX-scalar

                h_x = h_i(xk, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m])                                  # h(x_{k+1})
                print(f"hx {h_x}")
                phi_i = h_x
                phi_k_list.append(phi_i)

            phi_k = cs.vertcat(*phi_k_list)                # m×1
            cons.append(phi_k)         # m×1

        # final (m*horizon)×1 vector
        self.cbf_const_list_noslack = cs.vertcat(*cons)
    
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
        self.const_noslack()

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
                            R_sym_flat, self.xpred_hor, self.ypred_hor),
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
            "ipopt": {"max_iter": 500, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 500, "print_level": 0, "warm_start_init_point": True},
        }

        MPC_solver = cs.nlpsol("solver", "ipopt", nlp, opts)

        return MPC_solver
    
