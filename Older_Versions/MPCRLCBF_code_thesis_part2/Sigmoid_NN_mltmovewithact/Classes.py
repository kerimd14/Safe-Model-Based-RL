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
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED, CONSTRAINTS_U



class Obstacles:

    def __init__(self, positions, radii):

        """
        Initialize obstacles.

        Args:
            positions: List of (x, y) obstacle centers.
            radii:     List of obstacle radii.
        """
        if len(positions) != len(radii):
            raise ValueError(f"Expected same number of positions and radii, "
                            f"but got {len(positions)} positions and {len(radii)} radii.")
        self.positions = positions
        self.radii = radii
        self.obstacle_num = len(positions)
        self.dt = SAMPLING_TIME

    def h_obsfunc(self, x, xpred_list, ypred_list):
        """
        [NOT USED IN CODE]
        
        Numerically evaluate CBF h_i(x) = (x - x_pred[k])^2 + (y - y_pred[k])^2 - r^2
        for each obstacle over the dynamic constraints.
        
        Args:
            x:       Current state vector [x, y, …].
            xpred:   Flattened list of predicted obstacle x-positions.
            ypred:   Flattened list of predicted obstacle y-positions.

        Returns:
            A list of h_i(x) values (one per obstacle). 
        """
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        h_list = []
        for (r, x_pred, y_pred )in zip(self.radii, xpred_list, ypred_list):
            
            h_list.append((x[0] - x_pred)**2 + (x[1] - y_pred)**2 - r**2)
            
        return h_list

    def make_h_functions(self):
        """
        Build symbolic CasADi Functions for each obstacle h_i(x) i for 1,...,obstacle_num (not pythony index)

        Returns:
            A list of CasADi Functions for each obstacle over the dynamic obstacle constraints.
            Function Inputs: (x: MX[4x1], xpred_list: MX[num], ypred_list: MX[num])
            Funciton outputs: h_i : MX[1x1].
        """
        funcs = []
        
        #states
        x = cs.MX.sym("x", 4)
        
        #vx and vy are supposed represent the change in postion of obstacles, that is how i will show velocity
        xpred_list = cs.MX.sym("vx", self.obstacle_num)  # velocity in x direction
        ypred_list= cs.MX.sym("vy", self.obstacle_num) # velocity in y direction

        # Build one function per obstacle 
        for  idx, (r) in enumerate(self.radii):
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
        Responsible for taking care of the motion of the obstacles. 
        (Kind of equivalent to the env enviroment but for obstacles)
        
        Args:
            positions:    List of (x,y) initial centers for each obstacle.
            modes:        List of motion types, one per obstacle. Each must be one of
                          {"static","random","sinusoid","step_bounce","orbit"}.
            mode_params:  List of dicts (one per obstacle) specifying parameters:
                - step_bounce: {"bounds": (xmin,xmax), "speed": v, "dir": ±1}
                - orbit:       {"omega": omega, "center": (cx0,cy0)}
                - sinusoid:    {"amp": A, "freq": f, "phase": phi}
                - random:      {"sigma": sigma}
        """
        if not (len(positions) == len(modes) == len(mode_params)):
            raise ValueError(
                f"positions, modes, and mode_params must all have the same length, "
                f"but got {len(positions)}, {len(modes)}, and {len(mode_params)}."
            )
        self.m           = len(positions)
        self.modes       = modes
        #deepcopy to avoid modifying the original mode_params
        self.mode_params = copy.deepcopy(mode_params)

        #intializing states and parameters for velocity models
        self.vx = np.zeros(self.m)
        self.vy = np.zeros(self.m)
        
        self.cx = np.array([p[0] for p in positions], dtype=float)
        self.cy = np.array([p[1] for p in positions], dtype=float)
        
        self.dt = SAMPLING_TIME
        self.t  = 0.0
        
        # different types of movements that can be used
        self._step_functions = {
                "static"  : self._step_static,
                "random"  : self._step_random_walk,
                "sinusoid": self._step_sinusoid,
                "step_bounce": self._step_bounce,
                "orbit"      : self._step_orbit,
            }
        
        # intial state, will be used to restore the obstacles 
        # to their initial state when reset is called
        self._init_state = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t":  self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }


    def step(self):
        """
        Advance the obstacle motions by one time step.

        For each obstacle i, calls the appropriate stepping method
        (static, random_walk, sinusoid, bounce, or orbit), updates
        its velocity, then updates its position.
        
         Returns:
            Updated (cx, cy) positions of all obstacles for one time step.
        """
        for i, mode in enumerate(self.modes):
            vx_i, vy_i = self._step_functions[mode](i)
            self.vx[i], self.vy[i] = vx_i, vy_i

        self.cx += - self.vx * self.dt
        self.cy += - self.vy * self.dt

        self.t += self.dt
        return self.cx.copy(), self.cy.copy()

    def _step_static(self, i: int):
        """
        Obstacle i remains fixed in place.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i (so none here)
        """
        return 0.0, 0.0

    def _step_random_walk(self, i: int):
        """
        Gaussian random-walk for obstacle i.
        
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        sigma = self.mode_params[i].get("sigma", 0.1)
        vx_new = self.vx[i] + np.random.randn() * sigma
        vy_new = self.vy[i] + np.random.randn() * sigma
        return vx_new, vy_new

    def _step_sinusoid(self, i: int):
        """
        Sinusoidal motion for obstacle i.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        mp   = self.mode_params[i]
        amp  = mp.get("amp", 1.0)
        freq = mp.get("freq", 0.5)
        phase= mp.get("phase", 0.0)
        phi    = 2 * np.pi * freq * self.t + phase
        vx_i = amp * np.sin(phi)
        vy_i = amp * np.cos(phi)
        return vx_i, vy_i 
    
    def _step_bounce(self, i):
        """
        Step-bounce motion along the x-axis for obstacle i.

        - bounces between xmin and xmax
        - flips direction when hitting a bound
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
        """
        mp     = self.mode_params[i]
        xmin, xmax = mp["bounds"]         
        speed = mp["speed"]                
        # read+update current direction in-place:
        dir   = mp.get("dir", -1)          
        next_x = self.cx[i] - dir*speed*self.dt
        if next_x < xmin or next_x > xmax:
            dir *= -1
        mp["dir"] = dir                    # save flipped direction
        return dir*speed, 0.0
    
    def _step_orbit(self, i):
        """
        Rotate around a center point at angular rate omega.
        Velocity v = omega x r, where r is the distance to the center of rotation.
        
        Returns:
            Updated velocities (vx,vy) of obstacle i
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
        Simulate N steps ahead (without committing) and return the
        predicted trajectories for x and y positions.
        
        After predicting, restores the original obstacle state.
        
        Args: 
            N : Number of future steps to simulate. [integer]
        
        Returns:
            x_pred_flat : np.ndarray, shape ((N+1)*m,)
            y_pred_flat : np.ndarray, shape ((N+1)*m,)
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
        x_pred = np.zeros((N+1, self.m))
        y_pred = np.zeros((N+1, self.m))
        x_pred[0, :] = self.cx
        y_pred[0, :] = self.cy
        
        for k in range(1,N+1):
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
        
        Returns:
            Reset positions, velocities and time of all obstacles.
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

    def __init__(self, layers_size, positions, radii, mode_params, modes):
        """
        layers_size = list of layer sizes, including input and output sizes, for example: [5, 7, 7, 1]
        hidden_layers = number of hidden layers
        """
        
        self.obst = Obstacles(positions, radii)
        
        self.layers_size = layers_size
        self.ns = NUM_STATES
        # list of weights and biases and activations
        self.weights = []  
        self.biases  = [] 
        self.activations = []
        self.radii = radii
        self.positions = positions
        self.modes       = modes
        self.mode_params = mode_params


        self.build_network()
        self.np_random = np.random.default_rng(seed=SEED)
        
        # self.bounds_x = np.array([mode_params[i]["bounds"] for i in range(self.obst.obstacle_num)])  # shape (m, 2)
        # self.bounds_y = np.array([(py, py) for (_, py) in self.positions])
        bx, by = [], []
        for i in range(self.obst.obstacle_num):
            px, py = self.positions[i]
            mode   = self.modes[i]
            mp     = self.mode_params[i]

            if mode == "static":
                bx.append((px, px))
                by.append((py, py))

            elif mode == "step_bounce":
                xmin, xmax = mp["bounds"]
                bx.append((xmin, xmax))
                by.append((py, py))
                
        self.bounds_x = np.array(bx)  # shape (m, 2)
        self.bounds_y = np.array(by)  # shape (m, 2)
                
    def relu(self, x):
        """Standard ReLU: max(x, 0)."""
        return cs.fmax(x, 0)
    
    def tanh(self, x):
        """Hyperbolic tangent activation."""
        return cs.tanh(x)

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU to avoid dead neurons."""
        return cs.fmax(x, 0) + alpha * cs.fmin(x, 0)

    # def sigmoid(self, x):
    #     #sigmoid activation, used for the last output layer
    #     return 1 / (1 + cs.exp(-x))
    
    def shifted_sigmoid(self, x, epsilon=1e-6):
        """
        Sigmoid shifted into (epsilon, 1) since we cant have a 0
        """
        return epsilon + (1 - epsilon) * (1 / (1 + cs.exp(-x)))
    
    def normalization_z(self, nn_input):
        
        """
        Normalizes based on maximum and minimum values of the states and h(x) values.

        """
        
        
        # split the input into three to normalize each seprately
        x_raw    = nn_input[:self.ns]
        h_raw    = nn_input[self.ns:self.ns+self.obst.obstacle_num]
        pos_x    = nn_input[self.ns+self.obst.obstacle_num:self.ns+2*self.obst.obstacle_num]
        pos_y    = nn_input[self.ns+2*self.obst.obstacle_num:self.ns+3*self.obst.obstacle_num]
        u_raw  = nn_input[-(NUM_INPUTS):]
        
        
        def scale_centered(z, zmin, zmax, eps=1e-12):
            return 2 * (z - zmin) / (zmax - zmin + eps) - 1
  
        
        Xmax, Ymax = CONSTRAINTS_X[0], CONSTRAINTS_X[1]
        Vxmax, Vymax = CONSTRAINTS_X[2], CONSTRAINTS_X[3]  
        
        x_min = cs.DM([-Xmax, -Ymax, -Vxmax, -Vymax])
        x_max = cs.DM([   0.,    0.,  Vxmax,  Vymax ])
        
        # x_norm = (x_raw-x_min)/(x_max-x_min + 1e-9) # normalize the states based on the maximum values # normalize the states based on the maximum values
        x_norm = scale_centered(x_raw, x_min, x_max)
        
        corners = [
        (-Xmax, -Ymax), (-Xmax, 0.0),
        (0.0,   -Ymax), (0.0,   0.0)
        ]
        h_max_list = []
        
        
        for r, (xmin, xmax), (ymin, ymax) in zip(self.radii, self.bounds_x, self.bounds_y):
            obs_corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
            d2 = max((rx - ox)**2 + (ry - oy)**2
                        for (rx, ry) in corners
                        for (ox, oy) in obs_corners)
            h_max_list.append(d2 - r**2)
       
        h_norm_list = []
        
        h_raw = cs.reshape(h_raw, -1, 1)  # reshape to (m, 1) where m is the number of obstacles

        h_raw_split = cs.vertsplit(h_raw)  # returns a list of (1x1) MX elements

        for i, h_i in enumerate(h_raw_split):
            h_max_i = h_max_list[i]
            h_norm_i = scale_centered(h_i, 0.0, h_max_i)
            # h_norm_i = h_i / h_max_i
            # h_norm_i = cs.fmin(cs.fmax(h_norm_i, 0), 1)
            h_norm_list.append(h_norm_i)
            
        h_norm = cs.vertcat(*h_norm_list)
        
        # #position of object normalization
        # # bounds[i] == (xmin_i, xmax_i)
        # bounds_DM_x = cs.DM(self.bounds_x)  # shape (m,2)
        # pos_x_norm = [
        #     (pos_x[i] - bounds_DM_x[i,0]) / (bounds_DM_x[i,1] - bounds_DM_x[i,0])
        #     for i in range(self.obst.obstacle_num)
        # ]
        # pos_y_norm = [cs.DM(0.5) for _ in range(self.obst.obstacle_num)] #since it doesnt move we normalize to always be 0.5
        
        # pos_norm = cs.vertcat(*pos_x_norm, *pos_y_norm)
        
        eps = 1e-12
        bx = cs.DM(self.bounds_x)  # (m,2)
        by = cs.DM(self.bounds_y)  # (m,2)

        pos_x_norm = []
        pos_y_norm = []

        for i in range(self.obst.obstacle_num):
            mode_i = self.modes[i]

            if mode_i == "static":
                # stays put → constant mid-range
                # nx = cs.DM(0.5)
                # ny = cs.DM(0.5)
                nx = cs.DM(0)
                ny = cs.DM(0)

            elif mode_i == "step_bounce":
                # moves along one axis (you said: x moves, y fixed)
                # nx = (pos_x[i] - bx[i,0]) / (bx[i,1] - bx[i,0] + eps)
                # ny = cs.DM(0.5)
                nx = scale_centered(pos_x[i], bx[i,0], bx[i,1])
                ny = cs.DM(0)

            # else:
            #     # default: normalize both using the precomputed bounds (orbit/sinusoid/random etc.)
            #     nx = (pos_x[i] - bx[i,0]) / (bx[i,1] - bx[i,0] + eps)
            #     ny = (pos_y[i] - by[i,0]) / (by[i,1] - by[i,0] + eps)

            # # clip to [0,1] to be safe
            # nx = cs.fmin(cs.fmax(nx, 0), 1)
            # ny = cs.fmin(cs.fmax(ny, 0), 1)

            pos_x_norm.append(nx)
            pos_y_norm.append(ny)

        pos_norm = cs.vertcat(*pos_x_norm, *pos_y_norm)
        
        return cs.vertcat(x_norm, h_norm, pos_norm, u_raw)
    
    def _scale_to_spectral_radius(self, W: np.ndarray, target: float = 1.0, eps: float = 1e-12,
                              power_iters: int | None = None) -> np.ndarray:
        """
        Uniform -> rescale so max biggest absolute eigenvalue = target.
        """

        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals)).real

        if rho > 0.0:
            W = (float(target) / (rho + eps)) * W
        return W
    
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
        obs_pred_x = cs.MX.sym('obs_pred_x', self.obst.obstacle_num, 1)
        obs_pred_y = cs.MX.sym('obs_pred_y', self.obst.obstacle_num, 1)
        u = cs.MX.sym('u', NUM_INPUTS, 1)
      
        input = cs.vertcat(x, h_func_list, obs_pred_x, obs_pred_y, u)

        y = self.forward(input)

       

        return cs.Function('NN', [x, h_func_list, obs_pred_x, obs_pred_y, u, self.get_flat_parameters()],[y])
    

    def initialize_parameters(self):
        """
        initialization for the neural network (he normal for relu)
        """
        weight_values = []
        bias_values = []
        
        
        neg_slope = 0.01
        gain_leaky = np.sqrt(2.0 / (1.0 + neg_slope**2))  # around 1.414
        target_rho = getattr(self, "spectral_radius", 0.9)
        
        
        for i in range(len(self.layers_size) - 1):
            fan_in = self.layers_size[i] #5 input dim # 7 input dim #7 input dim
            fan_out = self.layers_size[i + 1] #7 output dim # 7 output dim # 1 output dim

            if i < len(self.layers_size) - 2:
                
                bound = np.sqrt(6.0 / fan_in) / gain_leaky
                
                # bound_low = np.sqrt(6.0 / fan_in)
                # bound_high = np.sqrt(6.0 / fan_out)
                W_val = self.np_random.uniform(low=-bound, high=bound, size=(fan_out, fan_in))
                
            else:
                
                bound = 0.1*np.sqrt(6.0 / (fan_in + fan_out))
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
            alpha_list = self.alpha_nn(cs.vertcat(xk, *nn_in_list, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m], uk))
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
            alpha_list = self.alpha_nn(cs.vertcat(xk, *nn_in_list, self.xpred_hor[k*self.m:(k+1)*self.m], self.ypred_hor[k*self.m:(k+1)*self.m], uk))
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
        cbf_cost = self.weight_cbf * sum(
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
            "ipopt": {"max_iter": 1000, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 1000, "print_level": 0, "warm_start_init_point": True},
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
            "ipopt": {"max_iter": 1000, "print_level": 0, "warm_start_init_point": "yes"},
            #"fatrop": {"max_iter": 1000, "print_level": 0, "warm_start_init_point": True},
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
            "ipopt": {"max_iter": 1000, "print_level": 0, "warm_start_init_point": "yes"},
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


####################### RL #######################


class RLclass:

        def __init__(
            self, 
            params_innit, 
            seed, 
            alpha, 
            gamma, 
            decay_rate, 
            layers_list, 
            noise_scalingfactor, 
            noise_variance, 
            patience_threshold, 
            lr_decay_factor, 
            horizon, 
            positions, 
            radii, 
            modes, 
            mode_params,
            slack_penalty_MPC,
            slack_penalty_RL,
            ):
            
            # Store random seed for reproducibility
            self.seed = seed

            # Create the environment
            self.env = env()
            
            # Penalty in RL stagecost on slacks
            self.slack_penalty_MPC = slack_penalty_MPC
            self.slack_penalty_RL = slack_penalty_RL

            # Initialize MPC and obstacle‐motion classes 
            self.mpc = MPC(layers_list, horizon, positions, radii, self.slack_penalty_MPC, mode_params, modes)
            self.obst_motion = ObstacleMotion(positions, modes, mode_params)
            
            # Parameters of experiments and states
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_innit = params_innit

            #learning rate
            self.alpha = alpha
            
            # bounds
            #state bounded between CONSTRAINTS_X and -CONSTRAINTS_X
            # self.X_lower_bound = -CONSTRAINTS_X * np.ones(self.ns * (self.horizon)) #-1e6 * CONSTRAINTS_X * np.ones(mpc.ns * (mpc.horizon))
            # self.X_upper_bound = CONSTRAINTS_X * np.ones(self.ns * (self.horizon))#1e6 * CONSTRAINTS_X * np.ones(mpc.ns  * (mpc.horizon))
            self.X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
            self.X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)
            
            
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
            self.qlagrange_fn_jacob = self.mpc.generate_symbolic_mpcq_lagrange()

            #decay_rate 
            self.decay_rate = decay_rate

            #randomness
            self.solver_inst_random =self.mpc.MPC_solver_rand()

            #qp solver
            self.qp_solver = self.mpc.qp_solver_fn()

            # learning update initialization
            self.best_params = params_innit.copy() 
            self.best_stage_cost = np.inf
            self.patience_threshold = patience_threshold
            self.lr_decay_factor = lr_decay_factor
            
            #ADAM
            # theta_vector_num = cs.vertcat(cs.diag(self.params_innit["P"]), self.params_innit["nn_params"])
            theta_vector_num = cs.vertcat(self.params_innit["nn_params"])
            self.exp_avg = np.zeros(theta_vector_num.shape[0])
            self.exp_avg_sq = np.zeros(theta_vector_num.shape[0])
            self.adam_iter = 1
            
            
            #warmstart variables
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  

            self.x_prev_QMPC        = cs.DM()  
            self.lam_x_prev_QMPC    = cs.DM()  
            self.lam_g_prev_QMPC    = cs.DM()  

            self.x_prev_VMPCrandom  = cs.DM()  
            self.lam_x_prev_VMPCrandom = cs.DM()  
            self.lam_g_prev_VMPCrandom = cs.DM()
            
            # used to do numerical forward to store alpha value
            self.fwd_func = self.mpc.nn.numerical_forward()
            
            #used to make h_func by using predicted list and input
            self.h_funcs = self.mpc.nn.obst.make_h_functions()
             
        
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
                    plt.close(fig)
                    print(f"Figure saved as: {file_path}")
            else:
                print("Figure not saved")
                
        def make_system_obstacle_animation(self,
            states: np.ndarray,
            obs_positions: np.ndarray,
            radii: list,
            constraints_x: float,
            out_path: str,
            ):
            """
            
            Args:
                states        : (T,4) array of system [x,y,vx,vy]
                obs_positions : (T, m, 2) array of obstacle centers
                radii         : list of length m
                constraints_x : scalar for plotting window
                out_path      : path to save the .gif
            """
            T, m = obs_positions.shape[:2]
            system_xy = states[:, :2]

            fig, ax = plt.subplots()
            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title("system + Moving Obstacles")

            # fixed zoom
            span = constraints_x
            ax.set_xlim(-1.1*span, +0.5*span)
            ax.set_ylim(-1.1*span, +0.5*span)

            # system artists
            line, = ax.plot([], [], "o-", lw=2, label="system path")
            dot,  = ax.plot([], [], "ro", ms=6,    label="system")

            # grab a real list of colors from the “tab10” qualitative map
            cmap   = plt.get_cmap("tab10")
            colors = cmap.colors  # this is a tuple-list of length 10

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # one circle per obstacle
            circles = []
            for i, r in enumerate(radii):
                c = plt.Circle(
                    (0, 0), r,
                    fill=False,
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f"obstacle {i+1}"
                )
                ax.add_patch(c)
                circles.append(c)

            ax.legend(loc="upper right")

            def init():
                line.set_data([], [])
                dot.set_data([], [])
                for c in circles:
                    c.center = (0, 0)
                return [line, dot] + circles

            def update(k):
                # system
                xk, yk = system_xy[k]
                line.set_data(system_xy[:k+1, 0], system_xy[:k+1, 1])
                dot.set_data([xk], [yk])
                # obstacles
                for i, c in enumerate(circles):
                    cx, cy = obs_positions[k, i]
                    c.center = (cx, cy)
                return [line, dot] + circles

            ani = animation.FuncAnimation(
                fig, update, frames=T, init_func=init,
                blit=True, interval=100
            )
            ani.save(out_path, writer="pillow", fps=3, dpi=90)
            plt.close(fig)

        def make_system_obstacle_animation_v2(
            self,
            states_eval: np.ndarray,      # (T,4) or (T,2)
            pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
            obs_positions: np.ndarray,    # (T, m, 2)
            radii: list,                  # (m,)
            constraints_x: float,         # used for static window
            out_path: str,                # output GIF path

            # display controls
            figsize=(6, 6),
            dpi=140,
            legend_outside=True,
            legend_loc="upper left",

            # zoom / camera
            camera="static",              # "static" or "follow"
            follow_width=4.0,             # view width around agent when following
            follow_height=4.0,

            # timing & colors
            trail_len: int | None = None, # if None → horizon length
            fps: int = 12,                # save speed (lower = slower)
            interval_ms: int = 150,       # live/preview speed (higher = slower)
            system_color: str = "C0",     # trail color
            pred_color: str = "orange",   # prediction color (line + markers)

            # output/interaction
            show: bool = False,           # open interactive window (zoom/pan)
            save_gif: bool = True,
            save_mp4: bool = False,
            mp4_path: str | None = None,
        ):
            """Animated plot of system, moving obstacles, trailing path, and predicted horizon."""

            # ---- harmonize lengths (avoid off-by-one) ----
            T_state = states_eval.shape[0]
            T_pred  = pred_paths.shape[0]
            T_obs   = obs_positions.shape[0]
            T = min(T_state, T_pred, T_obs)  # clamp to shortest
            system_xy    = states_eval[:T, :2]
            obs_positions = obs_positions[:T]
            pred_paths    = pred_paths[:T]

            # shapes
            Np1 = pred_paths.shape[1]
            N   = max(0, Np1 - 1)
            if trail_len is None:
                trail_len = N
            m = obs_positions.shape[1]

            # colors
            sys_rgb  = mcolors.to_rgb(system_color)
            pred_rgb = mcolors.to_rgb(pred_color)

            # ---- figure/axes ----
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.35)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(r"System + Moving Obstacles + Horizon")

            # initial static window (camera="follow" will override per-frame)
            span = constraints_x
            ax.set_xlim(-1.1*span, +0.5*span)
            ax.set_ylim(-1.1*span, +0.5*span)

            # ---- artists ----
            # trail: solid line + fading dots (dots exclude current)
            trail_ln, = ax.plot([], [], "-", lw=2, color=sys_rgb, zorder=2.0, label=fr"last {trail_len} steps")
            trail_pts  = ax.scatter([], [], s=26, zorder=2.1)

            # system dot (topmost)
            agent_pt,  = ax.plot([], [], "o", ms=7, color="red", zorder=5.0, label="system")

            # prediction: fading line (LineCollection) + markers (all orange)
            pred_lc = LineCollection([], linewidths=2, zorder=2.2)
            ax.add_collection(pred_lc)
            horizon_markers = [ax.plot([], [], "o", ms=5, color=pred_rgb, zorder=2.3)[0] for _ in range(N)]
            # proxy line so it appears in legend
            ax.plot([], [], "-", lw=2, color=pred_rgb, label="predicted horizon", zorder=2.2)

            # obstacles
            cmap   = plt.get_cmap("tab10")
            colors = cmap.colors
            circles = []
            for i, r in enumerate(radii):
                c = plt.Circle((0, 0), r, fill=False, color=colors[i % len(colors)], lw=2, label=f"obstacle {i+1}", zorder=1.0)
                ax.add_patch(c)
                circles.append(c)

            # legend placement
            if legend_outside:
                # leave room on the right
                fig.subplots_adjust(right=0.80)
                ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
            else:
                ax.legend(loc="upper right", framealpha=0.9)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # ---- helpers ----
            def _trail_window(k):
                start = max(0, k - trail_len)
                return start, k + 1

            def _set_follow_view(xc, yc):
                half_w = follow_width  / 2.0
                half_h = follow_height / 2.0
                ax.set_xlim(xc - half_w, xc + half_w)
                ax.set_ylim(yc - half_h, yc + half_h)

            # ---- init & update ----
            def init():
                trail_ln.set_data([], [])
                trail_pts.set_offsets(np.empty((0, 2)))
                agent_pt.set_data([], [])
                pred_lc.set_segments([])
                for mkr in horizon_markers:
                    mkr.set_data([], [])
                for c in circles:
                    c.center = (0, 0)
                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles]

            def update(k):
                xk, yk = system_xy[k]
                agent_pt.set_data([xk], [yk])

                if camera == "follow":
                    _set_follow_view(xk, yk)

                # trail: line + fading dots (exclude current)
                s, e = _trail_window(k)
                tail_xy = system_xy[s:e]
                trail_ln.set_data(tail_xy[:, 0], tail_xy[:, 1])

                pts_xy = tail_xy[:-1]
                if len(pts_xy) > 0:
                    trail_pts.set_offsets(pts_xy)
                    n = len(pts_xy)
                    alphas = np.linspace(0.3, 1.0, n)  # old→light, new→solid
                    cols = np.tile((*sys_rgb, 1.0), (n, 1))
                    cols[:, 3] = alphas
                    trail_pts.set_facecolors(cols)
                    trail_pts.set_edgecolors('none')
                else:
                    trail_pts.set_offsets(np.empty((0, 2)))

                # prediction: fading line + markers
                ph = pred_paths[k]                  # (N+1, 2)
                if N > 0:
                    future = ph[1:, :]              # (N, 2)
                    pred_poly = np.vstack((ph[0:1, :], future))  # include current for first segment
                    segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)  # (N, 2, 2)
                    pred_lc.set_segments(segs)

                    seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
                    seg_cols[:, 3] = np.linspace(1.0, 0.35, N)  # near→far fade
                    pred_lc.set_colors(seg_cols)

                    for j in range(N):
                        horizon_markers[j].set_data([future[j, 0]], [future[j, 1]])
                else:
                    pred_lc.set_segments([])
                    for mkr in horizon_markers:
                        mkr.set_data([], [])

                # obstacles
                for i, c in enumerate(circles):
                    cx, cy = obs_positions[k, i]
                    c.center = (cx, cy)

                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles]

            # blit=False if camera follows (limits change each frame)
            blit_flag = (camera != "follow")
            ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                        blit=blit_flag, interval=interval_ms)

            # ---- save / show ----
            if save_gif:
                ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
            if save_mp4:
                try:
                    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
                    ani.save(mp4_path or out_path.replace(".gif", ".mp4"), writer=writer, dpi=dpi)
                except Exception as e:
                    print("MP4 save failed. Install ffmpeg or add it to PATH. Error:", e)

            if show:
                plt.show()   # interactive zoom/pan
            else:
                plt.close(fig)
                
                
        def make_system_obstacle_animation_v3(self,
        states_eval: np.ndarray,      # (T,4) or (T,2)
        pred_paths: np.ndarray,       # (T, N+1, 2), ph[0] = current x_k
        obs_positions: np.ndarray,    # (T, m, 2)
        radii: list,                  # (m,)
        constraints_x: float,         # used for static window
        out_path: str,                # output GIF path

        # display controls
        figsize=(6.5, 6),
        dpi=140,
        legend_outside=True,
        legend_loc="upper left",

        # zoom / camera
        camera="static",              # "static" or "follow"
        follow_width=4.0,             # view width around agent when following
        follow_height=4.0,

        # timing & colors
        trail_len: int | None = None, # if None → horizon length
        fps: int = 12,                # save speed (lower = slower)
        interval_ms: int = 300,       # live/preview speed (higher = slower)
        system_color: str = "C0",     # trail color
        pred_color: str = "orange",   # prediction color (line + markers)

        # output/interaction
        show: bool = False,           # open interactive window (zoom/pan)
        save_gif: bool = True,
        save_mp4: bool = False,
        mp4_path: str | None = None,
        ):
            """Animated plot of system, moving obstacles, trailing path, and predicted horizon.
            Now also draws faded, dashed obstacle outlines at the next N predicted steps.
            """

            # ---- harmonize lengths (avoid off-by-one) ----
            T_state = states_eval.shape[0]
            T_pred  = pred_paths.shape[0]
            T_obs   = obs_positions.shape[0]
            T = min(T_state, T_pred, T_obs)  # clamp to shortest
            system_xy     = states_eval[:T, :2]
            obs_positions = obs_positions[:T]
            pred_paths    = pred_paths[:T]

            # shapes
            Np1 = pred_paths.shape[1]
            N   = max(0, Np1 - 1)
            if trail_len is None:
                trail_len = N
            m = obs_positions.shape[1]

            # colors
            sys_rgb  = mcolors.to_rgb(system_color)
            pred_rgb = mcolors.to_rgb(pred_color)

            # ---- figure/axes ----
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.35)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(r"System + Moving Obstacles + Horizon")

            # initial static window (camera="follow" will override per-frame)
            span = constraints_x
            ax.set_xlim(-1.1*span, +0.2*span)   # widened so circles aren’t clipped
            ax.set_ylim(-1.1*span, +0.2*span)

            # ---- artists ----
            # trail: solid line + fading dots (dots exclude current)
            trail_ln, = ax.plot([], [], "-", lw=2, color=sys_rgb, zorder=2.0, label=fr"last {trail_len} steps")
            trail_pts  = ax.scatter([], [], s=26, zorder=2.1)

            # system dot (topmost)
            agent_pt,  = ax.plot([], [], "o", ms=7, color="red", zorder=5.0, label="system")

            # prediction: fading line (LineCollection) + markers (all orange)
            pred_lc = LineCollection([], linewidths=2, zorder=2.2)
            ax.add_collection(pred_lc)
            horizon_markers = [ax.plot([], [], "o", ms=5, color=pred_rgb, zorder=2.3)[0] for _ in range(N)]
            # proxy line so it appears in legend
            ax.plot([], [], "-", lw=2, color=pred_rgb, label="predicted horizon", zorder=2.2)

            # obstacles (current time k)
            cmap   = plt.get_cmap("tab10")
            colors = cmap.colors
            circles = []
            for i, r in enumerate(radii):
                c = plt.Circle((0, 0), r, fill=False, color=colors[i % len(colors)],
                            lw=2, label=f"obstacle {i+1}", zorder=1.0)
                ax.add_patch(c)
                circles.append(c)

            # --- NEW: predicted obstacle outlines for the next N steps (ghosted) ---
            # one dashed circle per (future step h=1..N, obstacle i=1..m)
            pred_alpha_seq = np.linspace(0.35, 0.3, max(N, 1))  # nearer -> darker, farther -> lighter
            pred_circles_layers = []  # list of lists: [layer_h][i] -> patch
            for h in range(1, N+1):
                layer = []
                a = float(pred_alpha_seq[h-1])
                for i, r in enumerate(radii):
                    pc = plt.Circle((0, 0), r, fill=False,
                                    color=colors[i % len(colors)],
                                    lw=1.2, linestyle="--", alpha=a,
                                    zorder=0.8)  # behind current circles
                    ax.add_patch(pc)
                    layer.append(pc)
                pred_circles_layers.append(layer)
            if N > 0:
                # legend proxy for predicted obstacle outlines
                ax.plot([], [], linestyle="--", lw=1.2, color=colors[0],
                        alpha=0.3, label="obstacle (predicted)")

            # legend placement
            if legend_outside:
                fig.subplots_adjust(right=0.75)
                ax.legend(loc=legend_loc, bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0, framealpha=0.9)
            else:
                ax.legend(loc="upper right", framealpha=0.9)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # ---- helpers ----
            def _trail_window(k):
                start = max(0, k - trail_len)
                return start, k + 1

            def _set_follow_view(xc, yc):
                half_w = follow_width  / 2.0 + max(radii)
                half_h = follow_height / 2.0 + max(radii)
                ax.set_xlim(xc - half_w, xc + half_w)
                ax.set_ylim(yc - half_h, yc + half_h)

            # ---- init & update ----
            def init():
                trail_ln.set_data([], [])
                trail_pts.set_offsets(np.empty((0, 2)))
                agent_pt.set_data([], [])
                pred_lc.set_segments([])
                for mkr in horizon_markers:
                    mkr.set_data([], [])
                for c in circles:
                    c.center = (0, 0)
                for layer in pred_circles_layers:
                    for pc in layer:
                        pc.center = (0, 0)
                        pc.set_visible(False)
                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles,
                        *[pc for layer in pred_circles_layers for pc in layer]]

            def update(k):
                xk, yk = system_xy[k]
                agent_pt.set_data([xk], [yk])

                if camera == "follow":
                    _set_follow_view(xk, yk)

                # trail: line + fading dots (exclude current)
                s, e = _trail_window(k)
                tail_xy = system_xy[s:e]
                trail_ln.set_data(tail_xy[:, 0], tail_xy[:, 1])

                pts_xy = tail_xy[:-1]
                if len(pts_xy) > 0:
                    trail_pts.set_offsets(pts_xy)
                    n = len(pts_xy)
                    alphas = np.linspace(0.3, 1.0, n)  # old→light, new→solid
                    cols = np.tile((*sys_rgb, 1.0), (n, 1))
                    cols[:, 3] = alphas
                    trail_pts.set_facecolors(cols)
                    trail_pts.set_edgecolors('none')
                else:
                    trail_pts.set_offsets(np.empty((0, 2)))

                # prediction: fading line + markers
                ph = pred_paths[k]                  # (N+1, 2)
                if N > 0:
                    future = ph[1:, :]              # (N, 2)
                    pred_poly = np.vstack((ph[0:1, :], future))   # include current for first segment
                    segs = np.stack([pred_poly[:-1], pred_poly[1:]], axis=1)  # (N, 2, 2)
                    pred_lc.set_segments(segs)

                    seg_cols = np.tile((*pred_rgb, 1.0), (N, 1))
                    seg_cols[:, 3] = np.linspace(1.0, 0.35, N)  # near→far fade
                    pred_lc.set_colors(seg_cols)

                    for j in range(N):
                        horizon_markers[j].set_data([future[j, 0]], [future[j, 1]])
                else:
                    pred_lc.set_segments([])
                    for mkr in horizon_markers:
                        mkr.set_data([], [])

                # obstacles (current time k)
                for i, c in enumerate(circles):
                    cx, cy = obs_positions[k, i]
                    c.center = (cx, cy)

                # --- predicted obstacle outlines at k+1..k+N ---
                if N > 0:
                    for h, layer in enumerate(pred_circles_layers, start=1):
                        t = min(k + h, T - 1)  # clamp to last available pose
                        for i, pc in enumerate(layer):
                            cx, cy = obs_positions[t, i]
                            pc.center = (cx, cy)
                            pc.set_visible(True)

                return [trail_ln, trail_pts, agent_pt, *horizon_markers, *circles,
                        *[pc for layer in pred_circles_layers for pc in layer]]

            # blit=False if camera follows (limits change each frame)
            blit_flag = (camera != "follow")
            ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                                        blit=blit_flag, interval=interval_ms)

            # ---- save / show ----
            if save_gif:
                ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
            if save_mp4:
                try:
                    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
                    ani.save(mp4_path or out_path.replace(".gif", ".mp4"),
                            writer=writer, dpi=dpi)
                except Exception as e:
                    print("MP4 save failed. Install ffmpeg or add it to PATH. Error:", e)

            if show:
                plt.show()
            else:
                plt.close(fig)     

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

            denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
            
            dtheta = -step_size * (exp_avg / denom)
            return dtheta, exp_avg, exp_avg_sq

        def update_learning_rate(self, current_stage_cost, params):
            """
            Update the learning rate based on the current stage cost metric.
            """

            if current_stage_cost < self.best_stage_cost:
                self.best_params = params.copy() 
                self.best_stage_cost = current_stage_cost
                self.current_patience = 0
            else:
                self.current_patience += 1

            if self.current_patience >= self.patience_threshold:
                old_alpha = self.alpha
                self.alpha *= self.lr_decay_factor  # decay 
                print(f"Learning rate decreased from {old_alpha} to {self.alpha} due to no stage cost improvement.")
                self.current_patience = 0  # reset 
                params = self.best_params  # revert to best params

            return params

        def noise_scale_by_distance(self, x, y, max_radius=0.5):
            """
            Compute a scaling factor for exploration noise based on distance from the origin. 
            Close to the origin, noise is scaled down; at max_radius, it is 1.

            Args:
                x (float): current x positon of the system.
                y (float): current y position of the system.
                max_radius (float): distance beyond which noise scaling caps at 1.

            Returns:
                float: a factor in [0, 1] by which to multiply noise.
            """
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)

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
        
        def check_whh_spectral_radii(self, params_nn_list):
            """
            Check and print the spectral radii of the recurrent weight matrices in the NN.

            Args:
                params_nn_list (_type_): _description_
            """
            
            self.layers_list
            
            for i in range(len(self.layers_list)-1):
                Whh = params_nn_list[3*i + 2]  # recurrent weights
                eigvals = np.linalg.eigvals(Whh)
                rho = np.max(np.abs(eigvals)).real 
                print(f"Hidden layer {i+1} recurrent weight matrix spectral radius: {rho:.4f}")

        def V_MPC(self, params, x, xpred_list, ypred_list):
            """
            Solve the value-function MPC problem for the current state.

            Args:
                params (dict):  
                    Dictionary of system and NN parameters
                x (ns,):  
                    Current state of the system.
                xpred_list (m*(horizon+1),):  
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):  
                    Predicted obstacle y-positions over the horizon.
                hidden_in:  
                    Current hidden-state vectors for the NN layers.

            Returns:
                u_opt (na,):  
                    The first optimal control action.
                V_val (solution["f"]):  
                    The optimal value function V(x).
                hidden_t1 :  
                    Updated hidden states after one NN forward pass.
            """
            
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  
                                  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound, 
                                  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"],
                                                       P_diag, Q_flat, R_flat,  params["nn_params"], 
                                                       xpred_list, ypred_list),
                x0    = self.x_prev_VMPC,
                lam_x0 = self.lam_x_prev_VMPC,
                lam_g0 = self.lam_g_prev_VMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # warmstart variables for next iteration
            self.x_prev_VMPC     = solution["x"]
            self.lam_x_prev_VMPC = solution["lam_x"]
            self.lam_g_prev_VMPC = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the evaluation stage cost)
            self.S_VMPC = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]
            
            #check output  of NN
            alpha_list = []
            h_func_list = [h_func for h_func in self.mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
            alpha_list.append(cs.DM(self.fwd_func(x, h_func_list, xpred_list[:self.mpc.nn.obst.obstacle_num], 
                                                  ypred_list[:self.mpc.nn.obst.obstacle_num], u_opt, params["nn_params"])))

            return u_opt, solution["f"], alpha_list
        
        def V_MPC_rand(self, params, x, rand, xpred_list, ypred_list):
            """
            Solve the value-function MPC problem with injected randomness.

            This is identical to V_MPC, but includes a random noise term in the optimization
            to encourage exploration.

            Args:
                params (dict):
                    Dictionary of system and NN parameters:
                x (ns,):
                    Current system state vector.
                rand (na,1):
                    Random noise vector added to first control action in MPC objective
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current NN hidden-state from previous time step.

            Returns:
                u_opt (na,):
                    The first optimal control action (with randomness).
                hidden_t1 (list of MX):
                    Updated NN hidden-state 
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
                x0    = self.x_prev_VMPCrandom,
                lam_x0 = self.lam_x_prev_VMPCrandom,
                lam_g0 = self.lam_g_prev_VMPCrandom,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # warmstart variables for next iteration
            self.x_prev_VMPCrandom = solution["x"]
            self.lam_x_prev_VMPCrandom = solution["lam_x"]
            self.lam_g_prev_VMPCrandom = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the RL stage cost)
            self.S_VMPC_rand = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]
            
            #check output  of NN
            alpha_list = []
            h_func_list = [h_func for h_func in self.mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
            alpha_list.append(cs.DM(self.fwd_func(x, h_func_list, xpred_list[:self.mpc.nn.obst.obstacle_num], 
                                                  ypred_list[:self.mpc.nn.obst.obstacle_num], u_opt,  params["nn_params"])))

            return u_opt, alpha_list

        def Q_MPC(self, params, action, x, xpred_list, ypred_list):
            
            """"
            
            Solve the Q-value MPC problem for current state and current action.
            
            Similar to V_MPC, but includes the action in the optimization and computes the Q-value.
            
            Args:
                params (dict):
                    Dictionary of system and NN parameters.
                action (na,):
                    Current control action vector.
                x (ns,):
                    Current state of the system.
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current hidden-state vectors for the NN layers.
            Returns:
                x_opt (ns*(horizon+1),):
                    Optimal state trajectory over the horizon.
                Q_val (solution["f"]):
                    Optimal Q-value for the current state and action.
                lagrange_mult_g (solution["lam_g"]):
                    Lagrange multipliers for the constraints.
                lam_lbx (solution["lam_x"]):
                    Lagrange multipliers for the lower bounds on x.
                lam_ubx (solution["lam_x"]):
                    Lagrange multipliers for the upper bounds on x.
                lam_p (solution["lam_p"]):
                    Lagrange multipliers for the parameters.
                hidden_t1 (list of MX):
                    Updated hidden states after one NN forward pass.
            """

            # Build input‐action bounds (note horizon−1 controls remain free after plugging in `action`)
            U_lower_bound = -np.ones(self.na * (self.horizon-1))
            U_upper_bound = np.ones(self.na * (self.horizon-1))

            #Assemble full lbx/ubx: [ x0; X(1…H); action; remaining U; slack ]
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, 
                                  np.asarray(action).flatten(), U_lower_bound,  
                                  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound,
                                  np.asarray(action).flatten(), U_upper_bound, 
                                  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)
            
            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag,
                                                       Q_flat, R_flat, params["nn_params"],
                                                       xpred_list, ypred_list),
                x0    = self.x_prev_QMPC,
                lam_x0 = self.lam_x_prev_QMPC,
                lam_g0 = self.lam_g_prev_QMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            
            # Extract lagrange multipliers needed for the lagrangian:
            lagrange_mult_g = solution["lam_g"]
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]
            
            # warmstart variables for next iteration
            self.lam_g_prev_QMPC = solution["lam_g"]
            self.x_prev_QMPC = solution["x"]
            self.lam_x_prev_QMPC = solution["lam_x"]

            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p
            
        def stage_cost(self, action, state, S, hx):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            hx = np.array(hx)
            
            violations = np.clip(-hx, 0, None)
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action +self.slack_penalty_RL*(np.sum(S)/(self.horizon+self.mpc.nn.obst.obstacle_num)) #+ np.sum(4e5*violations)
            )
            
        def stage_cost_validation(self, action, state, hx):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            hx = np.array(hx)
            
            violations = np.clip(-hx, 0, None)
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + np.sum(3e5*violations)
            )
        
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
                # Reset environment & obstacle motion
                state, _ = self.env.reset(seed=self.seed, options={})
                self.obst_motion.reset()
                
                # Prepare storage for rollout data
                states_eval = [state]
                actions_eval = []
                stage_cost_eval = []
                stage_cost_validation = []
                
                #Pre-compute initial obstacle predictions and initial nn hidden state
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                obs_positions = [self.obst_motion.current_positions()]
                
                hx = [ float(hf(cs.DM(state), xpred_list[0:self.mpc.nn.obst.obstacle_num], ypred_list[0:self.mpc.nn.obst.obstacle_num])) 
                             for hf in self.h_funcs ] 
                hx_list = [hx]
                
                #for NN outputs
                alphas = []
            
                # Initialize warmstart variables
                self.x_prev_VMPC        = cs.DM()  
                self.lam_x_prev_VMPC    = cs.DM()  
                self.lam_g_prev_VMPC    = cs.DM()  

                for i in range(episode_duration):
                    action, _, alpha= self.V_MPC(params=params, x=state, 
                                                 xpred_list=xpred_list, 
                                                 ypred_list=ypred_list)
                    
                    statsv = self.solver_inst.stats()
                    if statsv["success"] == False:
                        print("V_MPC NOT SUCCEEDED in EVALUATION")
                    alphas.append(alpha)
                    
                    action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                    state, _, done, _, _ = self.env.step(action)
                    
                    states_eval.append(state)
                    actions_eval.append(action)
                    stage_cost_eval.append(self.stage_cost(action, state, self.S_VMPC, hx))
                    stage_cost_validation.append(self.stage_cost_validation(action, state, hx))
                    
                    hx = [ float(hf(cs.DM(state), xpred_list[0:self.mpc.nn.obst.obstacle_num], ypred_list[0:self.mpc.nn.obst.obstacle_num])) 
                             for hf in self.h_funcs ]
                    hx_list.append(hx) 
                    
                    
                    # Advance obstacle motion and re-predict
                    _ = self.obst_motion.step()
                    xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                    obs_positions.append(self.obst_motion.current_positions())

                states_eval = np.array(states_eval)
                actions_eval = np.array(actions_eval)
                stage_cost_eval = np.array(stage_cost_eval)
                stage_cost_eval = stage_cost_eval.reshape(-1) 
                obs_positions = np.array(obs_positions)
                hx_list = np.vstack(hx_list)
                alphas = np.array(alphas)
                alphas = np.squeeze(alphas)
                
                sum_stage_cost = np.sum(stage_cost_eval)
                print(f"Stage Cost: {sum_stage_cost}")

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
                plt.xlim([-CONSTRAINTS_X[0], 0])
                plt.ylim([-CONSTRAINTS_X[1], 0])

                # Set labels and title
                plt.xlabel("$x$")
                plt.ylabel("$y$")
                plt.title("Trajectories")
                plt.legend()
                plt.axis("equal")
                plt.grid()
                self.save_figures([(figstates,
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
                self.save_figures([(figactions,
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
                self.save_figures([(figstagecost,
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
                self.save_figures([(figsvelocity,
                f"velocity_MPCeval_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                experiment_folder, "Evaluation")
                
                for i in range(hx_list.shape[1]):
                    fig_hi = plt.figure()
                    plt.plot(hx_list[:,i], "o", label=rf"$h_{{{i+1}}}(x_k)$")
                    plt.xlabel(r"Iteration $k$")
                    plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
                    plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
                    plt.grid()
                    self.save_figures([(fig_hi,
                                f"hx_obstacle_{self.eval_count}_SC_{sum_stage_cost}.svg")],
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
                self.save_figures([(fig_alpha,
                            f"alpha_{self.eval_count}_SC_{sum_stage_cost}.svg")],
                            experiment_folder, "Evaluation")
                
                # T_pred = plans_eval.shape[0]
                target_folder = os.path.join(experiment_folder, "evaluation")
                out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost}.gif")
                self.make_system_obstacle_animation(
                states_eval,
                obs_positions,
                self.mpc.nn.obst.radii,
                CONSTRAINTS_X[0],
                out_gif,
                )

                self.update_learning_rate(sum_stage_cost, params)

                self.eval_count += 1
                
                # self.stage_cost_valid.append(np.sum(stage_cost_validation))
                self.stage_cost_valid.append(sum_stage_cost)

                return 
        
        def parameter_updates(self, params, B_update_avg):

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            # theta_vector_num = cs.vertcat(P_diag, params["nn_params"])
            
            theta_vector_num = cs.vertcat(params["nn_params"])

            # L  = self.cholesky_added_multiple_identity(A_update_avg)
            # A_update_chom = L @ L.T
            
            identity = np.eye(theta_vector_num.shape[0])


            # alpha_vec is resposible for the updates
            # alpha_vec = cs.vertcat(self.alpha*np.ones(3), self.alpha, self.alpha, self.alpha*np.ones(theta_vector_num.shape[0]-5)*1e-2)
            alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0])*1e-2)
            # alpha_vec = cs.vertcat(self.alpha*np.ones(4), self.alpha*np.ones(theta_vector_num.shape[0]-4)*1e-2)
            # alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-2), self.alpha,self.alpha*1e-5)
            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 

            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, B_update_avg)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.01*np.abs(theta_vector_num[5:]))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.01*np.abs(theta_vector_num[5:]))

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.0001*np.ones(theta_vector_num.shape[0]-5))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.0001*np.ones(theta_vector_num.shape[0]-5))
            
            # constrained update qp update
            solution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, dtheta),
                    # lbg = cs.vertcat(np.zeros(4), -np.inf*np.ones(theta_vector_num.shape[0]-4)),
                    # ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    lbg= cs.vertcat(-np.inf*np.ones(theta_vector_num.shape[0])),
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

            # P_diag_shape = self.ns*1
            # # #constructing the diagonal posdef P matrix 
            # P_posdef = cs.diag(theta_vector_num[:P_diag_shape])

            # params["P"] = P_posdef
            # params["nn_params"] = theta_vector_num[P_diag_shape:]       
            params["nn_params"] = theta_vector_num  

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
            obs_positions = [self.obst_motion.current_positions()]

            B_update_buffer = deque(maxlen=replay_buffer)

            states = [(x)]

            actions = []
            
            xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
            
            
            hx =  [ float(hf(cs.DM(x), xpred_list[0:self.mpc.nn.obst.obstacle_num], ypred_list[0:self.mpc.nn.obst.obstacle_num])) 
                             for hf in self.h_funcs ]
            
            hx_list = [hx]
            
            #for NN outputs
            alphas = []
            
            #intialize
            k = 0
            
            #used to catch errors --> if error happened we ignore and dont record
            self.error_happened = False
            self.eval_count = 1
            
            #warmstart variables
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  

            self.x_prev_QMPC        = cs.DM()  
            self.lam_x_prev_QMPC    = cs.DM()  
            self.lam_g_prev_QMPC    = cs.DM()  

            self.x_prev_VMPCrandom  = cs.DM()  
            self.lam_x_prev_VMPCrandom = cs.DM()  
            self.lam_g_prev_VMPCrandom = cs.DM()
            
            self.stage_cost_valid = []


            for i in range(1,episode_duration*num_episodes):  
                
                # if i == 21*50*150:
                #     self.alpha = self.alpha*0.01    
                 
                
                noise = self.noise_scalingfactor*self.noise_scale_by_distance(x[0],x[1])
                rand = noise * self.np_random.normal(loc=0, scale=self.noise_variance, size = (2,1))

                u, alpha = self.V_MPC_rand(params=params, x=x, rand = rand, xpred_list=xpred_list, ypred_list=ypred_list)
                u = cs.fmin(cs.fmax(cs.DM(u), -1), 1)
                
                alphas.append(alpha)
                actions.append(u)

                statsvrand = self.solver_inst_random.stats()
                if statsvrand["success"] == False:
                    print("V_MPC_RANDOM NOT SUCCEEDED")
                    self.error_happened = True

     
                solution, Qcost, lagrange_mult_g, lam_lbx, lam_ubx, _ = self.Q_MPC(params=params, 
                                                                                   action=u, 
                                                                                   x=x, 
                                                                                   xpred_list=xpred_list, 
                                                                                   ypred_list=ypred_list)
     

                statsq = self.solver_inst.stats()
                if statsq["success"] == False:
                    print("Q_MPC NOT SUCCEEDED")
                    self.error_happened = True


                S = solution[self.na * (self.horizon) + self.ns * (self.horizon+1):]
                stage_cost = self.stage_cost(action=u,state=x, S=self.S_VMPC_rand, hx = hx)
                
                # enviroment update step
                x, _, done, _, _ = self.env.step(u)

                # append trajectory points for plotting
                states.append(x)
                hx = [ float(hf(cs.DM(x), xpred_list[0:self.mpc.nn.obst.obstacle_num], 
                                ypred_list[0:self.mpc.nn.obst.obstacle_num])) 
                             for hf in self.h_funcs ]
                hx_list.append(hx)

                _, Vcost, _ = self.V_MPC(params=params, x=x, xpred_list=xpred_list, ypred_list=ypred_list)

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
                
                
                # outer_product = qlagrange_numeric_jacob @ qlagrange_numeric_jacob.T
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
                
                obs_positions.append(self.obst_motion.current_positions())
        
                xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)               

                if (k == episode_duration):                     
                    # -1 because loop starts from 1
                    if (i-1) % (episode_duration*episode_updatefreq) == 0:
                        
                        # self.evaluation_step(S=S, params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)
                        print (f"updatedddddd")
                        B_update_avg = np.mean(B_update_buffer, 0)
                        # buf = np.asarray(B_update_buffer)            # [N, d]
                        # batch = int(0.3*len(buf))          # pick number you want 

                        # # Uniform replay (simple & stable):
                        # idx = self.np_random.choice(len(buf), size=batch, replace=False)
                        # B_update_avg = buf[idx].mean(axis=0)
                        
                        B_update_history.append(B_update_avg)

                        params = self.parameter_updates(params = params, B_update_avg = B_update_avg)
                        
                        params_history_P.append(params["P"])

                        
                        self.noise_scalingfactor = self.noise_scalingfactor*(1-self.decay_rate)

                        print(f"noise scaling: {self.noise_scalingfactor}")


                    sum_stage_cost_history.append(np.sum(stage_cost_history))
                    TD_history.append(np.mean(TD_episode))

                    stage_cost_history = []
                    TD_episode = []


                    # plotting the trajectories under the noisy policies explored
                    current_episode = i // episode_duration
                    if (current_episode % 50) == 0:
                        states = np.array(states)
                        actions = np.asarray(actions)
                        TD_temp = np.asarray(TD_temp)
                        obs_positions = np.array(obs_positions)
                        hx_list = np.vstack(hx_list)
                        alphas = np.array(alphas)
                        alphas = np.squeeze(alphas)
 

                        figstate=plt.figure()
                        plt.plot(
                            states[:, 0], states[:, 1],
                            "o-"
                        )

                        # Plot the obstacle
                        for (cx, cy), r in zip(self.mpc.nn.obst.positions, self.mpc.nn.obst.radii):
                            circle = plt.Circle((cx, cy), r, color="k", fill=False, linewidth=2)
                            plt.gca().add_patch(circle)
                        plt.xlim([-CONSTRAINTS_X[0], 0])
                        plt.ylim([-CONSTRAINTS_X[1], 0])

                        # Set labels and title
                        plt.xlabel(r"$x$")
                        plt.ylabel(r"$y$")
                        plt.title(r"Trajectories of states while policy is trained$")
                        plt.legend()
                        plt.axis("equal")
                        plt.grid()
                        self.save_figures([(figstate,
                            f"position_plotat_{i}.svg")],
                            experiment_folder, "Learning")


                        figvelocity=plt.figure()
                        plt.plot(states[:, 2], "o-", label=r"Velocity x")
                        plt.plot(states[:, 3], "o-", label=r"Velocity y")    
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Velocity Value")
                        plt.title(r"Velocity Plot")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figvelocity,
                            f"figvelocity{i}.svg")],
                            experiment_folder, "Learning")

                        # Plot TD
                        indices = np.arange(len(TD_temp))
                        figtdtemp = plt.figure(figsize=(10, 5))
                        plt.scatter(indices,TD_temp, label=r"TD")
                        plt.yscale('log')
                        plt.title(r"TD Over Training (Log Scale) - Colored by Proximity")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"TD")
                        plt.legend()
                        plt.grid(True)
                        self.save_figures([(figtdtemp,
                            f"TD_plotat_{i}.svg")],
                            experiment_folder, "Learning")

                        figactions=plt.figure()
                        plt.plot(actions[:, 0], "o-", label=r"Action 1")
                        plt.plot(actions[:, 1], "o-", label=r"Action 2")
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"Action")
                        plt.title(r"Actions")
                        plt.legend()
                        plt.grid()
                        plt.tight_layout()
                        self.save_figures([(figactions,
                            f"action_plotat_{i}.svg")],
                            experiment_folder, "Learning")

                        gradst = np.asarray(grad_temp)
                        gradst = gradst.squeeze(-1)


                        labels = [f'P[{i},{i}]' for i in range(4)]
                        nn_grads = gradst[:, 4:]
                        # take mean across rows (7,205) --> (7,)
                        mean_mag = np.mean(np.abs(nn_grads), axis=1)

                        P_figgrad = plt.figure()
                        
                        for idx, lbl in enumerate(labels):
                                plt.plot(gradst[:, idx], "o-", label=lbl)
                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"P gradient")
                        plt.title(r"P parameter gradients over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(P_figgrad,
                            f"P_grad_plotat_{i}.svg")],
                            experiment_folder, "Learning")




                        NN_figgrad = plt.figure()
                        plt.plot(mean_mag, "o-", label=r"mean abs(nn grad)")

                        plt.xlabel(r"Iteration $k$")
                        plt.ylabel(r"nn Mean absolute gradient")
                        plt.title(r"nn mean acoss nn params gradient magnitude over training")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        self.save_figures([(NN_figgrad,
                            f"NN_grad_plotat_{i}.svg")],
                            experiment_folder, "Learning")
                        # plt.show()
                        
                        
                        for i in range(hx_list.shape[1]):
                            fig_hi = plt.figure()
                            plt.plot(hx_list[:,i], "o", label=rf"$h_{{{i+1}}}(x_k)$")
                            plt.xlabel(r"Iteration $k$")
                            plt.ylabel(rf"$h_{{{i+1}}}(x_k)$")
                            plt.title(rf"Obstacle {i+1}: $h_{{{i+1}}}(x_k)$ Over Time")
                            plt.grid()
                            self.save_figures([(fig_hi,
                                        f"hx_obstacle_plotat_{i}.svg")],
                                        experiment_folder, "Learning")
                            
                        # Alphas from nn
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
                        self.save_figures([(fig_alpha,
                                    f"alpha_plotat_{i}.svg")],
                                    experiment_folder, "Learning")

                        target_folder = os.path.join(experiment_folder, "learning_process")
                        out_gif = os.path.join(target_folder, f"system_and_obstacle_{self.eval_count}_SC_{sum_stage_cost_history[-1]}.gif")
                        self.make_system_obstacle_animation(
                        states,
                        obs_positions,
                        self.mpc.nn.obst.radii,
                        CONSTRAINTS_X[0],
                        out_gif,
                        )
                        
                        #evaluation step
                        self.evaluation_step(params=params, experiment_folder=experiment_folder, episode_duration=episode_duration)

                    # reset the environment and the obstacle motion
                    x, _ = self.env.reset(seed=self.seed, options={})
                    self.obst_motion.reset()
                    k=0
            
                    states = [(x)]
                    TD_temp = []
                    actions = []
                    grad_temp = []
                    obs_positions = [self.obst_motion.current_positions()]
                    xpred_list, ypred_list = self.obst_motion.predict_states(self.horizon)
                    
                    hx =  [ float(hf(cs.DM(x), xpred_list[0:self.mpc.nn.obst.obstacle_num], ypred_list[0:self.mpc.nn.obst.obstacle_num])) 
                             for hf in self.h_funcs ]
            
                    hx_list = [hx]
                    #for NN outputs
                    alphas = []
                    
                    self.x_prev_VMPC        = cs.DM()  
                    self.lam_x_prev_VMPC    = cs.DM()  
                    self.lam_g_prev_VMPC    = cs.DM()  

                    self.x_prev_QMPC        = cs.DM()  
                    self.lam_x_prev_QMPC    = cs.DM()  
                    self.lam_g_prev_QMPC    = cs.DM()  

                    self.x_prev_VMPCrandom  = cs.DM()  
                    self.lam_x_prev_VMPCrandom = cs.DM()  
                    self.lam_g_prev_VMPCrandom = cs.DM()
                    
                    print("reset")
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
            plt.plot(params_history_P[:, 0, 0], label=r"$P_{1,1}$")
            plt.plot(params_history_P[:, 1, 1], label=r"$P_{2,2}$")
            plt.plot(params_history_P[:, 2, 2], label=r"$P_{3,3}$")
            plt.plot(params_history_P[:, 3, 3], label=r"$P_{4,4}$")
            # plt.title("Parameter: P",        fontsize=24)
            plt.xlabel(r"Update Number",     fontsize=20)
            plt.ylabel(r"Value",             fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid()
            plt.tight_layout()
            self.save_figures([(figP,
                   f"P.svg")],
                 experiment_folder)
            

            figstagecost = plt.figure()
            plt.plot(sum_stage_cost_history, 'o', label=r"Stage Cost")
            plt.yscale('log')
            # plt.title("Stage Cost Over Training (Log Scale)", fontsize=24)
            plt.xlabel(r"Episode Number",                    fontsize=20)
            plt.ylabel(r"Stage Cost",                        fontsize=20)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figstagecost,
                   f"stagecost.svg")],
                 experiment_folder)

            
            figtd = plt.figure()
            plt.plot(TD_history, 'o', label=r"TD")
            plt.yscale('log')
            plt.title(r"TD Over Training (Log Scale)")
            plt.xlabel(r"Episode Number")
            plt.ylabel(r"TD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            self.save_figures([(figtd,
                   f"TD.svg")],
                 experiment_folder)

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

            ax.set_xlabel(r"Episode Number", fontsize=20)
            ax.set_ylabel(r"Stage Cost",     fontsize=20)
            ax.tick_params(labelsize=12)
            ax.grid(True)
            ax.legend(fontsize=16)
            figstagecost_nice.tight_layout()
            self.save_figures([(figstagecost_nice,
                   f"stagecost_smoothed.svg")],
                 experiment_folder)
            
            npz_payload = {
            "episodes": episodes,
            "stage_cost": cost,
            "td": TD_history,
            "running_mean": running_mean,
            "running_std": running_std,
            "smoothing_window": np.array([window], dtype=int),
            "params_history_P": params_history_P,
            "stage_cost_valid": self.stage_cost_valid,
            }
            
            np.savez_compressed(os.path.join(experiment_folder, "training_data_nn.npz"), **npz_payload)
            
            return self.best_params
        
        