import numpy as np
import casadi as cs

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED
from obstacles.obstacles import Obstacles


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
        
        return cs.vertcat(x_norm, h_norm, pos_norm)
    
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
      
        input = cs.vertcat(x, h_func_list, obs_pred_x, obs_pred_y)

        y = self.forward(input)

       

        return cs.Function('NN', [x, h_func_list, obs_pred_x, obs_pred_y, self.get_flat_parameters()],[y])
    

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

