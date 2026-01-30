import numpy as np
import casadi as cs

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED
from obstacles.obstacles import Obstacles


class RNN:
    def __init__(self, layers_list, positions, radii, horizon, mode_params, modes):

        """
        Build an Elman-style RNN for CBF-based MPC.

        Parameters
        ----------
        layers_list : list[int]
            Network architecture as [input_dim, hidden_dim1, ..., output_dim].
        positions : list[tuple]
            List of (x, y) centers for obstacles (used by CBF input functions).
        radii : list[float]
            Obstacle radii corresponding to positions.
        horizon : int
            Number of time steps to unroll the RNN in MPC.

        Attributes
        ----------
        obst : Obstacles
            Obstacles object containing positions and radii.
        layers_list : list --> layers_list = [input_dim, hidden_dim, ..., output_dim]
            Stored network dimensions.
        rnn_weights_ih : list[cs.MX]
            Input-to-hidden weight symbols per layer.
        rnn_weights_hh : list[cs.MX]
            Hidden-to-hidden recurrent weight symbols (for all but last layer).
        rnn_biases_ih : list[cs.MX]
            Bias symbols per layer.
        hidden_sym_list : list[cs.MX]
            Initial hidden-state symbols for each layer (except output layer).
        input_sym : cs.MX
            Symbolic placeholder for the entire input sequence (input_dim (state,hx) x horizon).
        activations : list
            List of activation functions per layer.
        horizon : int
            Horizon length of MPC and accordingly also for the RNN.
        """
        self.obst         = Obstacles(positions, radii)
        self.layers_list  = layers_list
        self.ns = NUM_STATES
        # four lists of CasADi symbols, one per Rrnn layer:
        self.rnn_weights_ih = []
        self.rnn_weights_hh = []
        self.rnn_biases_ih  = []
        self.hidden_sym_list = []
        # Sequence input for multi-step forward (used in forward_rnn)
        self.input_sym = cs.MX.sym("x", self.layers_list[0], horizon)  # input vector
        self.activations = []
        self._build_network()
        self.np_random = np.random.default_rng(seed=SEED)
        self.horizon = horizon
        
        self.positions = positions
        self.radii     = radii
        #defining bounds for normalizations of positions of the object
        # self.bounds =[ self.obst.mode_params[i]["bounds"] for i in range(self.obst.obstacle_numm) ]
        # self.bounds = np.array(self.bounds)
        self.modes       = modes
        self.mode_params = mode_params

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
                
            #OTHER MODES CAN BE ADDED HERE, NOT IMPLEMENTED YET
                
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
    
    def shifted_sigmoid(self, x, epsilon=1e-6):
        """
        Sigmoid shifted into (epsilon, 1) since we cant have a 0
        """
        return epsilon + (1 - epsilon) / (1 + cs.exp(-x))
    
    def sigmoid(self, x):
        """
        Sigmoid shifted into (epsilon, 1) since we cant have a 0
        """
        return (1) / (1 + cs.exp(-x))
    
    def normalization_z(self, rnn_input):
        
        """
        Normalizes based on maximum and minimum values of the states, h(x) values and obstacle movement values.

        """
        # split the input into three to normalize each seprately
        x_raw    = rnn_input[:self.ns]
        h_raw    = rnn_input[self.ns:self.ns+self.obst.obstacle_num]
        pos_x    = rnn_input[self.ns+self.obst.obstacle_num:self.ns+2*self.obst.obstacle_num]
        pos_y    = rnn_input[self.ns+2*self.obst.obstacle_num:self.ns+3*self.obst.obstacle_num]
        u_raw  = rnn_input[-(NUM_INPUTS):]
        
        def scale_centered(z, zmin, zmax, eps=1e-12):
            return 2 * (z - zmin) / (zmax - zmin + eps) - 1
  
        # x_norm = (nn_input[:4] - mu_states) / sigma_states
        
        #x_min  = cs.DM([-CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0], -CONSTRAINTS_X[0]]) # minimum values of the states
        #x_max = cs.DM([0, 0, 0, 0]) # maximum values of the states
        Xmax, Ymax = CONSTRAINTS_X[0], CONSTRAINTS_X[1]
        Vxmax, Vymax = CONSTRAINTS_X[2], CONSTRAINTS_X[3]  
        
        x_min = cs.DM([-Xmax, -Ymax, -Vxmax, -Vymax])
        x_max = cs.DM([   0.,    0.,  Vxmax,  Vymax ])
        # x_norm = (x_raw-x_min)/(x_max-x_min + 1e-9) # normalize the states based on the maximum values
        x_norm = scale_centered(x_raw, x_min, x_max)
        
        # h_max_list = []
        # for (px, py), r in zip(self.positions, self.radii):
        #     dx = x_min[0] - px
        #     dy = x_min[1] - py
        #     h_max_i = dx**2 + dy**2 - r**2
        #     h_max_list.append(h_max_i)
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
        # for (px, py), r in zip(self.positions, self.radii):
        #     d2 = [ (cx - px)**2 + (cy - py)**2 for (cx, cy) in corners ]
        #     h_max_list.append(max(d2) - r**2)
                    
        h_norm_list = []
        
        h_raw = cs.reshape(h_raw, -1, 1)  # reshape to (m, 1) where m is the number of obstacles

        h_raw_split = cs.vertsplit(h_raw)  # returns a list of (1x1) MX elements
        
        for i, h_i in enumerate(h_raw_split):
            h_max_i = h_max_list[i]
            h_norm_i = scale_centered(h_i, 0.0, h_max_i)
            # h_norm_i = h_i / h_max_i
            # h_norm_i = cs.fmin(cs.fmax(h_norm_i, 0), 1)
            # h_norm_i = cs.fmin(cs.fmax(h_norm_i, -1), 1)
            h_norm_list.append(h_norm_i)
        
        h_norm = cs.vertcat(*h_norm_list)
        #position of object normalization
        # bounds[i] == (xmin_i, xmax_i)
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
                nx = scale_centered(pos_x[i], bx[i,0], bx[i,1])
                # ny = cs.DM(0.5)
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
    # ─── low‐level linear + cell ─────────────────────────────────────────────────
    def linear(self, inp, weight, bias=None):
        
        """
        Compute a linear transform: out = weight @ inp + bias.

        Parameters
        ----------
        inp : cs.MX or cs.DM
            Input vector of shape (input_dim x 1).
        weight : cs.MX
            Weight matrix of shape (output_dim x input_dim).
        bias : cs.MX, optional
            Bias vector of shape (output_dim x 1). If None, no bias added.

        Returns
        -------
        cs.MX
            Output of linear transform (output_dim x 1).
        """
        
        out = weight @ inp
        if bias is not None:
            out = out + bias
        return out

    def rnn_cell(self, inp, hidden, Wih, Whh, bih, activation):
        """
        Combination of hidden and input to compute the new output and hidden state.

        In short it Computes:
            h_new = activation(Wih @ inp + bih + Whh @ h_prev)

        Parameters
        ----------
        inp : cs.MX or cs.DM
            Current input (input_dim x 1).
        hidden : cs.MX or cs.DM or None
            Previous hidden state (output_dim x 1). If None, initialized to zeros.
        Wih : cs.MX
            Input-to-hidden weights (output_dim x input_dim).
        Whh : cs.MX
            Hiddenxtoxhidden weights (output_dim x output_dim).
        bih : cs.MX
            Bias for input-to-hidden (output_dim x 1).
        activation : the activation function
            Element-wise activation function.

        Returns
        -------
        cs.MX
            the output (output_dim x 1).
        """
        if hidden is None:
            hidden = cs.DM.zeros(Wih.shape[0], 1)
            
        pre = self.linear(inp, Wih, bih) + self.linear(hidden, Whh)
        return activation(pre)


    def make_rnn_step(self):
        """
        Build a CasADi Function for one RNN step.

        Returns
        -------
        cs.Function
            Rstep(h_current, ..., h_current_K, (x, h_cbf(x_t))= x_t
                   Wih_0, bih_0, Whh_0, ..., Wih_{L}, bih_{L})`
            → (h_next_1, ..., h_next_K, y_t).

        Inputs
        ------
        h_current                     : MX, shape (hidden_dim_i x 1)
        obs_positions?
        x_t = (x, h_cbf(x_t), obs_positions)        : MX, shape (input_dim x 1)
        Wih_i                        : MX, shape (hidden_dim_i x input_dim_i)
        bih_i                        : MX, shape (hidden_dim_i x 1)
        Whh_i                        : MX, shape (hidden_dim_i x hidden_dim_i) (for i < output layer)

        Outputs
        -------
        h_next    : MX, updated hidden state for next layer
        y_t        : MX, final-layer output (output_dim x 1)
        """
        
        L        = len(self.rnn_weights_ih) # number of RNN layers\
            
            
        print(f"L:{L}")    
        input_sym = cs.MX.sym("x", self.layers_list[0],1)  # input vector 
        params   = []
        next_hid_list = []
        
        for i in range(L-1):
            params += [
                self.rnn_weights_ih[i],
                self.rnn_biases_ih [i],
                self.rnn_weights_hh[i],
            ]
        # now the last layer only has Wih & bih:
        params += [
        self.rnn_weights_ih[L-1],
        self.rnn_biases_ih [L-1],
            ]
                  
        out_int = input_sym
        for i in range(L):
            if i<L-1:
                out_int = self.rnn_cell(
                    out_int, self.hidden_sym_list[i],
                    self.rnn_weights_ih[i],
                    self.rnn_weights_hh[i],
                    self.rnn_biases_ih[i],
                    self.activations[i]
                )
                next_hid_list.append(out_int)
            else:
                out_int = self.rnn_cell(
                    out_int, None,
                    self.rnn_weights_ih[i],
                    self.Whh0,  # no Whh for last layer
                    self.rnn_biases_ih[i],
                    self.activations[i]
                )
        y_out = out_int   # row‐vector output
    
        
        return cs.Function("Rstep", [*self.hidden_sym_list, input_sym, *params], [*next_hid_list, y_out])
    
    
    def forward_rnn(self):
        """
        Unroll the RNN for horizon steps.

        Returns
        -------
        cs.Function
            rnn_forward(flat_input, h0_1,...,h0_K, Wih_0,bih_0,Whh_0,...,Wih_L,bih_L)
            → (H_stack, Y_stack).

        Inputs
        ------
        flat_input : MX, shape ((input_dim + m) * horizon × 1)
            Stacked [x0; h_obs(x0); x1; h_obs(x1); ...].
        h0_i       : MX, initial hidden states.
        Wih_i, bih_i, Whh_i : MX, as in `make_rnn_step`.

        Outputs
        -------
        H_stack    : MX, all hidden states over time, shape (sum(hidden_dims)*horizon × 1)
        Y_stack    : MX, all final‐layer outputs over time, shape (output_dim*horizon × 1)
        """
        #TODO : CHECK IS IT REALLY GIVING THE RIGHT HIDDEN STATES BACK
        rnn_step = self.make_rnn_step()

        h = self.hidden_sym_list   # MX
        params = self.get_flat_parameters_list()  

        h_history = [cs.vertcat(*h)] # DO I NEED TO DO THIS?
        y_history = []
        
        

        for i in range(self.horizon):
            x_t_raw = self.input_sym[:, i]
            x_t = self.normalization_z(x_t_raw)
            # x_t = self.normalization_z(x_t)
            *h, y = rnn_step(*h, x_t, *params)
            h_history.append(cs.vertcat(*h))
            y_history.append(y)

        H_stack = cs.vertcat(*h_history)   # (hidden_dim * horizon)×1
        Y_stack = cs.vertcat(*y_history)   # (output_dim * horizon)×1
        
        flat_input_sym = cs.reshape(self.input_sym, -1, 1)

        return cs.Function(
        "rnn_forward",
        [flat_input_sym, *self.hidden_sym_list, *params],
        [H_stack, Y_stack],
        {"cse": True}
        )

    def _build_network(self):
            """
            Create symbolic parameters & initial-state symbols for each layer:
            - For layers 0..L-2: Wih, bih, Whh, hidden_sym
            - For last layer:     Wih, bih; Whh0 := zeros
            Also activations (leaky_relu and shifted_sigmoid).
            """
            for i in range(len(self.layers_list) - 1):

                in_dim  = self.layers_list[i]
                hid_dim = self.layers_list[i+1]

                Wih = cs.MX.sym(f"Wih{i}", hid_dim, in_dim)
                bih = cs.MX.sym(f"bih{i}", hid_dim, 1)
                
                self.rnn_weights_ih.append(Wih)
                self.rnn_biases_ih.append(bih)

                if i != (len(self.layers_list) - 2):  # not the last layer
                    hidden_sym = cs.MX.sym(f"hinit{i}", hid_dim, 1)
                    Whh = cs.MX.sym(f"Whh{i}", hid_dim, hid_dim)
                    self.rnn_weights_hh.append(Whh)
                    self.hidden_sym_list.append(hidden_sym)
                else:
                    # Whh0 = cs.MX.sym(f"Whhzero", self.layers_list[-1], self.layers_list[-1])
                    self.Whh0 = cs.DM.zeros(self.layers_list[-1], self.layers_list[-1])
    
                if i == len(self.layers_list) - 2:
                    self.activations.append(self.sigmoid)
                else:
                    self.activations.append(self.leaky_relu)
                    
                    
    def initialize_parameters(self):
        """
        He-uniform init to match exactly get_flat_parameters() ordering.
        Returns
        -------
        flat_params : casadi.DM, shape=(410,1)
            All Wih, bih, Whh (where applicable) stacked exactly as get_flat_parameters().
        Wih_vals    : list of numpy.ndarray
            The fan-in/fan-out weight matrices.
        Whh_vals    : list of numpy.ndarray
            The recurrent weight matrices (one per non-last layer).
        bih_vals    : list of numpy.ndarray
            The input-to-hidden biases (one per layer).
        """
        L = len(self.layers_list) - 1  # number of layers
        Wih_vals = []
        Whh_vals = []
        bih_vals = []
        
        
        neg_slope = 0.01
        gain_leaky = np.sqrt(2.0 / (1.0 + neg_slope**2))  # around 1.414
        target_rho = getattr(self, "spectral_radius", 0.95)

    
        for i in range(L):
            fan_in, fan_out = self.layers_list[i], self.layers_list[i+1]

            if i < L-1:
            # bound1 = 3*np.sqrt(6.0 / (fan_in + fan_out))
                # bound_low = np.sqrt(6.0 / fan_in)
                # bound_high = np.sqrt(6.0 / fan_out)
                bound = np.sqrt(6.0 / fan_in) / gain_leaky
                
                Wih_v = self.np_random.uniform(low=-bound, high=bound, size=(fan_out, fan_in)) #self.np_random.uniform(-bound1, bound1, size=(fan_out, fan_in))
                Wih_vals.append(Wih_v)

                Whh_v = self.np_random.uniform(low=-bound, high=bound, size=(fan_out, fan_out))
                # Whh_vals.append(Whh_v)

                Whh_v = self._scale_to_spectral_radius(Whh_v, target=target_rho)
                eigvals = np.linalg.eigvals(Whh_v)
                rho = np.max(np.abs(eigvals)).real
                print(f"max eigenvalue of Whh_{i}: {rho:.4f} (target: {target_rho})")
                Whh_vals.append(Whh_v)
            
            else:
                bound = 0.01*np.sqrt(6.0 / (fan_in + fan_out))
                Wih_v = self.np_random.uniform(-bound, bound, size=(fan_out, fan_in))
                Wih_vals.append(Wih_v)
            
            bih_v = np.zeros((fan_out, 1))
            bih_vals.append(bih_v)
                
        raws = []

        for i in range(L-1):
            raws.append(Wih_vals[i].reshape(-1, 1))
            raws.append(bih_vals[i].reshape(-1, 1))
            raws.append(Whh_vals[i].reshape(-1, 1))

        raws.append(Wih_vals[L-1].reshape(-1, 1))
        raws.append(bih_vals[L-1].reshape(-1, 1))

  
        flat = cs.vertcat(*[cs.DM(r) for r in raws])
        return flat, Wih_vals, Whh_vals, bih_vals

    def get_flat_parameters_list(self):
        """
       for mapaccum call
        """
        out = []
        L = len(self.layers_list) - 1  
 
        for i in range(L-1):
            out += [
                self.rnn_weights_ih[i],
                self.rnn_biases_ih [i],
                self.rnn_weights_hh[i],
            ]
     
        out += [
            self.rnn_weights_ih[L-1],
            self.rnn_biases_ih [L-1],
        ]
        return out

    def get_flat_parameters(self):
        """
        for NLP p-argument.
        """
        raws = []
        L = len(self.layers_list) - 1
      
        for i in range(L-1):
            raws += [
                cs.reshape(self.rnn_weights_ih[i], -1, 1),
                cs.reshape(self.rnn_biases_ih [i], -1, 1),
                cs.reshape(self.rnn_weights_hh[i], -1, 1),
            ]
 
        raws += [
            cs.reshape(self.rnn_weights_ih[L-1], -1, 1),
            cs.reshape(self.rnn_biases_ih [L-1], -1, 1),
        ]
        return cs.vertcat(*raws)
    
    def unpack_flat_parameters(self, flat_params):
        """
        Given a (Px1) vector flat_params in the same ordering as get_flat_parameters(),
        return a Python list [Wih0, bih0, Whh0, Wih1, bih1, Whh1, …, Wih_{L-1}, bih_{L-1}].
        Done so we turn params["rnn_params"] into a list of matrices as set up accordingly for this class.
        """
        # Number of layers
        L = len(self.layers_list) - 1

        # Precompute fan‐in/out dims
        dims = [(self.layers_list[i], self.layers_list[i+1]) for i in range(L)]

        idx = 0
        unpacked = []

        for i, (fan_in, fan_out) in enumerate(dims):
            # Wih_i has shape (fan_out, fan_in)
            num_Wih = fan_out * fan_in
            Wih_i = cs.reshape(flat_params[idx:idx+num_Wih], fan_out, fan_in)
            unpacked.append(Wih_i)
            idx += num_Wih

            # bih_i has shape (fan_out, 1)
            num_bih = fan_out
            bih_i = cs.reshape(flat_params[idx:idx+num_bih], fan_out, 1)
            unpacked.append(bih_i)
            idx += num_bih

            # Only non‐last layers have a Whh
            if i < L-1:
                num_Whh = fan_out * fan_out
                Whh_i = cs.reshape(flat_params[idx:idx+num_Whh], fan_out, fan_out)
                unpacked.append(Whh_i)
                idx += num_Whh

        return unpacked
