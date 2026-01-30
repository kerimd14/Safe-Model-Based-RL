"""""
Kerim Dzhumageldyev

2D double integrator implementation



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


        # xmin = [-5; -5; -5; -5];
        # xmax = [5; 5; 5; 5];
        # umin = [-1; -1];
        # umax = [1; 1];

        self.observation_space = Box(-5, 5, (self.ns,), np.float64)
        self.action_space = Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = sampling_time


        # x0 = [-5; -5; 0; 0];
        # time_total = 20.0;
        # dt = 0.2;
        # P = 100*eye(4);
        # Q = 10*eye(4);
        # R = eye(2);
        # N = 8;

        self.A = np.array([
            [1, 0, self.dt, 0], 
            [0, 1, 0, self.dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])


    def reset(self, seed, options):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-5, -5, 0, 0])
        assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
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
        assert self.action_space.contains(u), f"invalid action {u}"

        x_new = self.A @ self.x + self.B @ u
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self.x = x_new
        return x_new, np.nan, False, False, {}

    # def dynamics(self, x, u):
    #     """Computes the dynamics of the system."""
    #     f, g = self.dynamics_components(x)
    #     return f + g @ u

    # def h(self, y):
    #     """Safety constraint."""
    #     return y[0] - 1.8 * y[1]

class MPC:
    # constant of MPC class
    ns = 4 # num of states
    na = 2 # num of inputs
    horizon = 8 # MPC horizon
    # gamma = 0.5
    
    P = 100*np.identity(4)
    Q = 10*np.identity(4)
    R = np.identity(2)

    X_lower_bound = -5* np.ones(ns * (horizon))
    X_upper_bound = 5* np.ones(ns  * (horizon))
    state_const_lbg = np.zeros(1*ns * (horizon))
    state_const_ubg = np.zeros(1*ns  * (horizon))

    cbf_const_lbg = -np.inf * np.ones(1*(horizon))
    cbf_const_ubg = np.zeros(1*(horizon))



    def __init__(self):
        """
        Initialize the MPC class with parameters.
        """
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = MPC.horizon
        # self.x0 = cs.MX.sym("x0")
        # Riccati to calculate Sn
        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon+1)
        self.U_sym = cs.MX.sym("U",self.na, self.horizon)
        self.alpha_sym = cs.MX.sym("alpha")
        # self.S_sym= cs.MX.sym("S",self.ns, self.horizon)
 
        self.np_random = np.random.default_rng(seed=45)


        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)
        x_new  = self.A_sym @ self.x_sym + self.B_sym @ self.u_sym
        self.dynamics_f = cs.Function('f', [self.A_sym, self.B_sym, self.x_sym, self.u_sym], [x_new], ['A', 'B', 'x','u'], ['ode'])

        h = (self.x_sym[0]-(self.pos[0]))**2 + (self.x_sym[1]-(self.pos[1]))**2 - self.r**2 
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])

        


    
    # def dynamics_f(self, x, u):
    #     return self.A_sym@x + self.B_sym@u

    # def lie_derivative(self,
    #     ex, arg, field, order: int = 1
    # ):
    #     """Computes the Lie derivative of the expression ``ex`` with respect to the argument
    #     ``arg`` along the field ``field``.

    #     Parameters
    #     ----------
    #     ex : casadi SX or MX
    #         Expression to compute the Lie derivative of.
    #     arg : casadi SX or MX
    #         Argument with respect to which to compute the Lie derivative.
    #     field : casadi SX or MX
    #         Field along which to compute the Lie derivative.
    #     order : int, optional
    #         Order (>= 1) of the Lie derivative, by default ``1``.

    #     Returns
    #     -------
    #     casadi SX or MX
    #         The Lie derivative of the expression ``ex`` with respect to the argument ``arg``
    #         along the field ``field``.
    #     """
    #     # print(ex)
    #     # print(arg)
    #     deriv = cs.mtimes(cs.jacobian(ex, arg), field)
    #     if order <= 1:
    #         return deriv
    #     return self.lie_derivative(deriv, arg, field, order - 1)


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
            \phi_m(x_k) = \phi_{m-1}(x_{k+1}) - \phi_{m-1}(x_k) + \alpha_m(\phi_{m-1}(x_k))

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
        x_next = dynamics(self.A_sym, self.B_sym, x, u)
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
            # self.X_sym_state_const[:,k+1] = self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym
            state_const_list.append(self.X_sym[:,k+1] - (self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k]))
        self.state_const_list = cs.vertcat(*state_const_list)
        return 
    
    def cbf_func(self):
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f, [lambda y: self.alpha_sym * y])
        return cs.Function('cbff', [self.A_sym, self.B_sym, self.x_sym, self.u_sym, self.alpha_sym], [cbf], ['A', 'B', 'x','u','alpha'], ['cbff'])
        
    
    def cbf_const(self):
        """""
        used to construct cbf constraints 
        """
        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):
            # self.X_sym_state_const[:,k+1] = self.A_sym @ self.X_sym[:,k] + self.B_sym @ self.U_sym[:,k] + self.b_sym
            cbf_const_list.append(cbf_func(self.A_sym, self.B_sym, self.X_sym[:,k], self.U_sym[:,k], self.alpha_sym))   
        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        return 
    
    def objective_method(self):
        """""
        stage cost calculation
        """
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q @ self.X_sym[:, k] + 
            self.U_sym[:, k].T @ self.R @ self.U_sym[:, k])
            for k in range(self.horizon)
        )
        terminal_cost = cs.bilin(self.P, self.X_sym[:, -1])
        self.objective = terminal_cost + stage_cost 
        return
    
    def MPC_solver(self):
        """""
        solves the MPC according to V-value function setup
        """

        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)  # Flatten
        U_flat =cs.reshape(self.U_sym, -1, 1) 

        A_sym_flat = cs.reshape(self.A_sym , -1, 1)
        B_sym_flat = cs.reshape(self.B_sym , -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.alpha_sym),
            "f": self.objective, 
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
        #make casadi function out of this

        return MPC_solver
    

def MPC_func(x, mpc, alpha):
        dt = 0.2
        
        A = np.array([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ])

        B = np.array([
            [0.5 * dt**2, 0], 
            [0, 0.5 * dt**2], 
            [dt, 0], 
            [0, dt]
        ])

        # alpha = 1

        solver_inst = mpc.MPC_solver() 
        
        U_lower_bound = -np.ones(mpc.na * (mpc.horizon))
        U_upper_bound = np.ones(mpc.na * (mpc.horizon))

        lbx = np.concatenate([np.array(x).flatten(), mpc.X_lower_bound, U_lower_bound])  
        ubx = np.concatenate([np.array(x).flatten(), mpc.X_upper_bound, U_upper_bound])

        lbg = np.concatenate([mpc.state_const_lbg, mpc.cbf_const_lbg])  
        ubg = np.concatenate([mpc.state_const_ubg, mpc.cbf_const_ubg])

        solution = solver_inst(p = cs.vertcat(cs.reshape(A, -1 , 1), cs.reshape(B, -1, 1), alpha),
            ubx=ubx,  
            lbx=lbx,
            ubg =ubg,
            lbg=lbg
        )
        # print (f"here is the solution: {solution['x'][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]}")
        # print (solution["f"])
        u_opt = solution["x"][mpc.ns * (mpc.horizon+1):mpc.ns * (mpc.horizon+1) + mpc.na]

        return u_opt, solution["f"]

def calculate_trajectory_length(states):
    # compute eucldian distance and then sum
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)

def run_simulation(alpha_values, env):
    env = env(sampling_time=0.2)

    for alpha in alpha_values:
        state, _ = env.reset(seed=42, options={})
        
        states = [state[:2]]
        actions = []
        mpc = MPC()

        for i in range(600):
            action, _ = MPC_func(state, mpc, alpha)
            action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
            state, _, done, _, _ = env.step(action)
            states.append(state[:2])
            actions.append(action)
            
            if (1e-3 > np.abs(state[0])) & (1e-3 > np.abs(state[1])):
                break

        states = np.array(states)

        plt.plot(states[:, 0], states[:, 1],"o-" ,label=rf"$\gamma={alpha}$")

        trajectory_length = calculate_trajectory_length(states)
        print(f"Trajectory length for alpha={alpha}: {trajectory_length} units")
        
    # Plot obstacle
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])

    # Labels
    plt.xlabel('$X$',fontsize=20); plt.ylabel('$Y$', fontsize=20)
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

# Run simulations for different values of alpha
alpha_values = [0.1,0.2, 0.3, 0.9]
run_simulation(alpha_values, env)

# env = env(sampling_time=0.2)

# #reset environment
# state, _ = env.reset(seed=42, options={})


# states = [state[:2]] 
# actions = []
# mpc = MPC()

# for _ in range(200):
#     action, _ = MPC_func(state, mpc, alpha)
#     print(action)
#     action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
#     print(action)
#     state, _, done, _, _ = env.step(action)
#     states.append(state[:2]) 
#     actions.append(action)
    
#     # if (1.5)**2 >= ((state[0]+2)**2 + (state[1]+2.25)**2):
#     #     break  

# states = np.array(states)


# plt.figure(figsize=(6, 6))
# plt.plot(states[:, 0], states[:, 1], "bo-", label="Trajectory")
# # plt.scatter(-2, -2.25, s=300, color="r", label="Obstacle")  # Obstacle location
# circle = plt.Circle((-2, -2.25), 1.5, color="r", fill=False, linewidth=2)
# plt.gca().add_patch(circle) 


# plt.xlim([-10, 10])
# plt.ylim([-10, 10])

# plt.xlabel("$x$ (m)")
# plt.ylabel("$y$ (m)")
# plt.title("Double Integrator Trajectory with Obstacle")
# plt.legend()
# # plt.grid()
# plt.axis("equal")
# plt.show()