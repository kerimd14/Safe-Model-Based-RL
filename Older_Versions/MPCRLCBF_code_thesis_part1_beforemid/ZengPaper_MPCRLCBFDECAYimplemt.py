import gymnasium as gym 
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr

class env(gym.Env):
    def __init__(self, sampling_time: float):
        super().__init__()
        self.na = 2
        self.ns = 4
        self.umax = 1
        self.umin = -1

        self.observation_space = gym.spaces.Box(-5, 5, (self.ns,), np.float64)
        self.action_space = gym.spaces.Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = sampling_time

        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-5, -5, 0, 0])
        return self.x, {}

    def step(self, action):
        u = np.asarray(action).reshape(self.na)
        self.B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt]
        ])
        x_new = self.A @ self.x + self.B @ u
        self.x = x_new
        return x_new, np.nan, False, False, {}

class MPC:
    ns = 4  # number of states
    na = 2  # number of inputs
    horizon = 30  # MPC horizon
    Q = 10 * np.identity(4)
    R = np.identity(2)

    X_lower_bound = -5 * np.ones(ns * horizon)
    X_upper_bound = 5 * np.ones(ns * horizon)
    state_const_lbg = np.zeros(ns * horizon)
    state_const_ubg = np.zeros(ns * horizon)

    cbf_const_lbg = -np.inf * np.ones(horizon)
    cbf_const_ubg = np.zeros(horizon)

    def __init__(self, P=None):
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = MPC.horizon

        # Use provided terminal cost matrix P; otherwise default.
        if P is None:
            self.P = np.diag([100, 100, 100, 100])
        else:
            self.P = P

        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon + 1)
        self.U_sym = cs.MX.sym("U", self.na, self.horizon)
        self.omega_sym = cs.MX.sym("w")
        self.Pw_sym = cs.MX.sym("Pw")
        self.omega0_sym = cs.MX.sym("w0")

        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)
        x_new = self.A_sym @ self.x_sym + self.B_sym @ self.u_sym
        self.dynamics_f = cs.Function('f', [self.A_sym, self.B_sym, self.x_sym, self.u_sym],
                                       [x_new], ['A', 'B', 'x', 'u'], ['ode'])

        h = (self.x_sym[0] - self.pos[0])**2 + (self.x_sym[1] - self.pos[1])**2 - self.r**2 
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])

    def dcbf(self, h, x, u, dynamics, alphas):
        x_next = dynamics(self.A_sym, self.B_sym, x, u)
        phi = h(x)
        for alpha in alphas:
            phi_next = cs.substitute(phi, x, x_next)
            phi = phi_next - phi + alpha(phi)
        return phi

    def state_const(self):
        state_const_list = []
        for k in range(self.horizon):
            state_const_list.append(
                self.X_sym[:, k+1] - (self.A_sym @ self.X_sym[:, k] + self.B_sym @ self.U_sym[:, k])
            )
        self.state_const_list = cs.vertcat(*state_const_list)
        return

    def cbf_func(self):
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f,
                        [lambda y: self.omega_sym * y])
        return cs.Function('cbff', [self.A_sym, self.B_sym, self.x_sym, self.u_sym, self.omega_sym],
                           [cbf], ['A', 'B', 'x', 'u', 'alpha'], ['cbff'])

    def cbf_const(self):
        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):
            cbf_const_list.append(
                cbf_func(self.A_sym, self.B_sym, self.X_sym[:, k], self.U_sym[:, k], self.omega_sym)
            )
        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        return

    def objective_method(self):
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q @ self.X_sym[:, k] +
             self.U_sym[:, k].T @ self.R @ self.U_sym[:, k])
            for k in range(self.horizon)
        )
        terminal_cost = cs.bilin(self.P, self.X_sym[:, -1])
        opt_decay = self.Pw_sym * ((self.omega_sym - self.omega0_sym)**2)
        self.objective = terminal_cost + stage_cost + opt_decay
        return

    def MPC_solver(self):
        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)

        A_sym_flat = cs.reshape(self.A_sym, -1, 1)
        B_sym_flat = cs.reshape(self.B_sym, -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, self.omega_sym),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.Pw_sym, self.omega0_sym),
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

        solver = cs.nlpsol("solver", "fatrop", nlp, opts)
        return solver

def MPC_func(x, mpc, Pw, omega0):
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

    solver_inst = mpc.MPC_solver()

    U_lower_bound = -np.ones(mpc.na * mpc.horizon)
    U_upper_bound = np.ones(mpc.na * mpc.horizon)

    lbx = np.concatenate([np.array(x).flatten(), mpc.X_lower_bound, U_lower_bound, np.array([1e-6])])
    ubx = np.concatenate([np.array(x).flatten(), mpc.X_upper_bound, U_upper_bound, np.array([1])])

    lbg = np.concatenate([mpc.state_const_lbg, mpc.cbf_const_lbg])
    ubg = np.concatenate([mpc.state_const_ubg, mpc.cbf_const_ubg])

    solution = solver_inst(
        p=cs.vertcat(cs.reshape(A, -1, 1), cs.reshape(B, -1, 1), Pw, omega0),
        ubx=ubx,
        lbx=lbx,
        ubg=ubg,
        lbg=lbg
    )
    u_opt = solution["x"][mpc.ns * (mpc.horizon + 1): mpc.ns * (mpc.horizon + 1) + mpc.na]
    return u_opt, solution["f"]

def calculate_trajectory_length(states):
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)

def run_simulation_with_P_and_Pw(P_list, Pw_list, env_class):
    # Fix omega0 while varying P and Pw.
    omega0_fixed = cs.DM(0.5)
    env_instance = env_class(sampling_time=0.2)

    # Define distinct marker styles for each Pw value
    marker_list = ['o', 's', '^', 'd']
    
    for P in P_list:
        for pw_idx, Pw in enumerate(Pw_list):
            state, _ = env_instance.reset(seed=42, options={})
            states = [state[:2]]
            mpc = MPC(P=P)  # Terminal cost matrix changes per simulation

            for _ in range(200):
                action, _ = MPC_func(state, mpc, Pw, omega0_fixed)
                # Clip action within bounds
                action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                state, _, _, _, _ = env_instance.step(action)
                states.append(state[:2])
            states = np.array(states)
            plt.plot(
                states[:, 0], states[:, 1],
                linestyle='-', marker=marker_list[pw_idx],
                label=f"P: {np.diag(P)}, Pw: {float(Pw)}"
            )
            trajectory_length = calculate_trajectory_length(states)
            print(f"Trajectory length for P diag {np.diag(P)} and Pw {float(Pw)} is {trajectory_length:.2f} units")

    # Plot the obstacle (a circle)
    circle = plt.Circle((-2, -2.25), 1.5, color="k", fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.xlim([-5, 0])
    plt.ylim([-5, 0])
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title("Trajectories for varying terminal cost matrices P and Pw (fixed ω₀)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()

# Define lists for P and Pw values.
P_list = [
    np.diag([100, 100, 100, 100])
    # np.diag([105, 105, 100, 100]),
    # np.diag([100, 100, 95, 95]),
    # np.diag([105, 105, 95, 95])
]

Pw_list = [
    cs.DM(100)
    # cs.DM(10),
    # cs.DM(100),
    # cs.DM(1000)
]

run_simulation_with_P_and_Pw(P_list, Pw_list, env)

import gymnasium as gym 
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from control import dlqr

class env(gym.Env):
    def __init__(self, sampling_time: float):
        super().__init__()
        self.na = 2
        self.ns = 4
        self.umax = 1
        self.umin = -1

        self.observation_space = gym.spaces.Box(-5, 5, (self.ns,), np.float64)
        self.action_space = gym.spaces.Box(self.umin, self.umax, (self.na,), np.float64)
        self.dt = sampling_time

        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-5, -5, 0, 0])
        return self.x, {}

    def step(self, action):
        u = np.asarray(action).reshape(self.na)
        self.B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt]
        ])
        x_new = self.A @ self.x + self.B @ u
        self.x = x_new
        return x_new, np.nan, False, False, {}

class MPC:
    ns = 4  # number of states
    na = 2  # number of inputs
    horizon = 30  # MPC horizon
    Q = 10 * np.identity(4)
    R = np.identity(2)

    X_lower_bound = -5 * np.ones(ns * horizon)
    X_upper_bound = 5 * np.ones(ns * horizon)
    state_const_lbg = np.zeros(ns * horizon)
    state_const_ubg = np.zeros(ns * horizon)

    cbf_const_lbg = -np.inf * np.ones(horizon)
    cbf_const_ubg = np.zeros(horizon)

    def __init__(self, P=None):
        self.ns = MPC.ns
        self.na = MPC.na
        self.horizon = MPC.horizon

        # Use provided terminal cost matrix P; otherwise default.
        if P is None:
            self.P = np.diag([100, 100, 100, 100])
        else:
            self.P = P

        self.A_sym = cs.MX.sym("A", self.ns, self.ns)
        self.B_sym = cs.MX.sym("B", self.ns, self.na)
        self.X_sym = cs.MX.sym("X", self.ns, self.horizon + 1)
        self.U_sym = cs.MX.sym("U", self.na, self.horizon)
        self.omega_sym = cs.MX.sym("w")
        self.Pw_sym = cs.MX.sym("Pw")
        self.omega0_sym = cs.MX.sym("w0")

        self.pos = cs.DM([-2, -2.25])
        self.r = cs.DM(1.5)
        self.x_sym = cs.MX.sym("x", MPC.ns)
        self.u_sym = cs.MX.sym("u", MPC.na)
        x_new = self.A_sym @ self.x_sym + self.B_sym @ self.u_sym
        self.dynamics_f = cs.Function('f', [self.A_sym, self.B_sym, self.x_sym, self.u_sym],
                                       [x_new], ['A', 'B', 'x', 'u'], ['ode'])

        h = (self.x_sym[0] - self.pos[0])**2 + (self.x_sym[1] - self.pos[1])**2 - self.r**2 
        self.h_func = cs.Function('h', [self.x_sym], [h], ['x'], ['cbf'])

    def dcbf(self, h, x, u, dynamics, alphas):
        x_next = dynamics(self.A_sym, self.B_sym, x, u)
        phi = h(x)
        for alpha in alphas:
            phi_next = cs.substitute(phi, x, x_next)
            # alpha(phi) shrinks the safe set
            phi = phi_next - phi + alpha(phi)
        return phi

    def state_const(self):
        state_const_list = []
        for k in range(self.horizon):
            state_const_list.append(
                self.X_sym[:, k+1] - (self.A_sym @ self.X_sym[:, k] + self.B_sym @ self.U_sym[:, k])
            )
        self.state_const_list = cs.vertcat(*state_const_list)
        return

    def cbf_func(self):
        cbf = self.dcbf(self.h_func, self.x_sym, self.u_sym, self.dynamics_f,
                        [lambda y: self.omega_sym * y])
        return cs.Function('cbff', [self.A_sym, self.B_sym, self.x_sym, self.u_sym, self.omega_sym],
                           [cbf], ['A', 'B', 'x', 'u', 'alpha'], ['cbff'])

    def cbf_const(self):
        cbf_func = self.cbf_func()
        cbf_const_list = []
        for k in range(self.horizon):
            cbf_const_list.append(
                cbf_func(self.A_sym, self.B_sym, self.X_sym[:, k], self.U_sym[:, k], self.omega_sym)
            )
        self.cbf_const_list = cs.vertcat(*cbf_const_list)
        return

    def objective_method(self):
        stage_cost = sum(
            (self.X_sym[:, k].T @ self.Q @ self.X_sym[:, k] +
             self.U_sym[:, k].T @ self.R @ self.U_sym[:, k])
            for k in range(self.horizon)
        )
        terminal_cost = cs.bilin(self.P, self.X_sym[:, -1])
        opt_decay = self.Pw_sym * ((self.omega_sym - self.omega0_sym)**2)
        self.objective = terminal_cost + stage_cost + opt_decay
        return

    def MPC_solver(self):
        self.state_const()
        self.objective_method()
        self.cbf_const()

        X_flat = cs.reshape(self.X_sym, -1, 1)
        U_flat = cs.reshape(self.U_sym, -1, 1)

        A_sym_flat = cs.reshape(self.A_sym, -1, 1)
        B_sym_flat = cs.reshape(self.B_sym, -1, 1)

        nlp = {
            "x": cs.vertcat(X_flat, U_flat, self.omega_sym),
            "p": cs.vertcat(A_sym_flat, B_sym_flat, self.Pw_sym, self.omega0_sym),
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

        solver = cs.nlpsol("solver", "fatrop", nlp, opts)
        return solver

def MPC_func(x, mpc, Pw, omega0):
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

    solver_inst = mpc.MPC_solver()

    U_lower_bound = -np.ones(mpc.na * mpc.horizon)
    U_upper_bound = np.ones(mpc.na * mpc.horizon)

    lbx = np.concatenate([
        np.array(x).flatten(),
        mpc.X_lower_bound, 
        U_lower_bound, 
        np.array([1e-6])
    ])
    ubx = np.concatenate([
        np.array(x).flatten(),
        mpc.X_upper_bound, 
        U_upper_bound, 
        np.array([1])
    ])

    lbg = np.concatenate([mpc.state_const_lbg, mpc.cbf_const_lbg])
    ubg = np.concatenate([mpc.state_const_ubg, mpc.cbf_const_ubg])

    solution = solver_inst(
        p=cs.vertcat(cs.reshape(A, -1, 1), cs.reshape(B, -1, 1), Pw, omega0),
        ubx=ubx,
        lbx=lbx,
        ubg=ubg,
        lbg=lbg
    )
    u_opt = solution["x"][mpc.ns * (mpc.horizon + 1): mpc.ns * (mpc.horizon + 1) + mpc.na]
    return u_opt, solution["f"]

def calculate_trajectory_length(states):
    distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
    return np.sum(distances)

def run_simulation_with_P_and_Pw(P_list, Pw_list, env_class):
    omega0_fixed = cs.DM(0.5)
    
    # Define distinct marker styles for each Pw value for trajectory plotting
    marker_list = ['o', 's', '^', 'd']
    
    # Loop over the P and Pw values
    for P in P_list:
        for pw_idx, Pw in enumerate(Pw_list):
            state, _ = env_instance.reset(seed=42, options={})
            states = [state]
            mpc = MPC(P=P)

            for _ in range(200):
                action, _ = MPC_func(state, mpc, Pw, omega0_fixed)
                action = cs.fmin(cs.fmax(cs.DM(action), -1), 1)
                state, _, _, _, _ = env_instance.step(action)
                states.append(state)
            states = np.array(states)  # Shape: (time_steps, 4)
            
            # Create a separate velocity plot (states 2 and 3) for this run
            time = np.arange(len(states)) * env_instance.dt
            plt.figure()
            plt.plot(time, states[:, 2], "o-", label="Velocity x")
            plt.plot(time, states[:, 3], "o-", label="Velocity y")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity Value")
            plt.title(f"Velocity Plot for P diag {np.diag(P)}, Pw {float(Pw)}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
    
# Create one environment instance for all simulations
env_instance = env(sampling_time=0.2)

# Define lists for P and Pw values.
P_list = [
    np.diag([100, 100, 100, 100])
    # np.diag([105, 105, 100, 100]),
    # np.diag([100, 100, 95, 95]),
    # np.diag([105, 105, 95, 95])
]

Pw_list = [
    cs.DM(100)
    # cs.DM(10),
    # cs.DM(100),
    # cs.DM(1000)
]

run_simulation_with_P_and_Pw(P_list, Pw_list, env)