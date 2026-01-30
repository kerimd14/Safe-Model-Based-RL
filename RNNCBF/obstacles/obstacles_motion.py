# obstacles/obstacles_motion.py
import copy
import numpy as np

from config import SAMPLING_TIME


class ObstacleMotion:

    def __init__(self, positions: list, modes: list[str], mode_params: list[dict]):
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

        self.m = len(positions)
        self.modes = modes

        # deepcopy to avoid modifying the original mode_params
        self.mode_params = copy.deepcopy(mode_params)

        # intializing states and parameters for velocity models
        self.vx = np.zeros(self.m)
        self.vy = np.zeros(self.m)

        self.cx = np.array([p[0] for p in positions], dtype=float)
        self.cy = np.array([p[1] for p in positions], dtype=float)

        self.dt = SAMPLING_TIME
        self.t = 0.0

        # different types of movements that can be used
        self._step_functions = {
            "static": self._step_static,
            "random": self._step_random_walk,
            "sinusoid": self._step_sinusoid,
            "step_bounce": self._step_bounce,
            "orbit": self._step_orbit,
        }

        # intial state, will be used to restore the obstacles
        # to their initial state when reset is called
        self._init_state = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t": self.t,
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

        self.cx += -self.vx * self.dt
        self.cy += -self.vy * self.dt

        self.t += self.dt
        return self.cx.copy(), self.cy.copy()

    def _step_static(self, i: int):
        """Obstacle i remains fixed in place."""
        return 0.0, 0.0

    def _step_random_walk(self, i: int):
        """Gaussian random-walk for obstacle i."""
        sigma = self.mode_params[i].get("sigma", 0.1)
        vx_new = self.vx[i] + np.random.randn() * sigma
        vy_new = self.vy[i] + np.random.randn() * sigma
        return vx_new, vy_new

    def _step_sinusoid(self, i: int):
        """Sinusoidal motion for obstacle i."""
        mp = self.mode_params[i]
        amp = mp.get("amp", 1.0)
        freq = mp.get("freq", 0.5)
        phase = mp.get("phase", 0.0)

        phi = 2 * np.pi * freq * self.t + phase
        vx_i = amp * np.sin(phi)
        vy_i = amp * np.cos(phi)
        return vx_i, vy_i

    def _step_bounce(self, i: int):
        """
        Step-bounce motion along the x-axis for obstacle i.

        - bounces between xmin and xmax
        - flips direction when hitting a bound
        """
        mp = self.mode_params[i]
        xmin, xmax = mp["bounds"]
        speed = mp["speed"]

        # read+update current direction in-place:
        dir = mp.get("dir", -1)
        next_x = self.cx[i] - dir * speed * self.dt
        if next_x < xmin or next_x > xmax:
            dir *= -1
        mp["dir"] = dir

        return dir * speed, 0.0

    def _step_orbit(self, i: int):
        """
        Rotate around a center point at angular rate omega.
        Velocity v = omega x r, where r is the distance to the center of rotation.
        """
        mp = self.mode_params[i]
        omega = mp.get("omega", 1.0)
        cx0, cy0 = mp.get("center", (0.0, 0.0))

        dx = self.cx[i] - cx0
        dy = self.cy[i] - cy0

        vx_i = -omega * dy
        vy_i = omega * dx
        return vx_i, vy_i

    def current_positions(self):
        return list(zip(self.cx, self.cy))

    def predict_states(self, N: int):
        """
        Simulate N steps ahead (without committing) and return the
        predicted trajectories for x and y positions.

        After predicting, restores the original obstacle state.

        Returns:
            x_pred_flat : np.ndarray, shape ((N+1)*m,)
            y_pred_flat : np.ndarray, shape ((N+1)*m,)
        """
        prediction = {
            "cx": self.cx.copy(),
            "cy": self.cy.copy(),
            "vx": self.vx.copy(),
            "vy": self.vy.copy(),
            "t": self.t,
            "mode_params": copy.deepcopy(self.mode_params),
        }

        x_pred = np.zeros((N + 1, self.m))
        y_pred = np.zeros((N + 1, self.m))

        x_pred[0, :] = self.cx
        y_pred[0, :] = self.cy

        for k in range(1, N + 1):
            x_k, y_k = self.step()
            x_pred[k, :] = x_k
            y_pred[k, :] = y_k

        # restore from backup
        self.cx = prediction["cx"]
        self.cy = prediction["cy"]
        self.vx = prediction["vx"]
        self.vy = prediction["vy"]
        self.t = prediction["t"]
        self.mode_params = prediction["mode_params"]

        return x_pred.flatten(), y_pred.flatten()

    def reset(self):
        """
        Restore intial state of the obstacles.
        """
        st = self._init_state
        self.cx = st["cx"].copy()
        self.cy = st["cy"].copy()
        self.vx = st["vx"].copy()
        self.vy = st["vy"].copy()
        self.t = st["t"]
        self.mode_params = copy.deepcopy(st["mode_params"])
        return self.current_positions()
