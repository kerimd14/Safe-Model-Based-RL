# obstacles/obstacles.py
import casadi as cs
from dataclasses import dataclass
from typing import List, Tuple

from config import SAMPLING_TIME


@dataclass
class ObstacleConfig:
    positions: List[Tuple[float, float]]
    radii: List[float]
    modes: List[str]
    mode_params: List[dict]


class Obstacles:

    def __init__(self, positions, radii):

        """
        Initialize obstacles.

        Args:
            positions: List of (x, y) obstacle centers.
            radii:     List of obstacle radii.
        """
        if len(positions) != len(radii):
            raise ValueError(
                f"Expected same number of positions and radii, "
                f"but got {len(positions)} positions and {len(radii)} radii."
            )

        self.positions = positions
        self.radii = radii
        self.obstacle_num = len(positions)
        self.dt = SAMPLING_TIME

    def h_obsfunc(self, x, xpred_list, ypred_list):
        """
        [NOT USED IN CODE]

        Numerically evaluate CBF h_i(x) = (x - x_pred[k])^2 + (y - y_pred[k])^2 - r^2
        for each obstacle.

        Args:
            x:       Current state vector [x, y, …].
            xpred:   List of predicted obstacle x-positions (length m).
            ypred:   List of predicted obstacle y-positions (length m).

        Returns:
            A list of h_i(x) values (one per obstacle).
        """
        h_list = []
        for (r, x_pred, y_pred) in zip(self.radii, xpred_list, ypred_list):
            h_list.append((x[0] - x_pred) ** 2 + (x[1] - y_pred) ** 2 - r**2)
        return h_list

    def make_h_functions(self):
        """
        Build symbolic CasADi Functions for each obstacle h_i(x).

        Returns:
            List[cs.Function] with signature:
              h_i(x: MX[4], xpred_list: MX[m], ypred_list: MX[m]) -> MX[1]
        """
        funcs = []

        # states
        x = cs.MX.sym("x", 4)

        # predicted obstacle positions (you call them vx, vy but they are positions in your pipeline)
        xpred_list = cs.MX.sym("vx", self.obstacle_num)
        ypred_list = cs.MX.sym("vy", self.obstacle_num)

        for idx, r in enumerate(self.radii):
            xpred_i = xpred_list[idx]
            ypred_i = ypred_list[idx]

            hi_expr = (x[0] - xpred_i) ** 2 + (x[1] - ypred_i) ** 2 - r**2
            hi_fun = cs.Function(f"h{idx+1}", [x, xpred_list, ypred_list], [hi_expr])
            funcs.append(hi_fun)

        return funcs