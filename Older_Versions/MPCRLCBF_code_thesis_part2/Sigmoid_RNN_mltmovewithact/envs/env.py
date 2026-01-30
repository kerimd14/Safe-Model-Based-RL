import gymnasium as gym 
import numpy as np
import os # to communicate with the operating system
import copy
from gymnasium.spaces import Box
import casadi as cs
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
import matplotlib.pyplot as plt
from control import dlqr
from collections import deque
import pandas as pd
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from config import SAMPLING_TIME, NUM_INPUTS, NUM_STATES, CONSTRAINTS_X, SEED, CONSTRAINTS_U



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

