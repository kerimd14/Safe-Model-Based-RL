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

#make this into a class   
        
class AdamOptimizer:
    """
    ADAM optimizer with internal state.
    """

    def __init__(self, dim: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.dim = int(dim)
        self.beta1 = float(beta1) #decay rate for the first moment
        self.beta2 = float(beta2) #decay rate for the second moment
        self.eps = float(eps) #small constant to avoid division by zero

        # ADAM state
        self.exp_avg = np.zeros(self.dim)
        self.exp_avg_sq = np.zeros(self.dim)
        self.iteration = 1

    def step(self, gradient, learning_rate):
        """
        Computes the update's change according to Adam algorithm.

        Args:
            gradient (array-like): raw gradient vector
            learning_rate (float or array-like): step size (can be vector)

        Returns:
            dtheta (np.ndarray): the computed parameter increment (delta theta)
        """
        g = np.asarray(gradient).flatten()
        lr = np.asarray(learning_rate).flatten()

        # update moments
        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * g #running first moment estimate (EMA of the gradient; tracks direction)
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (g * g) #running second moment estimate (EMA of squared gradient; tracks magnitude).

        # bias correction
        bias_correction1 = 1 - self.beta1 ** self.iteration
        bias_correction2 = 1 - self.beta2 ** self.iteration

        step_size = lr / bias_correction1
        denom = (np.sqrt(self.exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

        dtheta = -step_size * (self.exp_avg / denom)

        self.iteration += 1
        return dtheta