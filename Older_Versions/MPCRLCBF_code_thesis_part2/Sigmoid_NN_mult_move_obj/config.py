""""
Configuration file for the application.
This file contains constants and settings used throughout the application."
"""
import numpy as np


SAMPLING_TIME = 0.2

NUM_STATES = 4
NUM_INPUTS = 2

CONSTRAINTS_X = np.array([
    5, # x1 bounds
    5, # x2 bounds
    5,  # x3 bounds
    5   # x4 bounds
])
# CONSTRAINTS_X = 5
CONSTRAINTS_U = 1

SEED = 69



