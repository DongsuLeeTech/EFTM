import torch
import numpy as np
from numpy import pi
import pandas as pd
import datetime

class Inverse_config:
    def __init__(self):
        self.SEED_NUMBER = 1004
        self.PI_STD = 0.4

        self.ADAM_LR = 1e-02
        self.ADAM_eps = 1e-8

        self.uns4IDM_range = [0, 1.]
        self.desvel_range = [0, 1.]
        self.mlp_range = [0, 1.]

        self.NUM_SAM = 1
        self.NUM_EP = 1
        self.NUM_IT = 2000
        self.NUM_coef = 3
