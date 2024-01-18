from MyTorch import Model, Parameter
import numpy as np


class Dropout(Model):
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        if self.training:
            mask = np.random.binomial(1, self.p, size=x.shape) / self.p
            return x * mask
        else:
            return x