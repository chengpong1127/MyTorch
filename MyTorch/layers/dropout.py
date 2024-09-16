from mytorch import Model
import numpy as np


class Dropout(Model):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            if self.p == 0.0:
                return x
            mask = np.random.binomial(1, self.p, size=x.shape) / self.p
            return x * mask
        else:
            return x