from MyTorch import Model, Parameter
import numpy as np
from MyTorch.operations import MatMul
from MyTorch import Parameter

class LayerNorm(Model):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(np.ones(input_dim))
        self.beta = Parameter(np.zeros(input_dim))
        self.eps = eps
        
    def forward(self, x):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        std = np.std(x.data, axis=-1, keepdims=True)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta
    
class BatchNorm(Model):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        self.gamma = Parameter(np.ones(input_dim))
        self.beta = Parameter(np.zeros(input_dim))
        self.eps = eps
        
    def forward(self, x):
        mean = np.mean(x.data, axis=0, keepdims=True)
        std = np.std(x.data, axis=0, keepdims=True)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta