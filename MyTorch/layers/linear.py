from MyTorch import Model, Parameter
import numpy as np
from MyTorch.operations import MatMul
from MyTorch import Parameter

class Linear(Model):
    def __init__(self, input_dim, output_dim):
        self.weights = Parameter(np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim))
        self.bias = Parameter(np.random.randn(1, output_dim) * np.sqrt(2.0 / input_dim))
        
    def forward(self, x):
        return MatMul()(x, self.weights) + self.bias
    