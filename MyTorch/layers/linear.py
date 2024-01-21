from MyTorch import Model, Parameter
import numpy as np
from MyTorch.operations import MatMul
from MyTorch import Parameter

class Linear(Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Parameter(np.empty((input_dim, output_dim)))
        self.bias = Parameter(np.empty((1, output_dim)))
        self.reset_parameters()
        
    def reset_parameters(self):
        # use Kaiming initialization
        self.weights.data = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2 / self.input_dim)
        self.bias.data = np.random.randn(1, self.output_dim) * np.sqrt(2 / self.input_dim)
        
    def forward(self, x):
        return MatMul()(x, self.weights) + self.bias
    