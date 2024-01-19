from MyTorch import Operation
from MyTorch import Tensor
import numpy as np

class ReLU(Operation):
    def forward(self, x):
        return Tensor(x.data * (x.data > 0))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * (self.input[0].data > 0))
        
class Sigmoid(Operation):
    def forward(self, x):
        return Tensor(1 / (1 + np.exp(-x.data)))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.output.data * (1 - self.output.data))
        
class Softmax(Operation):
    def __init__(self, dim = 1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        e_x = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
        return Tensor(e_x / np.sum(e_x, axis=self.dim, keepdims=True))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.output.data * (1 - self.output.data))