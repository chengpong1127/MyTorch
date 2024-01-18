from MyTorch import Tensor, Operation
import numpy as np
    
class Add(Operation):
    def forward(self, x, y):
        return Tensor(x.data + y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data)
        self.input[1].backward(grad.data)
    
class Mul(Operation):
    def forward(self, x, y):
        return Tensor(x.data * y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data)
        self.input[1].backward(grad.data * self.input[0].data)
        
class Pow(Operation):
    def forward(self, x, y):
        return Tensor(x.data ** y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data * (self.input[0].data ** (self.input[1].data - 1)))
        self.input[1].backward(grad.data * np.log(self.input[0].data) * self.input[0].data ** self.input[1].data)
        
class Log(Operation):
    def forward(self, x):
        return Tensor(np.log(x.data))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * (1 / self.input[0].data))
        
class MatMul(Operation):
    def forward(self, x, y):
        return Tensor(np.matmul(x.data, y.data))
    
    def backward(self, grad):
        self.input[0].backward(grad.data @ self.input[1].data.T)
        self.input[1].backward(np.matmul(self.input[0].data.T, grad.data))
        
class Flatten(Operation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return Tensor(x.data.reshape(x.shape[0], -1))
    
    def backward(self, grad):
        self.input[0].backward(grad.data.reshape(self.input[0].shape))