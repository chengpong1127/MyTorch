from . import Tensor as t
import numpy as np

class Operation:
    def __init__(self):
        self.input: t.Tensor = None
        self.output: t.Tensor = None
        
    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        tensor_args = []
        for arg in args:
            if isinstance(arg, t.Tensor):
                tensor_args.append(arg)
            else:
                tensor_args.append(t.Tensor(arg, requires_grad=False))
        self.input = tensor_args
        self.output = self.forward(*self.input)
        self.output.grad_fn = self
        return self.output
    
    
class Add(Operation):
    def forward(self, x, y):
        return t.Tensor(x.data + y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad)
        self.input[1].backward(grad)
    
class Mul(Operation):
    def forward(self, x, y):
        return t.Tensor(x.data * y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data)
        self.input[1].backward(grad.data * self.input[0].data)
        
class Pow(Operation):
    def forward(self, x, y):
        return t.Tensor(x.data ** y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data * (self.input[0].data ** (self.input[1].data - 1)))
        self.input[1].backward(grad.data * np.log(self.input[0].data) * self.input[0].data ** self.input[1].data)
        
class Relu(Operation):
    def forward(self, x):
        return t.Tensor(x.data * (x.data > 0))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * (self.input[0].data > 0))
        
class Sigmoid(Operation):
    def forward(self, x):
        return t.Tensor(1 / (1 + np.exp(-x.data)))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.output.data * (1 - self.output.data))
        
class Log(Operation):
    def forward(self, x):
        return t.Tensor(np.log(x.data))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * (1 / self.input[0].data))
        
class MatMul(Operation):
    def forward(self, x, y):
        return t.Tensor(np.matmul(x.data, y.data))
    
    def backward(self, grad):
        self.input[0].backward(grad.data @ self.input[1].data.T)
        self.input[1].backward(np.matmul(self.input[0].data.T, grad.data))
        
class Softmax(Operation):
    def forward(self, x):
        e_x = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
        return t.Tensor(e_x / np.sum(e_x, axis=1, keepdims=True))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.output.data * (1 - self.output.data))