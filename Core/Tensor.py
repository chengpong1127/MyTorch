import numpy as np
from Core.Operation import Operation, Add, Mul, Pow


np.set_printoptions(precision=3, suppress=True)
class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad: Tensor = None
        self.grad_fn: Operation = None
        self.requires_grad = requires_grad
        
    @property
    def shape(self):
        return self.data.shape
        
    def __add__(self, other):
        return Add()(self, other)
        
    def __sub__(self, other):
        return Add()(self, other * -1)
        
    def __mul__(self, other):
        return Mul()(self, other)
    
    def __truediv__(self, other):
        r_other = Pow()(other, -1)
        return Mul()(self, r_other)
    
    def __pow__(self, other):
        return Pow()(self, other)
    
    def __matmul__(self, other):
        return Mul()(self, other)
        
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
            
        if not isinstance(grad, Tensor):
            grad = Tensor(grad)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        if issubclass(type(self.grad_fn), Operation):
            self.grad_fn.backward(grad)
                
    def __str__(self):
        return f'Tensor({self.data}, shape = {self.data.shape})'
