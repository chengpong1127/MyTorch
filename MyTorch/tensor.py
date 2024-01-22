import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=True):
        from .module import Operation
        self.data = np.array(data, dtype=np.float32)
        self.grad: Tensor = None
        self.grad_fn: Operation = None
        self.requires_grad = requires_grad
        
    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, *shape):
        from .operations import Reshape
        return Reshape(*shape)(self)
    
    def transpose(self, *dims):
        from .operations import Transpose
        return Transpose(dims)(self)
    
    def pad(self, pad_width):
        from .operations import Pad
        return Pad(pad_width)(self)
        
    def __add__(self, other):
        from .operations import Add
        return Add()(self, other)
        
    def __sub__(self, other):
        from .operations import Add
        return Add()(self, other * -1)
        
    def __mul__(self, other):
        from .operations import Mul
        return Mul()(self, other)
    
    def __truediv__(self, other):
        from .operations import Mul, Pow
        r_other = Pow()(other, -1)
        return Mul()(self, r_other)
    
    def __pow__(self, other):
        from .operations import Pow
        return Pow()(self, other)
    
    def __matmul__(self, other):
        from .operations import MatMul
        return MatMul()(self, other)
    
    def __getitem__(self, index):
        from .operations import Index
        return Index(index)(self)
        
    def backward(self, grad=None):
        from .module import Operation
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