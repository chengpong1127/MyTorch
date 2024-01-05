import numpy as np
import Core.Tensor as t
from Core.Operation import Operation, Add, Mul, Pow, MatMul

class Model:
    def __init__(self):
        pass
    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError
    
    def get_parameters(self):
        submodels = [ m for m in self.__dict__.values() if isinstance(m, Model) ]
        self_parameters = [ p for p in self.__dict__.values() if isinstance(p, t.Tensor) ]
        params = []
        for submodel in submodels:
            params += submodel.get_parameters()
        params += self_parameters
        return params


class Linear(Model):
    def __init__(self, input_dim, output_dim):
        self.weights = t.Tensor(np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim))
        self.bias = t.Tensor(np.random.randn(1, output_dim) * np.sqrt(2.0 / input_dim))
        
    def forward(self, x):
        return MatMul()(x, self.weights) + self.bias
    
    