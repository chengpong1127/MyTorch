from .tensor import Tensor
from .parameter import Parameter
import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self.training = True
        
    @abstractmethod
    def forward(self, *args) -> Tensor:
        pass
    
    
class Model(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args) -> Tensor:
        return self.forward(*args)
    
    @abstractmethod
    def forward(self, *args) -> Tensor:
        pass
    
    def get_parameters(self) -> list:
        submodels = [m for m in self.__dict__.values() if isinstance(m, Model)]
        self_parameters = [p for p in self.__dict__.values() if isinstance(p, Parameter)]
        params = []
        for submodel in submodels:
            params += submodel.get_parameters()
        params += self_parameters
        return params
    
    def train(self):
        self.training = True
        for submodel in [m for m in self.__dict__.values() if isinstance(m, Model)]:
            submodel.train()
            
    def eval(self):
        self.training = False
        for submodel in [m for m in self.__dict__.values() if isinstance(m, Model)]:
            submodel.eval()
            
    def save(self, path):
        params = self.get_parameters()
        np.savez(path, *[p.data for p in params])
        
    def load(self, path):
        params = self.get_parameters()
        npzfile = np.load(path)
        arrays = [npzfile[key] for key in npzfile.files]
        for p, d in zip(params, arrays):
            if p.data.shape != d.shape:
                raise Exception('Shape mismatch: {} vs {}'.format(p.data.shape, d.shape))
            p.data = d


class Operation(Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args) -> Tensor:
        pass
    
    @abstractmethod
    def backward(self, *args):
        pass
    
    def __call__(self, *args) -> Tensor:
        tensor_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_args.append(arg)
            else:
                tensor_args.append(Tensor(arg, requires_grad=False))
        self.input = tensor_args
        self.output = self.forward(*self.input)
        self.output.grad_fn = self
        if np.isnan(self.output.data).any():
            raise Exception('Output have nan')
        return self.output