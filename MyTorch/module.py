from .tensor import Tensor
from .parameter import Parameter
from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self.training = True
        
    @abstractmethod
    def forward(self, *args):
        pass
    
    
class Model(Module):
    def __init__(self):
        pass
    
    def __call__(self, *args):
        return self.forward(*args)
    
    @abstractmethod
    def forward(self, *args):
        pass
    
    def get_parameters(self):
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


class Operation(Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args):
        pass
    
    @abstractmethod
    def backward(self, *args):
        pass
    
    def __call__(self, *args):
        tensor_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_args.append(arg)
            else:
                tensor_args.append(Tensor(arg, requires_grad=False))
        self.input = tensor_args
        self.output = self.forward(*self.input)
        self.output.grad_fn = self
        return self.output