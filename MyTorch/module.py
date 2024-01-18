from .tensor import Tensor
from .parameter import Parameter

class Module:
    def __init__(self):
        pass
        
    def forward(self, *args):
        raise NotImplementedError
    
    
class Model(Module):
    def __init__(self):
        pass
    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError
    
    def get_parameters(self):
        submodels = [ m for m in self.__dict__.values() if isinstance(m, Model) ]
        self_parameters = [ p for p in self.__dict__.values() if isinstance(p, Parameter) ]
        params = []
        for submodel in submodels:
            params += submodel.get_parameters()
        params += self_parameters
        return params

class Operation(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError
    
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