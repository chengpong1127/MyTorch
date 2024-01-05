import numpy as np

class Optimizer:
    def __init__(self) -> None:
        pass
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        raise NotImplementedError
    
    
    
    
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for p in self.parameters:
            grad = p.grad.data
            if grad.shape[0] != p.shape[0]:
                grad = grad.mean(axis=0)
                grad = grad[np.newaxis, :]
                
            p.data -= self.lr * grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None