from .base_optimizer import BaseOptimizer
import numpy as np
    
class SGD(BaseOptimizer):
    def __init__(self, parameters, lr=0.001) -> None:
        super().__init__(parameters)
        self.lr = lr
    
    def step(self):
        for p in self.parameters:
            grad = p.grad.data
            while(len(grad.shape) > len(p.shape)):
                grad = grad.sum(axis=0)
                
            if grad.shape[0] != p.shape[0] and p.shape[0] == 1:
                grad = grad.mean(axis=0)
                
            p.data -= self.lr * grad
    
    