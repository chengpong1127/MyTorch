from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self, parameters) -> None:
        self.parameters = parameters
    
    @abstractmethod
    def step(self):
        pass
    
    def zero_grad(self):
        for p in self.parameters:
            del p.grad
            p.grad = None
    