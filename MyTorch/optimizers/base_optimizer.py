
class BaseOptimizer:
    def __init__(self, parameters) -> None:
        self.parameters = parameters
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
    