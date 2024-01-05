import numpy as np
import Core.Tensor as t
from Core.Operation import Operation

class Loss(Operation):
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
class MSE(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return t.Tensor(np.mean((y_pred.data - y_true.data) ** 2))
    
    def backward(self, grad=None):
        if grad is None:
            grad = t.Tensor(np.ones_like(self.input[0].data))
        self.input[0].backward(2 * grad.data * (self.input[0].data - self.input[1].data))
        self.input[1].backward(-2 * grad.data * (self.input[0].data - self.input[1].data))
        