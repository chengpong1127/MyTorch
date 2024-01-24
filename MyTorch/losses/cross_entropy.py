from .base_loss import Loss
import numpy as np

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        return -np.sum(y_true.data * np.log(y_pred_clipped)) / len(y_pred_clipped)
    
    def backward(self, grad):
        epsilon = 1e-15
        self.input[0].backward(grad.data * (-self.input[1].data / (self.input[0].data + epsilon)))