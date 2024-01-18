from MyTorch import Operation
from abc import abstractmethod

class Loss(Operation):
    @abstractmethod
    def forward(self, y_pred, y_true):
        raise NotImplementedError