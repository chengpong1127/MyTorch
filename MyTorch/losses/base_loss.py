from MyTorch import Operation


class Loss(Operation):
    def forward(self, y_pred, y_true):
        raise NotImplementedError