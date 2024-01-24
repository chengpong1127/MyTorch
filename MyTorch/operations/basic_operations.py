from MyTorch import Tensor, Operation
import numpy as np
    
class Add(Operation):
    def forward(self, x, y):
        return Tensor(x.data + y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data)
        self.input[1].backward(grad.data)
    
class Mul(Operation):
    def forward(self, x, y):
        return Tensor(x.data * y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data)
        self.input[1].backward(grad.data * self.input[0].data)
        
class Pow(Operation):
    def forward(self, x, y):
        return Tensor(x.data ** y.data)
    
    def backward(self, grad):
        self.input[0].backward(grad.data * self.input[1].data * (self.input[0].data ** (self.input[1].data - 1)))
        self.input[1].backward(grad.data * np.log(self.input[0].data) * self.input[0].data ** self.input[1].data)
        
class Log(Operation):
    def forward(self, x):
        return Tensor(np.log(x.data))
    
    def backward(self, grad):
        self.input[0].backward(grad.data * (1 / self.input[0].data))
        
class MatMul(Operation):
    def forward(self, x, y):
        return Tensor(np.matmul(x.data, y.data))
    
    def backward(self, grad):
        transpose_axes0 = list(range(len(self.input[0].shape)))
        transpose_axes0[-2], transpose_axes0[-1] = transpose_axes0[-1], transpose_axes0[-2]
        transpose_axes1 = list(range(len(self.input[1].shape)))
        transpose_axes1[-2], transpose_axes1[-1] = transpose_axes1[-1], transpose_axes1[-2]
        
        self.input[0].backward(np.matmul(grad.data, self.input[1].data.transpose(transpose_axes1)))
        self.input[1].backward(np.matmul(self.input[0].data.transpose(transpose_axes0), grad.data))
        
class Flatten(Operation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return Tensor(x.data.reshape(x.shape[0], -1))
    
    def backward(self, grad):
        self.input[0].backward(grad.data.reshape(self.input[0].shape))
        
class Reshape(Operation):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return Tensor(x.data.reshape(self.shape))
    
    def backward(self, grad):
        self.input[0].backward(grad.data.reshape(self.input[0].shape))
        
class Transpose(Operation):
    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes
        
    def forward(self, x):
        return Tensor(x.data.transpose(self.axes))
    
    def backward(self, grad):
        self.input[0].backward(grad.data.transpose(self.axes))
        
class Index(Operation):
    def __init__(self, index):
        super().__init__()
        if isinstance(index, int) or isinstance(index, slice) or isinstance(index, list) or isinstance(index, tuple):
            self.index = index
        elif issubclass(type(index), Tensor):
            self.index = index.data.astype(int)
        else:
            raise Exception('Invalid index type')
        
    def forward(self, x):
        return Tensor(x.data[self.index])
    
    def backward(self, grad):
        _grad = np.zeros_like(self.input[0].data)
        grad_data = grad.data
        if len(grad_data.shape) > len(self.input[0].shape):
            grad_data = np.mean(grad_data, axis=tuple(range(len(grad_data.shape) - len(self.input[0].shape))))
        _grad[self.index] = grad_data
        self.input[0].backward(_grad)
        
class Pad(Operation):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width
        
    def forward(self, x):
        return Tensor(np.pad(x.data, self.pad_width))
    
    def backward(self, grad):
        self.input[0].backward(grad.data)