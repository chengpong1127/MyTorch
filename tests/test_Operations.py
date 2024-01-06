import pytest

from Core.Tensor import Tensor
from Core.Operation import Add, Mul, Pow, Relu, Sigmoid, Log, MatMul, Softmax, Flatten

import numpy as np


def test_add():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x + y
    assert np.array_equal(z.data, np.array([5, 7, 9]))
    z.backward()
    assert np.array_equal(x.grad.data, np.array([1, 1, 1]))
    assert np.array_equal(y.grad.data, np.array([1, 1, 1]))
    
    
def test_sub():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x - y
    assert np.array_equal(z.data, np.array([-3, -3, -3]))
    z.backward()
    assert np.array_equal(x.grad.data, np.array([1, 1, 1]))
    assert np.array_equal(y.grad.data, np.array([-1, -1, -1]))
    
    
def test_mul():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x * y
    assert np.array_equal(z.data, np.array([4, 10, 18]))
    z.backward()
    assert np.array_equal(x.grad.data, np.array([4, 5, 6]))
    assert np.array_equal(y.grad.data, np.array([1, 2, 3]))
    

def test_div():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x / y

    assert np.array_equal(z.data, np.array([1/4.0, 2/5.0, 3/6.0], dtype=np.float32))

    z.backward()
    
    assert np.array_equal(x.grad.data, np.array([1/4.0, 1/5.0, 1/6.0], dtype=np.float32))
    assert np.array_equal(y.grad.data, np.array([-1/16.0, -2/25.0, -3/36.0], dtype=np.float32))
    
def test_pow():
    x = Tensor([1, 2, 3])
    y = 2
    
    z = x ** y
    assert np.array_equal(z.data, np.array([1, 4, 9]))
    z.backward()
    assert np.array_equal(x.grad.data, np.array([2, 4, 6]))
    
    
def test_relu():
    x = Tensor([1, -2, 3])
    z = Relu()(x)
    assert np.array_equal(z.data, np.array([1, 0, 3]))
    z.backward()
    assert np.array_equal(x.grad.data, np.array([1, 0, 1]))
    
    
def test_matmul():
    x = Tensor([[1, 2, 3]])
    y = Tensor([[4, 5, 6], [7, 8, 9], [10, 11, 12]])
    z = MatMul()(x, y)
    assert np.array_equal(z.data, np.array([[48, 54, 60]]))
    
    z.backward()
    assert np.array_equal(x.grad.data, np.array([[15, 24, 33]]))
    assert np.array_equal(y.grad.data, np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    
def test_Flatten():
    x = Tensor([[[1, 2, 3], [4, 5, 6]]])
    z = Flatten()(x)
    assert np.array_equal(z.data, np.array([[1, 2, 3, 4, 5, 6]]))
    
    