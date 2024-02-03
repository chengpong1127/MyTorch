# MyTorch

## 1. Introduction

This is an implementation of a neural network using only numpy. The purpose of this project is to understand the basic concepts of neural networks and to implement them from scratch. The project is inspired by the PyTorch library, and the implementation is similar to PyTorch.

## 2. Framework

There are five main parts of the framework:
- `Tensor`: The main data structure of the framework. It is a multi-dimensional array with support for autograd operations.
- `Function`: The base class for all operations in the framework. It has a forward and backward method to compute the output and the gradient of the operation.
- `Module`: The base class for all layers in the framework. It has a forward and backward method to compute the output and the gradient of the layer.
- `Loss`: The base class for all loss functions in the framework. It has a forward and backward method to compute the output and the gradient of the loss.
- `Optimizer`: The base class for all optimizers in the framework. It has a step method to update the parameters of the model.

