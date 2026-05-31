import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def tanh(x):
    return np.tanh(x)

def sigmoid_grad(x):
    s = sigmoid(x)
    return (1.0 - s) * s


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def identity_grad(x):
    return np.ones_like(x)