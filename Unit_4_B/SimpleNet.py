import sys, os
sys.path.append(os.pardir)
import numpy as np
from Common.Math_Functions import numerical_gradient
from Common.Loss_Functions import cross_entropy_error
from Common.Activation_Functions import softmax

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # Initialization Using Gaussian Distribution

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

