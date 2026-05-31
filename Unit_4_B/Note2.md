# May 31th, Diary 2

## Summary

In this week's learning, I modified most of my work via separating many many functions into independent documents and create a new directory named __Common__ to store them. This regulation of work is really important because it makes my work more ordered and helped me learned how a big project is organized. 

Well, besides this, I succeeded in creating a class named __Double_Network__, which is a complete *two-layer neural network* which contains self-learning based on forwarding and gradient descent.

Here is something I modified and learned:

## 1. Modification of load_mnist Function
I realized that I should include one-hot label to my load_mnist function so that the label could be expressed in one-hot form. Here's new function:
```
def to_one_hot(labels, num_classes=10):
	one_hot = np.zeros((labels.size, num_classes))
	one_hot[np.arange(labels.size), labels] = 1
	return one_hot

def load_mnist(flatten = True, normalize = True, one_hot_label = False):

	(x_train, t_train), (x_test, t_test) = mnist.load_data()
	
	if normalize:
		x_train = x_train.astype(np.float32) / 255.0
		x_test = x_test.astype(np.float32) / 255.0
	
	if flatten:
		x_train = x_train.reshape(x_train.shape[0], -1)
		x_test = x_test.reshape(x_test.shape[0], -1)

	if one_hot_label:
		t_train = to_one_hot(t_train)
		t_test = to_one_hot(t_test)
	
	return (x_train, t_train), (x_test, t_test)
```
## 2. Design of DoubleNet Class:
```
import sys, os
sys.path.append(os.pardir)
from Common.Math_Functions import numerical_gradient
from Common.Activation_Functions import sigmoid, softmax
from Common.Loss_Functions import cross_entropy_error
import numpy as np

class DoubleNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # Initialize Weight:
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] =  weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
```

## 3. Realization of Issues of Gradient Descent:
I realized __how slow Gradient Descent could be!!!__ It is because I tried to run this code:
```
import sys, os
sys.path.append(os.pardir)
import numpy as np
from Common.Mnist_Functions import load_mnist
from DoubleNet import DoubleNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

train_loss_list = []

# Parameters:
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = DoubleNet(input_size = 784, hidden_size = 50, output_size = 10)

# Iteration:
for i in range(iters_num):
    # Obtain Mini_Batch:
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Compute Gradient:
    grad = network.numerical_gradient(x_batch, t_batch)

    # Renew Parameters:
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record Loss:
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# Draw Loss Graph:
plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
```
It runs  for over 40 minutes and is still running. Eventually I use Ctrl+C to stop it. Then I begin to analyze why it is so slow, and the calculation result of parameter number is:

- __One Iteration: 39760 Parameters__
  - __W1 = 780*54 = 39200__
  - __b1 = 50__
  - __W2 = 50*10 = 500__
  - __b2 = 10__
- __Two Calculation for Loss: 39760*2 = 79520 Iterations__
- __All Iterations: 79520*1000 ≈ 80 million iterations__

Therefore, the time complexity of _Gradient Descent_ is __O(P) (P is parameter number)__. It's __*really slow*__. 

__This is why we need Back Propagation__. 