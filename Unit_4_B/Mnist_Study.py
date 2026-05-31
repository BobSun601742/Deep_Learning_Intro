import sys, os
sys.path.append(os.pardir)
import numpy as np
from Common.Mnist_Functions import load_mnist
from DoubleNet import DoubleNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

train_loss_list = []

# Parameters:
iters_num = 100
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