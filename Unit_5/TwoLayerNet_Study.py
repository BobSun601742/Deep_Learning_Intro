import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from Common.Mnist_Functions import load_mnist
from TwoLayerNet import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=True
)

train_loss_list = []

# Parameters:
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(
    input_size=784,
    hidden_size=50,
    output_size=10
)

# Iteration:
for i in range(iters_num):
    # Obtain Mini-Batch:
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Compute Gradient by backpropagation:
    grad = network.gradient(x_batch, t_batch)

    # Update Parameters:
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Record Loss:
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print(f"iter {i+1}/{iters_num}, loss: {loss}")


# Save Loss Graph:
plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
plt.close()

print("Loss graph saved as training_loss.png")