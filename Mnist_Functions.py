from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def load_mnist(flatten = True, normalize = True):

	(x_train, t_train), (x_test, t_test) = mnist.load_data()
	
	if normalize:
		x_train = x_train.astype(np.float32) / 255.0
		x_test = x_test.astype(np.float32) / 255.0
	
	if flatten:
		x_train = x_train.reshape(x_train.shape[0], -1)
		x_test = x_test.reshape(x_test.shape[0], -1)
	
	return (x_train, t_train), (x_test, t_test)


def img_show(img):
	
	if img.ndim == 1:
		img = img.reshape(28, 28)
	
	plt.imshow(img, cmap = 'gray')
	plt.axis('off')
	plt.show()