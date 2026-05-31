import numpy as np

def mean_square_error(y, t):
	if y.ndim == 1:
		return 0.5 * np.sum((y-t)**2)
	batch_size = y.shape[0]
	return 0.5 * np.sum((y-t)**2) / batch_size

def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size