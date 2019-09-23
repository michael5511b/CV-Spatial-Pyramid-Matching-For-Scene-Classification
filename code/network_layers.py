import numpy as np
import scipy.ndimage
import os
import skimage.transform


def extract_deep_feature(x, vgg16_weights):
	"""
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H, W, 3)
	* vgg16_weights: list of shape (L, 3)

	[output]
	* feat: numpy.ndarray of shape (K)
	"""

	# Standardization of image (Although in the problem it was mentioned as "Normalization)
	# Using mean and standard deviation for standardization
	# will make the mean of the data 0, and the standard deviation 1
	x = x.astype('float') / 255
	# VGG16 only takes image inputs the size of 224 x 224
	x = skimage.transform.resize(x, (224, 224))

	mean = [0.485, 0.456, 0.406]
	# make mean list into np array
	mean = np.array(mean)
	# reshape so that each mean lines up with corresponding 3 channels of the image
	np.reshape(mean, (1, 1, 3))

	std = [0.229, 0.224, 0.225]
	std = np.array(std)
	np.reshape(std, (1, 1, 3))

	# If an np array A = [1, 2, 3]
	# A + 1 will result in [2, 3, 4]
	# thus we can write in this form
	x = (x - mean) / std

	# vgg16_weights: L x 3
	# L = number of layers in vgg16
	# vgg16_weights[i][0] : First column is a string indicating what type of layer this is
	# There are 4 types of layers:
	# (1) Convolution
	# (2) ReLU (Rectified Linear Unit)
	# (3) Max pooling
	# (4) Linear (Fully-Connected (fc))

	# vgg16_weights[i][1] = weight if convolution of linear, kernel size if the layer is a max pooling layer
	# vgg16_weights[i][2] = bias if convolution or linear

	L = len(vgg16_weights)
	for i in range(34):
		print("layer: ", i, " || ", vgg16_weights[i][0])
		if vgg16_weights[i][0] == 'conv2d':
			weight = vgg16_weights[i][1]
			bias = vgg16_weights[i][2]
			x = multichannel_conv2d(x, weight, bias)

		elif vgg16_weights[i][0] == 'relu':
			x = relu(x)

		elif vgg16_weights[i][0] == 'maxpool2d':
			kernel_size = vgg16_weights[i][1]
			x = max_pool2d(x, kernel_size)

		else:
			weight = vgg16_weights[i][1]
			bias = vgg16_weights[i][2]
			x = linear(x, weight, bias)

	return x


def multichannel_conv2d(x, weight, bias):
	"""
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H, W, input_dim), input_dim = K
	* weight: numpy.ndarray of shape (output_dim = J, input_dim = K, kernel_size, kernel_size)
	* bias: numpy.ndarray of shape (output_dim = J)

	[output]
	* feat: numpy.ndarray of shape (H, W, output_dim)
	"""
	# y(j) = sigma(k = 1, K) [x(k) * h(j,k)] + b[j]
	H = x.shape[0]
	W = x.shape[1]
	J = weight.shape[0]
	K = weight.shape[1]
	weight = np.flip(weight, (2, 3))
	feat = np.zeros((H, W, J))
	for j in range(J):
		xh = np.zeros((H, W))
		for k in range(K):
			xh = xh + scipy.ndimage.convolve(x[:, :, k], weight[j, k, :, :], mode='constant', cval=0.0)
		feat[:, :, j] = xh + bias[j]
	return feat


def relu(x):
	"""
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	"""
	y = np.maximum(x, 0)
	return y



def max_pool2d(x, size):
	"""
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H, W, input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size, W/size, input_dim)
	"""
	H = x.shape[0]
	W = x.shape[1]
	D = x.shape[2]
	y = np.zeros((int(H / size), int(W / size), D))
	for i in range(int(H / size)):
		for j in range(int(W / size)):
			pool = x[i * size:(i + 1) * size, j * size:(j + 1) * size, :]
			pool = np.reshape(pool, (-1, D)) # [size * size, D]
			pool = np.transpose(pool)	# [D, size * size]
			p_max = np.max(pool, axis=1) # [D, 1(max of pool)]
			y[i, j, :] = p_max

	return y


def linear(x, W, b):
	"""
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	"""
	# # y[j] = sigma(k = 1, K)[W[j,k]x[k]] + b[j]
	# x = np.reshape(x, (-1, 1))
	# y = np.dot(W, x) + np.reshape(b, (b.shape[0], 1))

	try:
		x = x.transpose(2, 0, 1)
	except:
		pass
	x = x.reshape((-1))
	y = np.dot(W, x) + b

	return y

