import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import math

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # Get image shape
    H, W = image[:, :, 0].shape

    # Convert RGB image into Lab color space
    image = skimage.color.rgb2lab(image)

    # Array of scale sizes for the filters, in unit pixels
    scale = [1, 2, 4, 8, 8 * math.sqrt(2)]

    # Allocate the output array of images
    output = np.zeros((H, W, 4 * len(scale) * 3))

    
    # ----- TODO -----
    for i in range(len(scale)):
        for j in range(3):
            # i for scale size, j for each Lab color channel

            # 3 channels have to be in the output array consecutively, thus the weird output image index 
            # (1) Gaussian
            output[:, :, (i * 12) + (0 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma = scale[i])

            # (2) Laplacian of Gaussian
            output[:, :, (i * 12) + (3 + j)] = scipy.ndimage.gaussian_laplace(image[:, :, j], sigma = scale[i])

            # (3) Derivative of Gaussian in the x direction
            # The 3rd argument in the gaussian_filter function is the derivative order in ? axis direction
            # scipy deals with arrays, not images, so (0, 1) is the direction of the change of the second index,
            # which if the second index changes, it is in the horizontal direction, thus the x axis
            output[:, :, (i * 12) + (6 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma = scale[i], order = (0, 1))

            # (4) Derivative of Gaussian in the y direction
            output[:, :, (i * 12) + (9 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma = scale[i], order = (1, 0))

    return output

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    
    pass


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    # ----- TODO -----
    
    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    
    pass


