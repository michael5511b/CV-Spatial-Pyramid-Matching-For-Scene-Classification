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
import imageio
from tempfile import TemporaryFile
all_filter_responses = []

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

    # Allocate the output filter response array of images
    filter_responses = np.zeros((H, W, 4 * len(scale) * 3))

    
    # ----- TODO -----
    for i in range(len(scale)):
        for j in range(3):
            # i for scale size, j for each Lab color channel

            # 3 channels have to be in the output array consecutively, thus the weird output image index 
            # (1) Gaussian
            filter_responses[:, :, (i * 12) + (0 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma=scale[i])

            # (2) Laplacian of Gaussian
            filter_responses[:, :, (i * 12) + (3 + j)] = scipy.ndimage.gaussian_laplace(image[:, :, j], sigma=scale[i])

            # (3) Derivative of Gaussian in the x direction
            # The 3rd argument in the gaussian_filter function is the derivative order in ? axis direction
            # scipy deals with arrays, not images, so (0, 1) is the direction of the change of the second index,
            # which if the second index changes, it is in the horizontal direction, thus the x axis
            filter_responses[:, :, (i * 12) + (6 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma=scale[i], order=(0, 1))

            # (4) Derivative of Gaussian in the y direction
            filter_responses[:, :, (i * 12) + (9 + j)] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma=scale[i], order=(1, 0))

    return filter_responses


def get_visual_words(image, dictionary):
    """
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    """

    # ----- TODO -----
    # dictionary (200, 60)
    H, W = image[:, :, 0].shape
    numOfPix = H * W
    eucliDist = scipy.spatial.distance.cdist(np.reshape(extract_filter_responses(image), (numOfPix, 60)), dictionary, 'euclidean')
    # numOfPix = 187500
    # euclidDist (187500, 200), euclidean distant of each pixel to each of the 200 visual words
    wordmap = np.zeros(numOfPix)
    min_euclid = np.zeros(numOfPix)
    for i in range(numOfPix):
        min_euclid[i] = min(eucliDist[i, :])
    for i in range(numOfPix):
        eucliDist_curr = eucliDist[i, :]
        word, = np.where(eucliDist_curr == min_euclid[i])
        wordmap[i] = word 

    # reshape back to original shape
    wordmap = np.reshape(wordmap, (H, W))
    return wordmap


def compute_dictionary_one_image(args):
    """
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    """

    i, alpha, image_path = args
    global all_filter_responses
    # ----- TODO -----

    image = imageio.imread("../data/" + image_path)
    filter_responses = extract_filter_responses(image)
    
    H, W = image[:, :, 0].shape
    numOfPix = H * W
    filter_responses_1D = np.reshape(filter_responses, (numOfPix, 60))
    
    rand_ind = np.random.permutation(numOfPix)
    
    rand_ind = rand_ind[0 : alpha]    
    
    filter_responses_random = filter_responses_1D[rand_ind, :]

    if i == 0:
        all_filter_responses = filter_responses_random
    else:
        np.concatenate((all_filter_responses, filter_responses_random), axis=1)
    pass


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    global all_filter_responses
    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----

    image_paths = train_data['files.npy']

    alpha = 200

    for i in range(len(image_paths)):
        args = i, alpha, image_paths[i]
        compute_dictionary_one_image(args)

    K = 200
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(all_filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy', dictionary)

    pass


