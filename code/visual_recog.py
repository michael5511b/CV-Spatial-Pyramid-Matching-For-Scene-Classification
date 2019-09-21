import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import imageio
import time
def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----


    image_paths = train_data['files.npy']
    layer_num = 3

    K = len(dictionary)
    feat_size = int(K*(4 ** layer_num - 1) / 3)
    features = np.zeros((feat_size, len(image_paths)))

    print("Total number of training img: ", len(image_paths))

    for i in range(len(image_paths)):
        start = time.time()
        image = imageio.imread("../data/" + image_paths[i])
        wordmap = visual_words.get_visual_words(image, dictionary)
        hist = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
        features[:, i] = hist
        end = time.time()
        print("Training img no. ", i, "done!")
        print("time: ", end - start)


    labels = train_data['labels.npy']

    np.savez('trained_system', dictionary, features, labels, layer_num)

    pass


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----

    pass


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----

    pass


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    N, K = histograms.shape
    sim = np.zeros(N)
    for i in range(N):
        m = np.minimum(word_hist, histograms[:,i])
        s = sum(m)
        sim[i] = s

    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    H, W = wordmap.shape
    hist = np.zeros(dict_size)
    for i in range(H):
        for j in range(W):
            hist[int(wordmap[i][j])] = hist[int(wordmap[i][j])] + 1
    
    for k in range(dict_size):
        hist[k] = hist[k] / (H * W)
    #np.set_printoptions(suppress=True)
    #print(hist)
    return hist

def sum_sub_squares(CELLS):
    H, W, D = CELLS.shape
    #print("shape = ", CELLS.shape)
    new_CELL = np.zeros((int(H / 2), int(H / 2), D))
    I = 0
    J = 0
    if H == 2:
        new_CELL[0, 0, :] = CELLS[0,0,:] + CELLS[0,1,:] + CELLS[1,0,:] + CELLS[1,1,:]
    else:
        for i in range(H - 2 + 1, 2):
            for j in range(H - 2 + 1, 2):
                sum = np.zeros(D)
                for p in range(i, 2 + i):
                    for q in range(j, 2 + j):
                        sum = sum + CELLS[p, q, :]

                new_CELL[I,J,:] = sum / 4
                J = J + 1
            I = I + 1
    return new_CELL

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----

    # Index of the layers start from 0
    L = layer_num - 1
    # hist_size = np.zeros(dict_size * (4 ^ layer_num - 1) / 3)
    H, W = wordmap.shape
    hist_all = []
    last_layer = []
    prev_CELLS = []

    for i in range(layer_num):
        # Start from the finest layer to the more coarsed
        l = layer_num - i - 1

        # l layer has 4 ^ l cells
        # l layer has the sides of the image devided into 2 ^ l parts
        cell_sides = 2 ** l
        cell_r_len = int(H / cell_sides)
        cell_c_len = int(W / cell_sides)

        weight = 0

        if l == 0:
            weight = 2 ** (-L)
        else:
            weight = 2 ** (l - L -1)

        hist_layer = []
        CELLS = np.zeros((cell_sides, cell_sides, dict_size))
        # Can be faster if I aggregate the coarse layers from the finer layers
        
        if i == 0:
            for r in range(cell_sides):
                for c in range(cell_sides):
                    cell = wordmap[r * cell_r_len : (r + 1) * cell_r_len, c * cell_c_len : (c + 1) * cell_c_len]
                    hist = get_feature_from_wordmap(cell, dict_size)
                    CELLS[r, c, :] = hist
                    hist_layer = np.concatenate((hist_layer, hist), axis = 0)
        else:
            CELLS = sum_sub_squares(prev_CELLS)
            H, W, D = CELLS.shape
            for r in range(H):
                for c in range(H):
                    hist_layer = np.concatenate((hist_layer, CELLS[r, c, :]), axis = 0)
            
        prev_CELLS = CELLS
        #last_layer = hist_layer
        
        """
        for r in range(cell_sides):
                for c in range(cell_sides):
                    cell = wordmap[r * cell_r_len : (r + 1) * cell_r_len, c * cell_c_len : (c + 1) * cell_c_len]
                    hist = get_feature_from_wordmap(cell, dict_size)
                    CELLS[r, c, :] = hist
                    hist_layer = np.concatenate((hist_layer, hist), axis = 0)
        """
        hist_layer = [x * weight for x in hist_layer]
        if len(hist_all) == 0:
            hist_all = hist_layer
        else:
            hist_all = np.concatenate((hist_all, hist_layer), axis = 0)
        

    SUM = sum(hist_all)
    hist_all = [x / SUM for x in hist_all]
    #print(len(hist_all))


    
    return hist_all