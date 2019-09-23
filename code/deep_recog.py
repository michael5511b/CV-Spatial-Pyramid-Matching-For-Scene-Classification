import numpy as np
import multiprocessing
import threading
import queue
import os, time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import multiprocessing as mp
import torch
import torch.nn
import torchvision.models as models
import scipy.spatial.distance


def build_recognition_system(vgg16, num_workers):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    """

    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----

    img_paths = train_data['files.npy']
    labels = train_data['labels.npy']
    N = len(img_paths)

    # Multiprocessing here somehow results in RAM insufficiency
    features = np.zeros((N, 4096))
    for i in range(N):
        args = i, img_paths[i], vgg16
        # moving tensor to numpy will break the graph and no gradients will be calculated
        # If no gradients are used in this tensor, simply detach
        features[i][:] = get_image_feature(args).detach().numpy()

    np.savez('trained_system_deep', features, labels)

    print(features.shape)
    pass


def evaluate_recognition_system_helper_func(args):
    i, test_img, vgg16, trained_feats, training_labels = args
    feat_args = i, test_img, vgg16

    # Pre-trained VGG16
    feat = get_image_feature(feat_args).detach().numpy()

    # Self-implemented deep feature extraction
    # feat = network_layers.extract_deep_feature(skimage.io.imread("../data/" + test_img), util.get_VGG16_weights())
    # feat = np.reshape(feat, (1, 4096))

    dist = distance_to_set(feat, trained_feats)
    ind_min = np.argmin(dist)
    evaluated_label = training_labels[ind_min]

    print("test_img:", test_img, "label = ", evaluated_label)

    return evaluated_label


def evaluate_recognition_system(vgg16, num_workers=2):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    """

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")
    # ----- TODO -----

    test_imgs = test_data['files.npy']
    test_labels = test_data['labels.npy']

    trained_feats = trained_system['arr_0.npy']
    training_labels = trained_system['arr_1.npy']

    args_list = []

    for i in range(len(test_imgs)):
        args = i, test_imgs[i], vgg16, trained_feats, training_labels
        args_list.append(args)

    # Parallel computing using pool
    pool = mp.Pool(1)
    evaluated_labels = pool.map(evaluate_recognition_system_helper_func, args_list)

    # Confusion Matrix
    conf = np.zeros((8, 8))
    for i in range(len(evaluated_labels)):
        conf[evaluated_labels[i]][test_labels[i]] = conf[evaluated_labels[i]][test_labels[i]] + 1

    accuracy = np.trace(conf) / np.sum(conf)

    return conf, accuracy


def preprocess_image(image):
    """
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    """

    # ----- TODO -----
    image = image.astype('float') / 255
    # VGG16 only takes image inputs the size of 224 x 224
    image = skimage.transform.resize(image, (224, 224))

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
    image = (image - mean) / std

    # We want the tensor shape to be (3, H, W), transposed from original, (H, W, 3)
    # (H, W, 3) -> (3, H, W)
    # (0, 1, 2) -> (2, 0, 1)
    image = np.transpose(image, (2, 0, 1))
    # tensor = torch.from_numpy(image).cuda()
    tensor = torch.from_numpy(image)
    return tensor


def get_image_feature(args):
    '''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.

	[output]
	* feat: evaluated deep feature
	'''

    i, image_path, vgg16 = args

    # ----- TODO -----
    image_path = "../data/" + image_path
    img = skimage.io.imread(image_path)

    tensor = preprocess_image(img)
    del img
    classifier = vgg16.classifier
    classifier_modified = torch.nn.Sequential(*list(classifier.children())[0:5])
    vgg16.classifier = classifier_modified

    # tensors are 4 dimensional, add one dimension
    feat = vgg16(tensor[np.newaxis, :, :, :])
    del tensor
    # print("Img ", i, " processed!")
    return feat


def distance_to_set(feature, train_features):
    """
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N, K)

	[output]
	* dist: numpy.ndarray of shape (N)
	"""

    # ----- TODO -----
    # Just the euclidian distance function
    dist = scipy.spatial.distance.cdist(feature, train_features, 'euclidean')
    return dist
