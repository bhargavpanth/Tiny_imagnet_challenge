"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script gives the network definition."""

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def create_network(img_prep, img_aug, learning_rate):
    """This function defines the network structure.

    Args:
        img_prep: Preprocessing function that will be done to each input image.
        img_aug: Data augmentation function that will be done to each training input image.

    Returns:
        The network."""

    # Input shape will be [batch_size, height, width, channels].
    network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    # First convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    network = conv_2d(network, 32, 5, activation='relu')
    # First batch normalization layer
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)
    # Second convolution layer. 32 filters of size 5. Activation function ReLU. 32x32x32 -> 32x32x32
    network = conv_2d(network, 32, 5, activation='relu')
    # Second batch normalization layer
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # First fully connected layer. 32x32x32 -> 1x32768 -> 1x1024. ReLU activation.
    network = fully_connected(network, 1024, activation='relu')
    # Third batch normalization layer
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # Dropout layer for the first fully connected layer.
    network = dropout(network, 0.5)
    # Second fully connected layer. 1x1024 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    network = fully_connected(network, 200, activation='softmax')
    # Loss function. Softmax cross entropy. Adam optimization.
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network