"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script gives the network definition."""

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def create_network(img_prep, img_aug, learning_rate):
    """
    Alexnet implementation
    
    # convolution layer
    # Max Pooling (down-sampling)
    # Apply Normalization
    # Apply Dropout
    """

    network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
    
    # First convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    conv1 = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    pool1 = max_pool_2d(conv1, 2)
    # First batch normalization layer
    norm1 = batch_normalization(pool1, stddev=0.002, trainable=True, restore=True, reuse=False)
    # dropout layer
    #norm1 = tf.nn.dropout(norm1, 1)
    norm1 = dropout(norm1, 1)


    # Second convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    conv2 = conv_2d(norm1, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    pool2 = max_pool_2d(conv2, 2)
    # First batch normalization layer
    norm2 = batch_normalization(pool2, stddev=0.002, trainable=True, restore=True, reuse=False)
    # dropout layer
    #norm2 = tf.nn.dropout(norm2, 1)
    norm2 = dropout(norm2, 1)


    # Second convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    conv3 = conv_2d(norm2, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    pool3 = max_pool_2d(conv3, 2)
    # First batch normalization layer
    norm3 = batch_normalization(pool3, stddev=0.002, trainable=True, restore=True, reuse=False)
    # dropout layer
    #norm3 = tf.nn.dropout(norm3, 1)
    norm3 = dropout(norm3, 1)

    
    return network



    # Input shape will be [batch_size, height, width, channels].
    # network = input_data(shape=[None, 64, 64, 3],
    #                      data_preprocessing=img_prep,
    #                      data_augmentation=img_aug)
    # # First convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    # network = conv_2d(network, 32, 5, activation='relu')
    # # First batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # # Pooling layer. 64x64x32 -> 32x32x32
    # network = max_pool_2d(network, 2)
    # # Second convolution layer. 32 filters of size 5. Activation function ReLU. 32x32x32 -> 32x32x32
    # network = conv_2d(network, 32, 5, activation='relu')
    # # Second batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # # First fully connected layer. 32x32x32 -> 1x32768 -> 1x1024. ReLU activation.
    # network = fully_connected(network, 1024, activation='relu')
    # # Third batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # # Dropout layer for the first fully connected layer.
    # network = dropout(network, 0.5)
    # # Second fully connected layer. 1x1024 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    # network = fully_connected(network, 200, activation='softmax')
    # # Loss function. Softmax cross entropy. Adam optimization.
    # network = regression(network, optimizer='adam',
    #                      loss='categorical_crossentropy',
    #                      learning_rate=learning_rate)
    # return network
    
    #network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
    # Step 1: Convolution
    #network = conv_2d(network, 32, 3, activation='relu')
    # Step 2: Max pooling
    #network = max_pool_2d(network, 2)
    # Step 3: Convolution
    #network = conv_2d(network, 64, 3, activation='relu')
    # Step 4: Convolution
    #network = conv_2d(network, 64, 3, activation='relu')
    # Step 5: Max pooling
    #network = max_pool_2d(network, 2)
    # Step 6: Fully-connected 512 node neural network
    #network = fully_connected(network, 512, activation='relu')
    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    #network = dropout(network, 0.5)
    # Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
    #network = fully_connected(network, 2, activation='softmax')
    # Tell tflearn how we want to train the network
    #network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
    
  
    # VGG Net
    
