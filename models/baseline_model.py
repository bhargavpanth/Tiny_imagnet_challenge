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



    ##
    network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')
    # Step 2: Max pooling
    network = max_pool_2d(network, 2)
    # Step 3: Convolution
    network = conv_2d(network, 64, 3, activation='relu')
    # Step 4: Convolution
    network = conv_2d(network, 64, 3, activation='relu')
    # Step 5: Max pooling
    network = max_pool_2d(network, 2)
    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')
    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)
    # Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
    network = fully_connected(network, 2, activation='softmax')
    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='baseline_cnn.tfl.ckpt')
    # Train it! We'll do 100 training passes and monitor it as it goes.
    model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=96,snapshot_epoch=True,run_id='baseline_cnn')
    # save the model results
    model.save("baseline_cnn.tfl")
    print("Network trained and saved as baseline_cnn.tfl")


    # ## Input shape will be [batch_size, height, width, channels].
    # network = input_data(shape=[None, 64, 64, 3],
    #                      data_preprocessing=img_prep,
    #                      data_augmentation=img_aug)
    # ## First convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    # network = conv_2d(network, 32, 5, activation='relu')
    # ## First batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # ## Pooling layer. 64x64x32 -> 32x32x32
    # network = max_pool_2d(network, 2)
    # ## Second convolution layer. 32 filters of size 5. Activation function ReLU. 32x32x32 -> 32x32x32
    # network = conv_2d(network, 32, 5, activation='relu')
    # ## Second batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # ## First fully connected layer. 32x32x32 -> 1x32768 -> 1x1024. ReLU activation.
    # network = fully_connected(network, 1024, activation='relu')
    # ## Third batch normalization layer
    # network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # ## Dropout layer for the first fully connected layer.
    # network = dropout(network, 0.5)
    # ## Second fully connected layer. 1x1024 -> 1x200. Maps to class labels. Softmax activation to get probabilities.
    # network = fully_connected(network, 200, activation='softmax')
    # ## Loss function. Softmax cross entropy. Adam optimization.
    # network = regression(network, optimizer='adam',
    #                      loss='categorical_crossentropy',
    #                      learning_rate=learning_rate)
    # model.save("baseline.tfl")
    # print("Network trained and saved as baseline.tfl")

    return network