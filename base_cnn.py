# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import sys
import os
import argparse
import tflearn
import h5py

sys.path.insert(0, os.path.dirname(__file__))

from datasets.tiny_imagenet import *
from models.baseline_model import *

from tflearn.data_utils import shuffle
# import pickle


def get_data(data_dir, hdf5):

    # Get the filenames of the lists containing image paths and labels.
    train_file, val_file = build_dataset_index(data_dir)

    # Check if (creating and) loading from hdf5 database is desired.
    if hdf5:
        # Create folder to store dataset.
        if not os.path.exists('hdf5'):
            os.makedirs('hdf5')
        # Check if hdf5 databases already exist and create them if not.
        if not os.path.exists('hdf5/tiny-imagenet_train.h5'):
            from tflearn.data_utils import build_hdf5_image_dataset
            print ' Creating hdf5 train dataset.'
            build_hdf5_image_dataset(train_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_train.h5', categorical_labels=True, normalize=True)

        if not os.path.exists('hdf5/tiny-imagenet_val.h5'):
            from tflearn.data_utils import build_hdf5_image_dataset
            print ' Creating hdf5 val dataset.'
            build_hdf5_image_dataset(val_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_val.h5', categorical_labels=True, normalize=True)

        # Load training data from hdf5 dataset.
        h5f = h5py.File('hdf5/tiny-imagenet_train.h5', 'r')
        X = h5f['X']
        Y = h5f['Y']

        # Load validation data.
        h5f = h5py.File('hdf5/tiny-imagenet_val.h5', 'r')
        X_test = h5f['X']
        Y_test = h5f['Y']    

    # Load images directly from disk when they are required.
    else:
        from tflearn.data_utils import image_preloader
        X, Y = image_preloader(train_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
        X_test, Y_test = image_preloader(val_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)

    # Randomly shuffle the dataset.
    X, Y = shuffle(X, Y)

    return X, Y, X_test, Y_test

def main():
	batch_size = 256
    num_epochs = 10
    learning_rate = 0.001

    # Load in data.
    X, Y, X_test, Y_test = get_data(data_dir, hdf5)

	# Load the data set
	# X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

	# Shuffle the data
	X, Y = shuffle(X, Y)

	# Make sure the data is normalized
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Create extra synthetic training data by flipping, rotating and blurring the
	# images on our data set.
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	img_aug.add_random_blur(sigma_max=3.)

	# Define our network architecture:

	# Input is a 32x32 image with 3 color channels (red, green and blue)
	network = input_data(shape=[None, 32, 32, 3],
	                     data_preprocessing=img_prep,
	                     data_augmentation=img_aug)

	# Step 1: Convolution
	network = conv_2d(network, 32, 3, activation='relu')

	# Step 2: Max pooling
	network = max_pool_2d(network, 2)

	# Step 3: Convolution again
	network = conv_2d(network, 64, 3, activation='relu')

	# Step 4: Convolution yet again
	network = conv_2d(network, 64, 3, activation='relu')

	# Step 5: Max pooling again
	network = max_pool_2d(network, 2)

	# Step 6: Fully-connected 512 node neural network
	network = fully_connected(network, 512, activation='relu')

	# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
	network = dropout(network, 0.5)

	# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
	network = fully_connected(network, 2, activation='softmax')

	# Tell tflearn how we want to train the network
	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)

	# Wrap the network in a model object
	model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')

	# Train it! We'll do 100 training passes and monitor it as it goes.
	model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
	          show_metric=True, batch_size=96,
	          snapshot_epoch=True,
	          run_id='bird-classifier')

	# Save model when training is complete to a file
	model.save("bird-classifier.tfl")
	print("Network trained and saved as bird-classifier.tfl!")