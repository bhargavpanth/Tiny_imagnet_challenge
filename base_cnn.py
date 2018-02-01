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
	train_file, val_file = build_dataset_index(data_dir)

	if hdf5:
		if not os.path.exists('hdf5'):
			os.makedirs('hdf5')
		if not os.path.exists('hdf5/tiny-imagenet_train.h5'):
			from tflearn.data_utils import build_hdf5_image_dataset
			build_hdf5_image_dataset(train_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_train.h5', categorical_labels=True, normalize=True)
		if not os.path.exists('hdf5/tiny-imagenet_val.h5'):
			from tflearn.data_utils import build_hdf5_image_dataset
			build_hdf5_image_dataset(val_file, image_shape=(64, 64), mode='file', output_path='hdf5/tiny-imagenet_val.h5', categorical_labels=True, normalize=True)

		h5f = h5py.File('hdf5/tiny-imagenet_train.h5', 'r')
		X = h5f['X']
		Y = h5f['Y']

		h5f = h5py.File('hdf5/tiny-imagenet_val.h5', 'r')
		X_test = h5f['X']
		Y_test = h5f['Y']
	else:
		from tflearn.data_utils import image_preloader
		X, Y = image_preloader(train_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)
		X_test, Y_test = image_preloader(val_file, image_shape=(64, 64), mode='file', categorical_labels=True, normalize=True, filter_channel=True)

	# Randomly shuffle the dataset.
	X, Y = shuffle(X, Y)
	return X, Y, X_test, Y_test

def main(data_dir, hdf5, name):
	batch_size = 256
	num_epochs = 10
	learning_rate = 0.001
	X, Y, X_test, Y_test = get_data(data_dir, hdf5)
	X, Y = shuffle(X, Y)
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	img_aug.add_random_blur(sigma_max=3.)
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
	model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')
	# Train it! We'll do 100 training passes and monitor it as it goes.
	model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=96,snapshot_epoch=True,run_id='bird-classifier')
	# Save model when training is complete to a file
	model.save("bird-classifier.tfl")
	print("Network trained and saved as bird-classifier.tfl!")

if __name__ == '__main__':
    # Parse arguments and create output directories.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/tiny-imagenet-200',
                        help='Directory in which the input data is stored.')
    parser.add_argument('--hdf5',
                        help='Set if hdf5 database should be created.',
                        action='store_true')
    parser.add_argument('--name', type=str,
                        default='default',
                        help='Name of this training run. Will store results in output/[name]')
    args, unparsed = parser.parse_known_args()
    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    if not os.path.exists('output'):
        os.makedirs('output')
    main(args.data_dir, args.hdf5, args.name)

# if __name__ == '__main__':
# 	main()