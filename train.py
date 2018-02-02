"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script trains a simple baseline_model on the tiny-imagenet dataset."""

import sys
import os
import argparse
import tflearn
import h5py

sys.path.insert(0, os.path.dirname(__file__))

from datasets.tiny_imagenet import *
from models.baseline_model import *

from tflearn.data_utils import shuffle

def get_data(data_dir, hdf5):
    """This function loads in the data, either by loading images on the fly or by creating and
    loading from a hdf5 database.

    Args:
        data_dir: Root directory of the dataset.
        hdf5: Boolean. If true, (create and) load data from a hdf5 database.

    Returns:
        X: training images.
        Y: training labels.
        X_test: validation images.
        Y_test: validation labels."""

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


def main(data_dir, hdf5, name):
    """This is the main function of the file.

    Args:
        data_dir: The root directory of the images.
        hdf5: Boolean if a hdf5 database should be created to load in the images.
        name: Name of the current training run."""

    # Set some variables for training.
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.001

    # Load in data.
    X, Y, X_test, Y_test = get_data(data_dir, hdf5)

    # Define some preprocessing options. These will be done on every image before either training or testing.
    img_prep = tflearn.data_preprocessing.ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Define some data augmentation options. These will only be done for training.
    img_aug = tflearn.data_augmentation.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    # Get the network definition.
    network = create_network(img_prep, img_aug, learning_rate)

    # Training. It will always save the best performing model on the validation data, even if it overfits.
    checkpoint_path = 'output/'+name+'/'
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard', best_checkpoint_path=checkpoint_path)
    model.fit(X, Y, n_epoch=num_epochs, shuffle=True, validation_set=(X_test, Y_test),
    show_metric=True, batch_size=batch_size, run_id=name)

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
