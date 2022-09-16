#!/bin/env python3.8

"""
Allister Liu
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import flags
from tqdm import trange

FLAGS = flags.FLAGS
flags.DEFINE_integer("sample_size", 1000, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 32, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_integer("random_seed", 31415, "Random seed")


def import_data(rng):
    # import data from csv to pandas dataframe
    data_df = pd.read_csv("./train.csv")
    # put data into numpy array
    data_arr = np.array(data_df)
    # shuffle the data
    rng.shuffle(data_arr)
    # separate labels and image data
    labels = data_arr[:, 0]     # shape=(42000,) the first column is the label
    pixels = data_arr[:, 1:]    # shape=(42000, 784) other columns are grayscale values of each pixel

    return pixels, labels


def preprocess(pixels, labels):
    # normalize the pixel grayscale value to between 0 and 1 by dividing by 255.
    pixels_normalized = pixels / 255.
    # reshape the 1d array of each image to 2d (784,) => (28, 28)
    pixels_processed = np.array([np.reshape(xs, (28, 28)) for xs in pixels_normalized])

    # one-hot encode the class labels
    labels_onehot = np.zeros((len(labels), labels.max()+1))     # labels.max()+1 = number of classes => 10 classes
    labels_onehot[np.arange(labels.size), labels] = 1

    # split the data => 80% train + 10% validation + 10% test
    train_range = range(0, int(0.8 * len(labels)))
    val_range = range(int(0.8 * len(labels)), int(0.9 * len(labels)))
    test_range = range(int(0.9 * len(labels)), len(labels))

    # print(f"Train:\t{train_range}\n"
    #       f"Val:\t{val_range}\n"
    #       f"Test:\t{test_range}\n")

    train_pix_arr = pixels_processed[train_range]   # shape=(33600, 28, 28)
    train_lbl_arr = labels_onehot[train_range]      # shape=(33600, 10)
    val_pix_arr = pixels_processed[val_range]       # shape=(4200 , 28, 28)
    val_lbl_arr = labels_onehot[val_range]          # shape=(4200 , 10)
    test_pix_arr = pixels_processed[test_range]     # shape=(4200 , 28, 28)
    test_lbl_arr = labels_onehot[test_range]        # shape=(4200 , 10)

    print(f"Train x:\t{train_pix_arr.shape}\tTrain y:\t{train_lbl_arr.shape}\n"
          f"Val x:\t\t{val_pix_arr.shape}\t\tVal y:\t\t{val_lbl_arr.shape}\n"
          f"Test x:\t\t{test_pix_arr.shape}\t\tTest y:\t\t{test_lbl_arr.shape}\n")

    # put data into Tensors
    train_df = tf.data.Dataset.from_tensor_slices((train_lbl_arr, train_pix_arr))
    val_df = tf.data.Dataset.from_tensor_slices((val_lbl_arr, val_pix_arr))
    test_df = tf.data.Dataset.from_tensor_slices((test_lbl_arr, test_pix_arr))
    print(train_df)


if __name__ == "__main__":
    # Handle the flags
    FLAGS(sys.argv)
    SAMPLE_SIZE = FLAGS.sample_size
    BATCH_SIZE = FLAGS.batch_size
    NUM_ITERS = FLAGS.num_iters
    RNG_SEED = FLAGS.random_seed

    # Set rng seed
    np_rng = np.random.default_rng(RNG_SEED)
    tf.random.Generator.from_seed(RNG_SEED)

    x, y = import_data(rng=np_rng)
    preprocess(pixels=x, labels=y)
