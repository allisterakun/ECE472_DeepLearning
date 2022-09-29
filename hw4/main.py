#!/bin/env python3.9

"""
Allister Liu
"""

import sys

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pickle
import tarfile
import pydot

from absl import flags
import keras.layers
from keras.callbacks import LearningRateScheduler
from keras.layers import BatchNormalization, RandomFlip, RandomRotation, Activation, concatenate, AveragePooling2D
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 64, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 50, "Number of epochs")


def extract_and_reshape(dic):
    """
    This function takes in the un-pickled data dictionary, extracts the data array (10000, 3072) and label array
    (10000, 1). It reshapes the data array into (10000, 32, 32, 3) so that each row contains the pixel-wise RGB value
    of each image. It then returns the reshaped data array and the label array
    -----------------------------------------------------------------------------------------------------------------
    :prams dic: the dictionary object containing the un-pickled data
        :return: reshaped data array and label array
    """
    return dic[b'data'].reshape((len(dic[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32'), \
           np.array(dic[b'labels'])


def unzip_and_unpickle(file):
    """
    This function takes the name/filepath of a .tar.gz file which contains the desired dataset as input, it unzips and
    iterates through the files to extract the data and label of each entry. It returns the combined training and
    validation data and label arrays, and the testing data and label arrays
    -----------------------------------------------------------------------------------------------------------------
    :param file: name/filepath leading to the ".tar.gz" file the contains the dataset
        :return: training_and_validation_data, training_and_validation_label, testing_data, testing_label
    """
    train_val_pix = []
    train_val_lbl = []
    testing_pix = []
    testing_lbl = []
    # https://stackoverflow.com/questions/37474767/read-tar-gz-file-in-python
    f = tarfile.open(file, 'r:gz', encoding='utf-8')
    for files in f.getmembers():
        if files.name.__contains__('_batch_'):
            fp = f.extractfile(files)
            if fp:
                # https://stackoverflow.com/questions/49045172/cifar10-load-data-takes-long-time-to-download-data
                dic = pickle.load(fp, encoding='bytes')
                if not len(train_val_pix):
                    train_val_pix, train_val_lbl = (extract_and_reshape(dic=dic))
                else:
                    train_val_pix = np.concatenate((train_val_pix, extract_and_reshape(dic=dic)[0]), axis=0)
                    train_val_lbl = np.concatenate((train_val_lbl, extract_and_reshape(dic=dic)[1]), axis=0)
        elif files.name.__contains__('test_batch'):
            fp = f.extractfile(files)
            if fp:
                # https://stackoverflow.com/questions/49045172/cifar10-load-data-takes-long-time-to-download-data
                dic = pickle.load(fp, encoding='bytes')
                testing_pix, testing_lbl = extract_and_reshape(dic=dic)
    return train_val_pix, train_val_lbl, testing_pix, testing_lbl


def preprocess(train_val_pixels, train_val_labels, test_pixels, test_labels):
    """
    Preprocess data to get it ready for training:
        - normalize pixel-wise RGB data using z-score method => z-score = (xi - mean(xs)) / (std(xs) + eps)
        - split into train, validation, and test dataset
    -------------------------------------------------------------------------------------------------------
    :param train_val_pixels: pixel-wise RGB value of each image for training and validation
    :param train_val_labels: label of each image for training and validation
    :param test_pixels: pixel-wise RGB value of each image for testing
    :param test_labels: label of each image for testing
        :return: (train_x, train_y), (validation_x, validation_y), (test_x, test_y)
    """
    # normalize the pixel RGB value by calculating z-score
    epsilon = 1e-10
    train_val_pixels_mean = np.mean(train_val_pixels, axis=(0, 1, 2, 3))
    train_val_pixels_std = np.std(train_val_pixels, axis=(0, 1, 2, 3))

    test_pixels_mean = np.mean(test_pixels, axis=(0, 1, 2, 3))
    test_pixels_std = np.std(test_pixels, axis=(0, 1, 2, 3))

    train_val_pixels_normalized = (train_val_pixels - train_val_pixels_mean) / (train_val_pixels_std + epsilon)
    test_pixels_normalized = (test_pixels - test_pixels_mean) / (test_pixels_std + epsilon)

    # split the data => 80% train + 20% validation
    train_range = range(0, int(0.8 * len(train_val_labels)))
    val_range = range(int(0.8 * len(train_val_labels)), len(train_val_labels))

    train_pix_arr = train_val_pixels_normalized[train_range]
    train_lbl_arr = train_val_labels[train_range]
    val_pix_arr = train_val_pixels_normalized[val_range]
    val_lbl_arr = train_val_labels[val_range]
    test_pix_arr = test_pixels_normalized
    test_lbl_arr = test_labels

    return train_pix_arr, train_lbl_arr, val_pix_arr, val_lbl_arr, test_pix_arr, test_lbl_arr


def conv_module(inputs, num_filters, filter_size, strides, channel_dim, padding='same'):
    """
    ==> Conv2D ==> BN ==> ReLU ==>
    ------------------------------------------------------------------------------------------------
    :param inputs: input data
    :param num_filters: number of filters
    :param filter_size: kernel size
    :param strides: The stride of the sliding window for each dimension of input.
    :param channel_dim: channel dimension
    :param padding: Either the string "SAME" or "VALID" indicating the type of padding algorithm to use, or a list
                    indicating the explicit paddings at the start and end of each dimension.
        :return: Convolutional Module
    """
    output = Conv2D(filters=num_filters, kernel_size=filter_size, strides=strides, padding=padding)(inputs)
    output = BatchNormalization(axis=channel_dim)(output)
    output = Activation('relu')(output)
    return output


def inception_module(inputs, num_1x1_filters, num_3x3_filters, num_5x5_filters, num_pool_proj, channel_dim):
    """
    ==> 1x1 Conv ==================>|
    ==> 3x3 Conv ==================>|
                                    |=====> Concatenate
    ==> 5x5 Conv ==================>|
    ==> 3x3 MaxPool ==> 1x1 Conv ==>|
    --------------------------------------------------------
    :param inputs: input data
    :param num_1x1_filters: number of filters for 1x1 conv
    :param num_3x3_filters: number of filters for 3x3 conv
    :param num_5x5_filters: number of filters for 5x5 conv
    :param num_pool_proj: number of filters for max pooling
    :param channel_dim: channel dimension
        :return: Inception Module
    """
    conv_1x1 = conv_module(inputs=inputs, num_filters=num_1x1_filters, filter_size=(1, 1),
                           strides=(1, 1), channel_dim=channel_dim, padding='same')
    conv_3x3 = conv_module(inputs=inputs, num_filters=num_3x3_filters, filter_size=(3, 3),
                           strides=(1, 1), channel_dim=channel_dim, padding='same')
    conv_5x5 = conv_module(inputs=inputs, num_filters=num_5x5_filters, filter_size=(5, 5),
                           strides=(1, 1), channel_dim=channel_dim, padding='same')
    pool_projection = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    pool_projection = Conv2D(filters=num_pool_proj, kernel_size=(1, 1), padding='same',
                             activation='relu')(pool_projection)
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_projection], axis=channel_dim)
    return output


def downsample_module(inputs, num_filters, channel_dim):
    """
    ==> 3x3 Conv =====> |
                        |=====> Concatenate
    ==> 3x3 MaxPool ==> |
    --------------------------------------------------------
    :param inputs: input data
    :param num_filters: number of filters]
    :param channel_dim: channel dimension
        :return: Downsample Module
    """
    conv_3x3 = conv_module(inputs=inputs, num_filters=num_filters, filter_size=(3, 3), strides=(2, 2),
                           channel_dim=channel_dim, padding='valid')
    max_pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(inputs)
    output = concatenate([conv_3x3, max_pool], axis=channel_dim)
    return output


# https://machinelearningknowledge.ai/googlenet-architecture-implementation-in-keras-with-cifar-10-dataset/
def get_model(height, width, depth, num_classes):
    input_shape = (height, width, depth)
    channel_dim = -1

    inputs = keras.layers.Input(shape=input_shape)

    # one conv
    x = conv_module(inputs=inputs, num_filters=96, filter_size=(3, 3), strides=(1, 1), padding='same',
                    channel_dim=channel_dim)

    # two inception => downsample
    x = inception_module(inputs=x, num_1x1_filters=32, num_3x3_filters=32, num_5x5_filters=32, num_pool_proj=32,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=32, num_3x3_filters=48, num_5x5_filters=48, num_pool_proj=32,
                         channel_dim=channel_dim)
    x = downsample_module(inputs=x, num_filters=80, channel_dim=channel_dim)

    # five inception => downsample
    x = inception_module(inputs=x, num_1x1_filters=112, num_3x3_filters=48, num_5x5_filters=32, num_pool_proj=48,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=96, num_3x3_filters=64, num_5x5_filters=32, num_pool_proj=32,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=80, num_3x3_filters=80, num_5x5_filters=32, num_pool_proj=32,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=48, num_3x3_filters=96, num_5x5_filters=32, num_pool_proj=32,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=112, num_3x3_filters=48, num_5x5_filters=32, num_pool_proj=48,
                         channel_dim=channel_dim)
    x = downsample_module(inputs=x, num_filters=96, channel_dim=channel_dim)

    # two inception
    x = inception_module(inputs=x, num_1x1_filters=176, num_3x3_filters=160, num_5x5_filters=96, num_pool_proj=96,
                         channel_dim=channel_dim)
    x = inception_module(inputs=x, num_1x1_filters=176, num_3x3_filters=160, num_5x5_filters=96, num_pool_proj=96,
                         channel_dim=channel_dim)

    # global pool and dropout
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Dropout(0.5)(x)

    # softmax classifier
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=num_classes)(x)
    x = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(inputs, x, name='GoogleNet')
    return model


def learning_rate_scheduler(epoch):
    return .001 * (1 - .02) ** epoch


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.config.list_physical_devices('GPU'))
    # https://stackoverflow.com/questions/65697623/tensorflow-warning-found-untraced-functions-such-as-lstm-cell-6-layer-call-and
    absl.logging.set_verbosity(absl.logging.ERROR)  # used to ignore warning when saving custom model

    # Handle the flags
    FLAGS(sys.argv)
    BATCH_SIZE = FLAGS.batch_size
    NUM_ITERS = FLAGS.num_iters

    # import and preprocess data
    x_train_val, y_train_val, x_test, y_test = unzip_and_unpickle(file="./cifar-10-python.tar.gz")
    train_x, train_y, val_x, val_y, test_x, test_y = preprocess(train_val_pixels=x_train_val,
                                                                train_val_labels=y_train_val,
                                                                test_pixels=x_test,
                                                                test_labels=y_test)
    # data augmentation method modified from
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
        samplewise_std_normalization=False, zca_whitening=False,
        zca_epsilon=1e-06, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0,
        fill_mode='nearest', cval=0.0,
        horizontal_flip=True, vertical_flip=False,
        rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
    )

    # train and evaluate model
    myModel = get_model(height=32, width=32, depth=3, num_classes=10)
    myModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # export model structure and architecture
    print(myModel.summary())
    with open('./model/model_summary.txt', 'w') as f:
        myModel.summary(print_fn=lambda x: f.write(x + '\n'))
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
    tf.keras.utils.plot_model(
        myModel, to_file='./model/model_architecture.png',
        show_shapes=True, show_dtype=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=200, layer_range=None, show_layer_activations=True
    )

    # model checkpoint and log
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/checkpoint/cifar10/{epoch}', monitor='val_loss',
        save_best_only=True, verbose=1
    )

    hist = myModel.fit_generator(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE), epochs=NUM_ITERS,
                                 steps_per_epoch=(train_x.shape[0] // BATCH_SIZE), validation_data=(val_x, val_y),
                                 verbose=1, callbacks=[LearningRateScheduler(learning_rate_scheduler),
                                                       model_checkpoint_callback])
    test_loss, test_acc = myModel.evaluate(x=test_x, y=test_y, verbose=1)
    print('Test loss\t\t:', test_loss)
    print('Test accuracy\t:', test_acc)

    # plotting the training accuracy and loss
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=200)
    axs[0].set_title('Training Accuracy Histogram')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].plot(hist.history['accuracy'], label='training accuracy')
    axs[0].plot(hist.history['val_accuracy'], label='validation accuracy')
    axs[0].legend(loc='lower right')

    axs[1].set_title('Training Loss Histogram')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].plot(hist.history['loss'], label='training loss')
    axs[1].plot(hist.history['val_loss'], label='validation loss')
    axs[1].legend(loc='upper right')

    plt.show()

