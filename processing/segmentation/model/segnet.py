"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Segnet implementation for segmenting images.
Includes training and testing routines (adopted from:
https://github.com/pradyu1993/segnet/blob/master/segnet.py)
"""

import model.settings as s

from keras.models import Sequential
from keras.layers.core import Layer, Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import numpy as np
import cv2
import os

def ConvNormReLU(model, filter_size):
    # same padding, since image is not reduced until pooling
    model.add(Convolution2D(filter_size, s.KERNEL, s.KERNEL, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

def DownsamplingBlock(model, num_conv, filter_size):
    for _ in range(num_conv):
        ConvNormReLU(model, filter_size)
    model.add(MaxPooling2D(pool_size=(s.POOL_SIZE, s.POOL_SIZE)))

def UpsamplingBlock(model, num_conv, filter_size):
    model.add(UpSampling2D(size=(s.POOL_SIZE, s.POOL_SIZE)))
    for _ in range(num_conv):
        ConvNormReLU(model, filter_size)

def default_segnet():
    model = Sequential()
    # input layer of image
    model.add(Layer(input_shape=(s.WIDTH, s.HEIGHT, 3)))

    DownsamplingBlock(model, 2, 12)
    DownsamplingBlock(model, 2, 32)
    DownsamplingBlock(model, 3, 64)
    DownsamplingBlock(model, 3, 128)
    DownsamplingBlock(model, 3, 256)

    UpsamplingBlock(model, 3, 256)
    UpsamplingBlock(model, 3, 128)
    UpsamplingBlock(model, 3, 64)
    UpsamplingBlock(model, 2, 32)
    UpsamplingBlock(model, 2, 12)

    # final output probability map (w/ softmax)
    model.add(Reshape((s.WIDTH * s.HEIGHT, 12)))
    model.add(Activation('softmax'))
    return model

# credit to: https://github.com/pradyu1993/segnet/blob/master/segnet.py
def normalize(rgb):
    norm = np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    norm[:,:,0] = cv2.equalizeHist(rgb[:,:,0])
    norm[:,:,1] = cv2.equalizeHist(rgb[:,:,1])
    norm[:,:,2] = cv2.equalizeHist(rgb[:,:,2])
    return norm

# credit to: https://github.com/pradyu1993/segnet/blob/master/segnet.py
def binarylab(labels):
    truth = np.zeros([s.WIDTH, s.HEIGHT, s.NUM_CLASSES])
    for i in range(s.WIDTH):
        for j in range(s.HEIGHT):
            truth[i, j, labels[i][j]] = 1
    return truth

# adopted from: https://github.com/pradyu1993/segnet/blob/master/segnet.py
def prep_data():
    training_set = {}
    print("Reading training data...")
    with open("{}/train.txt".format('./CamVid/')) as f:
        for line in f:
            label_data = [filename.strip().replace('/SegNet', '.') for 
                filename in line.split(' ')]
            training_set[label_data[0]] = label_data[1]
    
    train_data  = []
    train_label = []
    print("Formatting training data...")
    for training_image in training_set:
        training_data  = normalize(cv2.imread(training_image))
        training_label = cv2.imread(training_set[training_image])
        labelled_img   = binarylab(training_label[:,:,0])

        train_data.append(np.rollaxis(training_data, 2))
        train_label.append(labelled_img)
    return np.array(train_data), np.array(train_label)

# credit to: https://github.com/pradyu1993/segnet/blob/master/segnet.py
def train_model(model):
    if os.path.exists('./{}'.format(s.OUTPUT_FILE)):
        model.load_weights(s.OUTPUT_FILE)

    class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 
        0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    train_data, train_label = prep_data()
    train_label = np.reshape(train_label, 
        (s.TRAINING_INP_SIZE, s.WIDTH * s.HEIGHT, s.NUM_CLASSES))

    model.fit(train_data, train_label, batch_size=s.BATCH_SIZE, 
        nb_epoch=s.NB_EPOCH, show_accuracy=True, verbose=True, 
        class_weight=class_weighting)
    model.save_weights(s.OUTPUT_FILE)