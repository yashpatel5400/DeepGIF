"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Neural network implementation of the N4 architecture.
Used for segmented images into edge maps, which then feeds into watershed
for final area segmentation.
"""

import model.settings as s

import os
import cv2
import numpy as np

from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers.core import Layer, Activation, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def default_N4():
    params = {
        # size of filter outputs
        'fs1': 96,
        'fs2': 128,
        'fs3': 256,

        # convolution kernel sizes
        'k1': 7,
        'k2': 5,
        'k3': 3,

        # fully connected layer sizes
        'fc1': 768,
        'fc2': 768,
        'fc3': 16,

        # learning rate
        'lr': 0.1
    }
    return N4(params)

class N4:
    def train(self):
        # -------------------- Training Procedure ----------------------------#
        print("Loading training files...")
        X_train_dir   = "{}{}".format(s.RAW_INPUT_DIR, s.TRAIN)
        y_train_dir   = "{}{}".format(s.EDGE_INPUT_DIR, s.TRAIN)

        X_train_files = os.listdir(X_train_dir)
        y_train_files = os.listdir(y_train_dir)

        # have to sort to ensure the input image and corresonding truth line up
        X_train_files.sort()
        y_train_files.sort()

        X_train_reg = [cv2.imread("{}{}".format(X_train_dir, train_file)) for 
            train_file in X_train_files if train_file.split(".")[-1] == "jpg"]
        y_train_reg = [cv2.imread("{}{}".format(y_train_dir, train_file)) 
            for train_file in y_train_files]
        
        # need to convert to numpy array to have proper shapes
        total_pictures  = len(X_train_reg) * s.RAW_SEGMENTATION_SIZE
        reference_shape = X_train_reg[0].shape
        final_shape = (total_pictures, reference_shape[0], 
            reference_shape[1], reference_shape[2])

        X_train = np.empty(final_shape)
        y_train = np.empty(final_shape)

        for picture in range(total_pictures):
            y_train[picture] = y_train_reg[picture]

            # since each training image corresponds to four "truth images", we copy
            for copy in range(s.RAW_SEGMENTATION_SIZE):
                X_train[picture * s.RAW_SEGMENTATION_SIZE + copy] = X_train_reg[picture]

        print(X_train.shape)
        print(y_train.shape)

        print("Performing training...")
        sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9)
        self.model.compile(loss='mean_squared_error',
                      optimizer=sgd,
                      metrics=['accuracy'])
        self.model.fit(X_train, y_train, nb_epoch=20, batch_size=16)

        print("Saving model...")
        cache_path = "{}{}.h5".format(s.MODEL_CACHE, self.model_name)
        self.model.save_weights(cache_path)

        # ---------------- Testing/Validation Procedure ----------------------#
        print("Loading testing files...")
        X_test_files = os.listdir("{}{}".format(s.RAW_INPUT_DIR, s.TEST))
        y_test_files = os.listdir("{}{}".format(s.EDGE_INPUT_DIR, s.TEST))

        X_single_imgs = ["{}-0.jpg"(test_file.split("-")[0]) for 
            test_file in X_test_files]
        X_test = [cv2.imread(img) for img in X_single_imgs]
        y_test = [cv2.imread(test_file) for test_file in y_test_files 
            if test_file.split(".")[-1] == "jpg"]
        score = self.model.evaluate(X_test, y_test, batch_size=16)

    def predict(self, input_image):
        cache_path = "{}{}.h5".format(s.MODEL_CACHE, self.model_name)
        if os.path.exists(cache_path):
            self.model.load_weights(cache_path)

        filename = input_image.split(".")[0]

        image = cv2.imread("{}{}{}".format(s.RAW_INPUT_DIR, 
            s.TEST, input_image), cv2.IMREAD_COLOR)
        dest_shape = image.shape
        dest = np.zeros(dest_shape)

        norm_image = cv2.normalize(image, dest, alpha=0, beta=1, 
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        edge_img = self.model.predict(norm_image)
        imsave("{}{}".format(s.OUTPUT_DIR, filename), edge_img)

    def __init__(self, params):
        filter_1 = params['fs1']
        filter_2 = params['fs2']
        filter_3 = params['fs3']

        kernel_1 = params['k1']
        kernel_2 = params['k2']
        kernel_3 = params['k3']

        full_connected_1 = params['fc1']
        full_connected_2 = params['fc2']
        full_connected_3 = params['fc3']
        
        self.model_name    = "n4"
        self.learning_rate = params['lr']
        self.model = Sequential()
        self.model.add(Layer(input_shape=(None, None, 3)))
        
        # 1st convolution layer -- with pooling
        self.model.add(Convolution2D(filter_1, kernel_1, kernel_1,
            border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2nd convolution layer -- with pooling
        self.model.add(Convolution2D(filter_2, kernel_2, kernel_2,
            border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3rd convolution layer -- no pooling
        self.model.add(Convolution2D(filter_3, kernel_3, kernel_3,
            border_mode='same', activation='relu'))

        self.model.add(Dense(full_connected_1, activation='relu'))
        self.model.add(Dense(full_connected_2, activation='relu'))
        self.model.add(Dense(full_connected_3))