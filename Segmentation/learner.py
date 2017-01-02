# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import sys

import cv2
import numpy as np

import tensorflow as tf
import tifffile
from scipy.misc import imsave


def train(model, data_provider, data_folder, n_iterations=500):
    results_folder = data_folder + 'results/'
    ckpt_folder = results_folder + model.model_name + '/'

    print('Run tensorboard to visualize training progress')
    with tf.Session() as sess:
        # We will write our summaries here
        summary_writer = tf.summary.FileWriter(ckpt_folder + '/events', graph=sess.graph)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Iterate through the dataset
        for step, (inputs, labels) in enumerate(data_provider.batch_iterator(model.fov, model.out, model.inpt)):

            sess.run(model.train_step, feed_dict={
                model.image: inputs,
                model.target: labels
            })

            if step % 10 == 0:
                print('step :' + str(step))
                summary = sess.run(model.loss_summary, feed_dict={
                    model.image: inputs,
                    model.target: labels
                })

                summary_writer.add_summary(summary, step)

            if step % 500 == 0:
                # Measure validation error

                # Compute pixel error

                validation_sigmoid_prediction, validation_pixel_error_summary = \
                    sess.run([model.sigmoid_prediction, model.validation_pixel_error_summary],
                             feed_dict={model.image: reshaped_validation_input,
                                        model.target: reshaped_labels})

                summary_writer.add_summary(validation_pixel_error_summary, step)

                # Calculate rand and VI scores
                scores = evaluation.rand_error(model, data_folder, validation_sigmoid_prediction, num_validation_layers,
                                               validation_output_shape, watershed_high=0.95)
                score_summary = sess.run(model.score_summary_op)

                summary_writer.add_summary(score_summary, step)

            if step % 50 == 0:
                # Save the variables to disk.
                save_path = model.saver.save(sess, ckpt_folder + 'model.ckpt')
                print("Model saved in file: %s" % save_path)

            if step == n_iterations:
                break

    return scores


def predict(model, data_folder, input_image):
    results_folder = data_folder + 'results/'

    # Where we store model
    ckpt_folder = results_folder + 'N4/'

    # Where the data is stored
    data_prefix = data_folder

    # Get the h5 input
    image = cv2.imread(input_image, cv2.IMREAD_COLOR)  # uint8 image
    dest_shape = image.shape
    dest = np.zeros(dest_shape)

    norm_image = cv2.normalize(image, dest, alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input_shape = norm_image.shape[1]
        
    with tf.Session() as sess:
        # Restore variables from disk.
        model.saver.restore(sess, ckpt_folder + 'model.ckpt')
        print("Model restored.")
        pred = sess.run(model.sigmoid_prediction, feed_dict={
            model.image: mirrored_inpt[z].reshape(1, input_shape, input_shape, 1)})
        imsave("prediction.jpg", pred)