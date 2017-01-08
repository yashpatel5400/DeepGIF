"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Preprocessing code used for set of for both the neural
transfer standard and fast editions. Mostly an assortment of miscellaneous
image cleanup and transformation methods
"""

import settings as s

import os
import numpy as np
import cv2

# Credit to: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def deprocess_image(tensor):
	"""
	Convert a tensor into a valid image
	
	:param tensor: Tensor corresponding to an image
	:return: Numpy array representation of RGB image
	"""
	
	img = tensor.reshape((WIDTH, HEIGHT, 3))
	# Remove zero-center by mean pixel
	img[:, :, 0] += 103.939
	img[:, :, 1] += 116.779
	img[:, :, 2] += 123.68

	# 'BGR'->'RGB'
	img = img[:, :, ::-1]
	img = np.clip(img, 0, 255).astype('uint8')
	return img

def preprocess_gif(gif, input_dir=s.INPUT_CONTENT_DIR, 
	output_dir=s.OUTPUT_FRAME_DIR):
    """
    Breaks a GIF file into constituent frame images and saves them in the
    specified directory. If no directory specified, saves to default input/frames.
    Then runs and returns edge segmentation results on the image frame stack
    :return: Array of numpy arrays corresponding to edge segmentation
    """
    filename = gif.split(".")[0]
    
    input_dir = "{}{}".format(input_dir, filename)
    output_dir = "{}{}".format(output_dir, filename)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = []
    vidcap = cv2.VideoCapture("{}/{}".format(input_dir, gif))
    success, image = vidcap.read()
    count = 0
    while success:
        print('Read a new frame: {}'.format(success))
        cv2.imwrite("{}/{}.jpg".format(input_dir, count), image)
        frames.append("{}/{}.jpg".format(filename, count))

        success, image = vidcap.read()
        count += 1