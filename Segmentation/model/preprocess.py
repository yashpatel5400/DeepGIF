"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Preprocessing of images (testing/training/validation) that
is used for N4 edge detection
"""

import settings as s

import os
import cv2
import numpy as np
from scipy.misc import imsave
import scipy.io

def preprocess_imgs():
	file_split = [s.TRAIN, s.TEST, s.VAL]
	for split in file_split:
		print("Preprocessing {}".format(split))

		cur_dir = "{}{}".format(s.RAW_INPUT_DIR, split)
		files = os.listdir(cur_dir)
		filenames = [file for file in files if file.split(".")[-1] == s.INPUT_MAT]

		for file in filenames:
			truth_file = "{}{}".format(cur_dir, file)
			truth = scipy.io.loadmat(truth_file)
			for segment in range(s.RAW_SEGMENTATION_SIZE):
				segments = truth['groundTruth'][0, segment][0][0][0]
				edges    = truth['groundTruth'][0, segment][0][0][1]

				dest = np.zeros(edges.shape)
				norm_edges = cv2.normalize(edges, dest, alpha=0, beta=1, 
					norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

				filename = "{}-{}".format(file.split(".")[0], segment)
				imsave("{}{}{}.jpg".format(s.EDGE_INPUT_DIR, split, filename), norm_edges)
				imsave("{}{}{}.jpg".format(s.SEGMENTS_INPUT_DIR, split, filename), segments)