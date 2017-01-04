"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Preprocessing of images (testing/training/validation) that
is used for N4 edge detection
"""

import model.settings as s

import os
import shutil
import cv2
import numpy as np
from scipy.misc import imsave
import urllib2

def download_model():
	pretrained_model = '{}{}'.format(s.MODEL_CACHE, s.MODEL_FILENAME)
	#if os.path.exists(pretrained_model):
	#	print("Requirement already satisfied")
	#	return

	file_name = s.MODEL_SITE.split('/')[-1]
	u = urllib2.urlopen(s.MODEL_SITE)
	f = open(file_name, 'wb')
	meta = u.info()
	file_size = int(meta.getheaders("Content-Length")[0])
	print("Downloading: {} Bytes: {}".format(file_name, file_size))

	file_size_dl = 0
	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break

		file_size_dl += len(buffer)
		f.write(buffer)
		status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
		status = status + chr(8) * (len(status)+1)
		print(status)
	
	f.close()
	shutil.move("./{}".format(file_name), "{}{}".format(s.MODEL_CACHE, file_name))

def preprocess_imgs():
	file_split = [s.TRAIN, s.TEST]
	for split in file_split:
		print("Preprocessing {}".format(split))

		cur_dir = "{}{}".format(s.RAW_INPUT_DIR, split)
		files = os.listdir(cur_dir)
		
		img_files = [file for file in files if file.split(".")[-1] == "jpg"]
		for file in img_files:
			img_file = "{}{}".format(cur_dir, file)
			img  = cv2.imread(img_file)
			dest = np.zeros(img.shape)
			norm_img = cv2.normalize(img, dest, alpha=0, beta=1, 
				norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			imsave("{}{}".format(cur_dir, file), norm_img)

		mat_files = [file for file in files if file.split(".")[-1] == "mat"]
		for file in mat_files:
			truth_file = "{}{}".format(cur_dir, file)
			truth = scipy.io.loadmat(truth_file)
			for segment in range(s.RAW_SEGMENTATION_SIZE):
				segments = truth['groundTruth'][0, segment][0][0][0]
				edges	= truth['groundTruth'][0, segment][0][0][1]

				dest = np.zeros(edges.shape)
				norm_edges = cv2.normalize(edges, dest, alpha=0, beta=1, 
					norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

				filename = "{}-{}".format(file.split(".")[0], segment)
				imsave("{}{}{}.jpg".format(s.EDGE_INPUT_DIR, split, filename), norm_edges)
				imsave("{}{}{}.jpg".format(s.SEGMENTS_INPUT_DIR, split, filename), segments)