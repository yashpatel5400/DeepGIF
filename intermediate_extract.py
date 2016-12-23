"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Extract the styles and contents from input images using
standard 19-layer VGG CNN from an input image and outputs a
test image to verify working
"""

import settings as s

import numpy as np
from random import shuffle

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import PIL
from PIL import Image	
from scipy.misc import imsave

def stitch_images(images):
	n = 4
	margin = 5

	width = n * s.WIDTH + (n - 1) * margin
	height = n * s.HEIGHT + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))

	for i in range(n):
		for j in range(n):
			img = images[i * n + j]
			stitched_filters[(s.WIDTH + margin) * i: (s.WIDTH + margin) * i + s.WIDTH,
							 (s.HEIGHT + margin) * j: (s.HEIGHT + margin) * j + s.HEIGHT, :] = img
	return stitched_filters

def main(filename):
	unextended_filename = filename.split(".")[0]

	base_model = VGG19(weights='imagenet')
	intermediates = [name for name in 
		[layer.name for layer in base_model.layers] if "pool" in name]
	VISUALIZE_FILTERS = 16

	img = image.load_img("{}/{}".format(s.INPUT_DIR, filename), 
		target_size=(s.WIDTH, s.HEIGHT))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	for i, intermediate in enumerate(intermediates):
		intermediate_model = Model(input=base_model.input, 
			output=base_model.get_layer(intermediate).output)
		pool_features = intermediate_model.predict(x)

		imgs = []
		shuffle(pool_features)
		for img_filter in range(VISUALIZE_FILTERS):
			img = Image.fromarray(pool_features[0,:,:,img_filter])
			imgs.append(img.resize((s.WIDTH, s.HEIGHT), 
				PIL.Image.ANTIALIAS).convert('RGB'))
		
		stitched_filters = stitch_images(imgs)
		imsave("{}/{}_layer-{}.png".format(s.OUTPUT_CONTENT_DIR, 
			unextended_filename,i), stitched_filters)

if __name__ == "__main__":
	main('cat.jpg')