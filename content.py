"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Extract the styles and contents from input images using
standard 19-layer VGG CNN from an input image and outputs a
test image to verify working
"""

import settings as s
from style import visualize_filters

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

def content_extract(features):
	VISUALIZE_FILTERS = 16
	imgs = []
	shuffle(features)
	for img_filter in range(VISUALIZE_FILTERS):
		img = Image.fromarray(features[0,:,:,img_filter])
		imgs.append(img.resize((s.WIDTH, s.HEIGHT), 
			PIL.Image.ANTIALIAS).convert('RGB'))
	return stitch_images(imgs)

def content_loss(original, generated):
	return K.sum(K.square(original - generated))

def style_extract(features):
	num_filters = features.shape[-1]

	# creates the Gram matrix
	gram = np.zeros((num_filters, num_filters))
	for i in range(num_filters):
		a = features[0,:,:,i]
		for j in range(num_filters):
			b = features[0,:,:,j]
			gram[i][j] = np.tensordot(a, b)

	img = Image.fromarray(gram)
	return img.resize((s.WIDTH, s.HEIGHT), 
		PIL.Image.ANTIALIAS).convert('RGB')

def main(filename):
	unextended_filename = filename.split(".")[0]

	base_model = VGG19(weights='imagenet')
	intermediates = [name for name in 
		[layer.name for layer in base_model.layers] if "pool" in name]
	
	img = image.load_img("{}/{}".format(s.INPUT_DIR, filename), 
		target_size=(s.WIDTH, s.HEIGHT))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	for i, intermediate in enumerate(intermediates):
		intermediate_model = Model(input=base_model.input, 
			output=base_model.get_layer(intermediate).output)
		# features = intermediate_model.predict(x)
		
		# content = content_extract(features)
		# imsave("{}/{}_layer-{}.png".format(s.OUTPUT_CONTENT_DIR, 
		#	unextended_filename, i), content)

		style = visualize_filters(intermediate_model, intermediates[:i+1])
		style.save("{}/{}_layer-{}.png".format(s.OUTPUT_STYLE_DIR, 
			unextended_filename, i))

if __name__ == "__main__":
	main('cat.jpg')