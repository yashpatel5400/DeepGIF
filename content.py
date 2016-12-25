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

from keras import backend as K
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
	model_input = []

	# input image
	input_img = image.load_img("{}/{}".format(s.INPUT_DIR, filename), 
		target_size=(s.WIDTH, s.HEIGHT))
	input_img_arr = image.img_to_array(input_img)
	input_img_arr = np.expand_dims(input_img_arr, axis=0)
	input_img_arr = preprocess_input(input_img_arr)
	input_img_tensor = K.variable(input_img_arr)
	model_input.append(input_img_tensor)
	
	# tensor used for "molding to" the desired combination
	transform_image_tensor = K.placeholder((1, s.WIDTH, s.HEIGHT, 3))
	model_input.append(transform_image_tensor)
		
	# get results of running VGG for input and the transformation
	combined_tensor = K.concatenate(model_input, axis=0)
	model = VGG19(input_tensor=combined_tensor, 
		weights='imagenet', include_top=False)
	layers = [name for name in 
		[layer.name for layer in model.layers] if "pool" in name]

	input_image_features = []
	transform_image_features = []

	for i, layer in enumerate(layers):
		# intermediate_model = Model(input=base_model.input, 
		# 	output=base_model.get_layer(intermediate).output)
		features = model.get_layer(layer).output
		input_image_features.append(features[0,:,:,:])
		transform_image_features.append(features[1,:,:,:])

		# content = content_extract(features)
		# imsave("{}/{}_layer-{}.png".format(s.OUTPUT_CONTENT_DIR, 
		#	unextended_filename, i), content)

		print('Processing layer {}'.format(i))
		style = visualize_filters(input_image_features, 
			transform_image_features, transform_image_tensor)
		imsave("{}/{}_layer-{}.png".format(s.OUTPUT_STYLE_DIR, 
			unextended_filename, i), style)

if __name__ == "__main__":
	main('starry_night.jpg')