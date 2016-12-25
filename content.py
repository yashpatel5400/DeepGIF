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

def img_tensor(filename):
	img = image.load_img(filename, target_size=(s.WIDTH, s.HEIGHT))
	img_arr = image.img_to_array(img)
	img_arr = np.expand_dims(img_arr, axis=0)
	img_arr = preprocess_input(img_arr)
	return K.variable(img_arr)

def main(trial_settings):
	content_img    = trial_settings['content_img']
	content_weight = trial_settings['content_weight']
	style_img      = trial_settings['style_img']
	style_weight   = trial_settings['style_weight']

	combined_name = "{}-{}".format(content_img.split(".")[0], 
		style_img.split(".")[0])

	model_input = []
	# input image: content
	content_img_tensor = img_tensor("{}/{}".format(
		s.INPUT_CONTENT_DIR, content_img))
	model_input.append(content_img_tensor)

	# input image: style
	style_img_tensor = img_tensor("{}/{}".format(
		s.INPUT_STYLE_DIR, style_img))
	model_input.append(style_img_tensor)
	
	# tensor used for "molding to" the desired combination
	transform_image_tensor = K.placeholder((1, s.WIDTH, s.HEIGHT, 3))
	model_input.append(transform_image_tensor)
		
	# get results of running VGG for input and the transformation
	combined_tensor = K.concatenate(model_input, axis=0)
	model = VGG19(input_tensor=combined_tensor, 
		weights='imagenet', include_top=False)
	layers = [name for name in 
		[layer.name for layer in model.layers] if "pool" in name]

	content_features   = []
	style_features     = []
	transform_features = []

	for i, layer in enumerate(layers):
		print('Processing layer {}'.format(i))
		features = model.get_layer(layer).output
		content_features.append(features[0,:,:,:])
		style_features.append(features[1,:,:,:])
		transform_features.append(features[2,:,:,:])

	final_img = visualize_filters(content_features, content_weight, 
		style_features, style_weight, transform_features, transform_image_tensor)
	imsave("{}/{}.png".format(s.OUTPUT_FINAL_DIR, combined_name), final_img)

if __name__ == "__main__":
	main({
		'content_img': 'cat.jpg',
		'content_weight': 1.0,

		'style_img': 'starry_night.jpg',
		'style_weight': 1.0,
	})