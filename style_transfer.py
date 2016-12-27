"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Extract the styles and contents from input images using
standard 19-layer VGG CNN from an input image and outputs a
test image to verify working
"""

import settings as s

import numpy as np
from random import shuffle
from scipy.optimize import fmin_l_bfgs_b

from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import PIL
from PIL import Image
from scipy.misc import imsave

# sizes of the input: initialized based on the content image
WIDTH  = None
HEIGHT = None

# util function to convert a tensor into a valid image
# from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def deprocess_image(x):
	x = x.reshape((WIDTH, HEIGHT, 3))
	# Remove zero-center by mean pixel
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68

	# 'BGR'->'RGB'
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x

# stores loss and gradients for efficiency
class Evaluator(object):
	def __init__(self, iterate):
		self.loss_value = None
		self.grads_values = None
		self.iterate = iterate

	def loss(self, x):
		x = x.reshape((1, WIDTH, HEIGHT, 3))
		outs = self.iterate([x])
		
		self.loss_value = outs[0]
		self.grad_values = np.array(outs[1:]).flatten().astype('float64')
		return self.loss_value

	def grads(self, x):
		grad_values = np.copy(self.grad_values)
		
		self.loss_value  = None
		self.grad_values = None
		return grad_values

# the following three functions are defined by their descriptions
# from the "Style Transfer" paper

def gram_matrix(output):
	output = K.permute_dimensions(output, (2, 0, 1))
	flat_output = K.batch_flatten(output)
	return K.dot(flat_output, K.transpose(flat_output))

def style_loss(original_feature, generated_feature):
	shape = original_feature.get_shape()
	# correspondingly the M_l and N_l from the paper description
	img_size = (shape[0] * shape[1]).value
	# num_filters = 3
	num_filters = shape[2].value
		
	G_orig = gram_matrix(original_feature)
	G_gen  = gram_matrix(generated_feature)
	return (1. / (4. * img_size ** 2 * 
		num_filters ** 2)) * K.sum(K.square(G_orig - G_gen))
	
def content_loss(original_features, generated_features):
	original_content  = original_features[s.CONTENT_FEATURE_LAYER]
	generated_content = generated_features[s.CONTENT_FEATURE_LAYER]
	return K.sum(K.square(original_content - generated_content))

# additional loss function that reduces pixelation
def total_variation_loss(generated_features):
	a = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, 1:, :HEIGHT-1, :])
	b = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, :WIDTH-1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

# adopted from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def transform(content_features, content_weight, style_features, style_weights, 
	transform_features, output_img, output_name):

	loss = K.variable(0.0)
	loss  +=  content_weight * content_loss(content_features, transform_features)
	for style_feature, transform_feature, weight in zip(style_features, 
		transform_features, style_weights):

		loss  += weight * style_loss(style_feature, transform_feature)
	loss  += total_variation_loss(output_img)

	grads = K.gradients(loss, output_img)[0]
	# this function returns the loss and grads given the input picture
	iterate = K.function([output_img], [loss, grads])

	# step size for gradient ascent
	shape = output_img.get_shape()
	img_width   = shape[1].value
	img_height  = shape[2].value
	
	evaluator = Evaluator(iterate)
	input_img_data = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

	for i in range(s.NUM_ITERATIONS):
		input_img_data, min_val, info = fmin_l_bfgs_b(evaluator.loss, 
			input_img_data.flatten(), fprime=evaluator.grads, maxfun=20)
		imsave("{}/{}-{}.png".format(s.OUTPUT_FINAL_DIR, 
			output_name, i), deprocess_image(input_img_data.copy()))
		print('Current loss value:', min_val)

def img_tensor(filename):
	global WIDTH
	global HEIGHT

	if WIDTH is None or HEIGHT is None:
		img = image.load_img(filename)
		img_arr = image.img_to_array(img)
		WIDTH   = img_arr.shape[0]
		HEIGHT  = img_arr.shape[1]
	else:
		img = image.load_img(filename, target_size=(WIDTH, HEIGHT))
		img_arr = image.img_to_array(img)

	img_arr = np.expand_dims(img_arr, axis=0)
	img_arr = preprocess_input(img_arr)
	return K.variable(img_arr)

def main(trial_settings):
	content_img	= trial_settings['content_img']
	content_weight = trial_settings['content_weight']
	style_img	   = trial_settings['style_img']
	style_weights  = trial_settings['style_weights']

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
	transform_image_tensor = K.placeholder((1, WIDTH, HEIGHT, 3))
	model_input.append(transform_image_tensor)
		
	# get results of running VGG for input and the transformation
	combined_tensor = K.concatenate(model_input, axis=0)
	model = VGG19(input_tensor=combined_tensor, 
		weights='imagenet', include_top=False)
	layers = [name for name in 
		[layer.name for layer in model.layers] if "conv1" in name]

	content_features   = []
	style_features	 = []
	transform_features = []

	for i, layer in enumerate(layers):
		print('Processing layer {}'.format(i))
		features = model.get_layer(layer).output
		content_features.append(features[0,:,:,:])
		style_features.append(features[1,:,:,:])
		transform_features.append(features[2,:,:,:])

	transform(content_features, content_weight, style_features, 
		style_weights, transform_features, transform_image_tensor, combined_name)

if __name__ == "__main__":
	main({
		'content_img': 'bagend.jpg',
		'content_weight': 0.00125,

		'style_img': 'scream.jpg',
		'style_weights': [.25, .25, .25, .25, .25]
	})