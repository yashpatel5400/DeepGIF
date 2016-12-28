"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Extract the styles and contents from input images using
standard 19-layer VGG CNN from an input image and outputs a
test image to verify working
"""

import settings as s

import numpy as np
import os
from random import shuffle
from scipy.optimize import fmin_l_bfgs_b

from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import cv2
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
# adopted from https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
class GradLoss:
	def __init__(self, iterate):
		self.iterate = iterate

	def loss(self, x):
		x = x.reshape((1, WIDTH, HEIGHT, 3))
		outs = self.iterate([x])
		self.grad_values = np.array(outs[1:]).flatten().astype('float64')
		return outs[0]

	def grad(self, x):
		return self.grad_values

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
# from https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
def coherence_loss(generated_features):
	a = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, 1:, :HEIGHT-1, :])
	b = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, :WIDTH-1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

# adopted from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def transform(content_features, content_weight, style_features, style_weights, 
	transform_features, output_img, output_name, is_video):

	loss = K.variable(0.0)
	loss  +=  content_weight * content_loss(content_features, transform_features)
	for style_feature, transform_feature, weight in zip(style_features, 
		transform_features, style_weights):

		loss  += weight * style_loss(style_feature, transform_feature)
	loss  += coherence_loss(output_img)

	grads = K.gradients(loss, output_img)[0]

	# step size for gradient ascent
	shape = output_img.get_shape()
	img_width   = shape[1].value
	img_height  = shape[2].value
	
	grad_loss = GradLoss(K.function([output_img], [loss, grads]))
	input_img_data = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

	for i in range(s.NUM_ITERATIONS):
		input_img_data, min_val, info = fmin_l_bfgs_b(grad_loss.loss, 
			input_img_data.flatten(), fprime=grad_loss.grad, maxfun=20)

		if not is_video:
			imsave("{}/{}-{}.png".format(s.OUTPUT_FINAL_DIR, 
				output_name, i), deprocess_image(input_img_data.copy()))
		print('Current loss value:', min_val)
	# only save the actual frame if processing a video
	if is_video:
		imsave("{}/{}.png".format(s.OUTPUT_FRAME_DIR, 
			output_name), deprocess_image(input_img_data.copy()))

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

def stylize_image(trial_settings, is_video=False):
	content_img	   = trial_settings['content_input']
	content_weight = trial_settings['content_weight']
	style_img	   = trial_settings['style_img']
	style_weights  = trial_settings['style_weights']

	combined_name = "{}-{}".format(content_img.split(".")[0], 
		style_img.split(".")[0])

	model_input = []
	# input image: content
	if not is_video:
		input_file = "{}/{}".format(s.INPUT_CONTENT_DIR, content_img)
		final_name = combined_name
	else: 
		frame = trial_params['frame']
		input_file = "{}/{}/{}".format(s.INPUT_FRAME_DIR, 
			combined_name, frame)
		final_name = "{}/{}".format(combined_name, frame)

	content_img_tensor = img_tensor(input_file)
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

	transform(content_features, content_weight, style_features, style_weights, 
		transform_features, transform_image_tensor, final_name, is_video)

def stylize_video(trial_settings):
	style_img  = trial_settings['style_img']
	video_file = trial_settings['content_input']

	combined_name = "{}-{}".format(video_file.split(".")[0], 
		style_img.split(".")[0])

	input_dir = "{}/{}".format(s.INPUT_FRAME_DIR, combined_name)
	output_dir = "{}/{}".format(s.OUTPUT_FRAME_DIR, combined_name)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if not os.path.exists(input_dir):
		os.makedirs(input_dir)

		vidcap = cv2.VideoCapture("{}/{}".format(s.INPUT_CONTENT_DIR, video_file))
		success, image = vidcap.read()
		count = 0
		while success:
			print('Read a new frame: {}'.format(success))
			cv2.imwrite("{}/{}.jpg".format(input_dir, count), image)
			success, image = vidcap.read()
			count += 1
	else:
		count = len(os.listdir(input_dir)) + 1

	for file in range(count):
		trial_params['frame'] = "{}.jpg".format(file)
		stylize_image(trial_params, is_video=True)

	video = None
	for file in os.listdir(output_dir):
		img = cv2.imread("{}/{}".format(output_dir, file))
		if video is None:	
			height, width, layers = img.shape
			video = cv2.VideoWriter('{}.avi'.format(combined_name), 
				-1, 1, (width, height))
		video.write(img)
	
	cv2.destroyAllWindows()
	video.release()

if __name__ == "__main__":
	trial_params = {
		'is_video': True,
		# only used for the video stylizations
		'frame': None,

		'content_input': 'starwars.gif',
		'content_weight': 0.025,

		'style_img': 'scream.jpg',
		'style_weights': [.75, .75, .75, .75, .75]
	}

	if trial_params['is_video']:
		stylize_video(trial_params)
	else: stylize_image(trial_params, is_video=False)