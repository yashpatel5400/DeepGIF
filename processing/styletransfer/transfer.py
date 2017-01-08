"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Extract the styles and contents from input images using
standard 19-layer VGG CNN from an input image and outputs a
test image to verify working
"""

import settings as s
from preprocess import deprocess_image, preprocess_gif

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
from PIL import Image
from scipy.misc import imsave
import imageio

# sizes of the input: initialized based on the content image in 
# "img_tensor" function
WIDTH  = None
HEIGHT = None

# stores loss and gradients for efficiency
# Credit to: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
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

# -------------------------- Loss Functions ----------------------------------#
def _gram_matrix(output):
	"""
	Calculates Gram Matrix (https://en.wikipedia.org/wiki/Gramian_matrix) for a
	given tensor. 
	
	:param output: Tensor for which Gram Matrix will be calculated (usually image)
	:return: Tensor corresponding to Gram Matrix result
	"""
	output = K.permute_dimensions(output, (2, 0, 1))
	flat_output = K.batch_flatten(output)
	return K.dot(flat_output, K.transpose(flat_output))

def style_loss(original_feature, generated_feature):
	"""
	Calculates style loss, given the original style tensor and one being molded
	
	:param original_feature: Tensor corresponding to input style
	:param generated_feature: Tensor corresponding to image being transformed.
		Only a single layer of the image tensor should be passed (from VGG)
	:return: Tensor corresponding to the loss as a result of style difference
	"""
	shape = original_feature.get_shape()
	# correspondingly the M_l and N_l from the paper description
	img_size = (shape[0] * shape[1]).value
	num_filters = shape[2].value
		
	G_orig = _gram_matrix(original_feature)
	G_gen  = _gram_matrix(generated_feature)
	return (1. / (4. * img_size ** 2 * 
		num_filters ** 2)) * K.sum(K.square(G_orig - G_gen))
	
def content_loss(original_features, generated_features):
	"""
	Calculates content loss, given the original content tensor and one being molded
	
	:param original_feature: Tensor corresponding to input content
	:param generated_features: Tensor corresponding to image being transformed.
		Note: All layers of the image tensor should be passed (from VGG)
	:return: Tensor corresponding to the loss as a result of content difference
	"""
	original_content  = original_features[s.CONTENT_FEATURE_LAYER]
	generated_content = generated_features[s.CONTENT_FEATURE_LAYER]
	return K.sum(K.square(original_content - generated_content))

# Credit to: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
def coherence_loss(generated_features):
	"""
	Calculates local coherence loss, given the tensor being molded
	
	:param generated_features: Tensor corresponding to image being transformed.
		Note: All layers of the image tensor should be passed (from VGG)
	:return: Tensor corresponding to the loss as a result of image being locally dissonant
	"""
	a = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, 1:, :HEIGHT-1, :])
	b = K.square(generated_features[:, :WIDTH-1, :HEIGHT-1, :] - 
		generated_features[:, :WIDTH-1, 1:, :])
	return K.sum(a + b)

# Credit to: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def transform(content_features, style_features, transform_features, 
	output_img, output_name, output_dir=s.OUTPUT_FINAL_DIR, save_intermediate=False):
	"""
	Produces an image (as numpy array) that melds both the contents and 
	style features provided. 
	
	:param content_features: Features from running VGG-19 on input content tensor
	:param style_features: Features from running VGG-19 on input style tensor
	:param transform_features: Features from running VGG-19 on empty input image tensor
	:param output_img: Tensor corresponding to interpolated image 
	:param output_name: Name (string) of file that will be outputted in the end
	:param save_intermediate: Whether or not files that are produced while running
		style transfer (that are not the final result) will be saved to disk
	:return: Numpy array for the image that interpolates the contents and styles inputted
	"""

	loss = K.variable(0.0)
	loss  +=  s.CONTENT_WEIGHT * content_loss(content_features, transform_features)
	for style_feature, transform_feature in zip(style_features, transform_features):
		loss  += s.STYLE_WEIGHT * style_loss(style_feature, transform_feature)
	loss  += s.COHERENCE_WEIGHT * coherence_loss(output_img)

	grads = K.gradients(loss, output_img)[0]

	# step size for gradient ascent
	shape = output_img.get_shape()
	img_width   = shape[1].value
	img_height  = shape[2].value
	
	grad_loss = GradLoss(K.function([output_img], [loss, grads]))
	final_img_data = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

	for i in range(s.NUM_ITERATIONS):
		final_img_data, min_val, info = fmin_l_bfgs_b(grad_loss.loss, 
			final_img_data.flatten(), fprime=grad_loss.grad, maxfun=20)

		if save_intermediate:
			imsave("{}{}-{}.png".format(output_dir, 
				output_name, i), deprocess_image(final_img_data.copy()))
		print('Current loss value:', min_val)

	# only save the actual frame if processing a video
	final_img = deprocess_image(final_img_data)
	imsave("{}{}.png".format(output_dir, output_name), final_img)
	return final_img

def img_tensor(filename):
	"""
	Produces tensor corresponding to the image located at the filename specified
	
	:param filename: Filename of the image to be read in as a tensor. Also sets the
		global WIDTH/HEIGHT variables
	:return: Tensor corresponding to input image
	"""
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

def stylize_image(content, style, content_dir=s.INPUT_CONTENT_DIR, 
	style_dir=s.INPUT_STYLE_DIR, output_dir=s.OUTPUT_FINAL_DIR):
	"""
	Stylizes the image with the content and style specified. If it is a video, the
	frame is also specified. Outputs are both returned and saved in "results" folder
	specifically "results/final" for images and "results/frames/[name]" for videos
	
	:param content: Filename of content. MUST be saved in "input/content folder"
	:param style: Filename of style. MUST be saved in "input/style folder"
	:return: Final stylized image as numpy array
	"""

	filename   = content.split(".")[0]
	file_extension = content.split(".")[-1]
	stylename  = style.split(".")[0]

	if file_extension == "gif":
		preprocess_gif(img)
		gif_content_dir  = "{}{}/".format(input_dir, filename)
		gif_output_dir = "{}{}/".format(output_dir, filename)
		gif_imgs = os.listdir(gif_content_dir)
			
		styled_frames = []
		for frame in gif_imgs:
			styled_frame = stylize_image(frame, style, content_dir=gif_content_dir, 
				style_dir=style_dir, output_dir=gif_output_dir, frame=None)
			styled_frames.append(styled_frame)
			print("Completed frame {}".format(frame))

		num_frames = len(os.listdir(gif_output_dir))
		frames = [imageio.imread("{}{}.png".format(gif_output_dir, frame)) 
			for frame in range(num_frames)]
		imageio.mimsave("{}{}.gif".format(output_dir, filename), frames)
		return styled_frames

	combined_name = "{}-{}".format(filename, stylename)
	input_file = "{}{}".format(content_dir, content)
	final_name = "{}{}".format(output_dir, combined_name)

	# holds the three-layered tensor of inputs passed in
	model_input = []

	# input image: content
	content_img_tensor = img_tensor(input_file)
	model_input.append(content_img_tensor)

	# input image: style
	style_img_tensor = img_tensor("{}{}".format(style_dir, style))
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
	style_features	   = []
	transform_features = []

	for i, layer in enumerate(layers):
		print('Processing layer {}'.format(i))
		features = model.get_layer(layer).output
		content_features.append(features[0,:,:,:])
		style_features.append(features[1,:,:,:])
		transform_features.append(features[2,:,:,:])

	final_img = transform(content_features, style_features, transform_features,
		transform_image_tensor, final_name)
	return final_img

if __name__ == "__main__":
	contents = ["bagend.jpg", "japan.jpg", "nature.jpg"]
	styles = ["bamboo.jpg", "blocks.jpg", "bokeh.jpg", "scream.jpg", "weird.jpg"]
	for content in contents:
		for style in styles:
			stylize_image(content=content, style=style)