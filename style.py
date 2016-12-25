"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Miscellaneous functions for calculating style loss
"""

from keras import backend as K
import numpy as np

# util function to convert a tensor into a valid image
# from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def deprocess_image(x):
	x = x.reshape((224, 224, 3))
	# Remove zero-center by mean pixel
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68

	# 'BGR'->'RGB'
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x

# from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def normalize(x):
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# the following three functions are defined by their descriptions
# from the "Style Transfer" paper
def gram_matrix(output):
	flat_output = K.batch_flatten(output)
	return K.dot(flat_output, K.transpose(flat_output))

def style_loss(originals, generated):
	cur_loss = K.variable(0.0)
	G_gen = gram_matrix(generated)
	for original in originals:
		shape = original.get_shape()
		# correspondingly the M_l and N_l from the paper description
		img_size = (shape[1] * shape[2]).value
		num_filters = shape[0].value
		
		G_orig = gram_matrix(original)
		cur_loss += (1 / (4 * img_size ** 2 * num_filters ** 2)) * K.sum(K.square(G_orig - G_gen))
	return cur_loss

# adopted from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def visualize_filters(model, layers):
	print('Processing layer {}'.format(len(layers)))

	outputs = []
	for layer in layers:
		output = model.get_layer(layer).output[0, :, :, :]
		outputs.append(K.permute_dimensions(output, (2, 1, 0)))

	input_shape = outputs[0].get_shape()
	img_width   = input_shape[1]
	img_height  = input_shape[2]

	input_img = K.placeholder((img_width, img_height, 3))
	
	# we build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	loss  = style_loss(outputs, input_img)
	grads = normalize(K.gradients(loss, input_img)[0])

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	# step size for gradient ascent
	step = 1.0

	input_img_data = np.random.random((img_width, img_height, 3)).astype(np.float32)
	input_img_data = (input_img_data - 0.5) * 20 + 128

	for i in range(25):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
		print('Current loss value:', loss_value)
			
		# some filters get stuck to 0, we can skip them
		if loss_value <= 0.:
			break

	return deprocess_image(input_img_data[0])