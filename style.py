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

def gram_matrix(output):
	# makes first dimension the number of filters
	switch_output = K.permute_dimensions(output, (2, 0, 1))
	flat_output = K.batch_flatten(switch_output)
	return K.dot(flat_output, K.transpose(flat_output))

def style_loss(original, generated):
	# correspondingly the M_l and N_l from the paper description
	img_size = (original.shape[0] * original.shape[1]).value
	num_filters = original.shape[2]

	G_orig = gram_matrix(original)
	G_gen = gram_matrix(generated)

	return (1 / (4 * img_size ** 2)) * K.sum(K.square(G_orig - G_gen))

def content_loss(original, generated):
	return K.sum(K.square(original - generated))

# adopted from: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
def visualize_filters(input_img, layer):
	SEARCH_FILTERS = 64

	imgs = []
	img_width  = input_img.get_shape()[1]
	img_height = input_img.get_shape()[2]

	for filter_index in range(SEARCH_FILTERS):
		print('Processing filter %d' % filter_index)
		original = layer.output[:, :, :, filter_index]

		# we build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		loss  = style_loss(original)
		grads = normalize(K.gradients(loss, input_img)[0])

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# step size for gradient ascent
		step = 1.0

		input_img_data = np.random.random((1, img_width, img_height, 3))
		input_img_data = (input_img_data - 0.5) * 20 + 128

		for i in range(25):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

			print('Current loss value:', loss_value)
			# some filters get stuck to 0, we can skip them
			if loss_value <= 0.:
				break

		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			imgs.append((img, loss_value))

	return imgs