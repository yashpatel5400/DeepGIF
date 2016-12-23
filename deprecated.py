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


def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def visualize_filters(input_img, layer):
	SEARCH_FILTERS = 64

	kept_filters = []
	img_width = input_img.get_shape()[1]
	img_height = input_img.get_shape()[2]

	for filter_index in range(SEARCH_FILTERS):
		print('Processing filter %d' % filter_index)
		
		# we build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		loss = K.mean(layer.output[:, :, :, filter_index])
		grads = normalize(K.gradients(loss, input_img)[0])

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# step size for gradient ascent
		step = 1.0

		# we start from a gray image with some random noise
		input_img_data = np.random.random((1, img_width, img_height, 3))
		input_img_data = (input_img_data - 0.5) * 20 + 128

		for i in range(25):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

			print('Current loss value:', loss_value)
			# some filters get stuck to 0, we can skip them
			if loss_value <= 0.:
				break

		# decode the resulting input image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append((img, loss_value))

	n = 4

	# the filters that have the highest loss are assumed to be better-looking.
	kept_filters.sort(key=lambda x: x[1], reverse=True)
	kept_filters = kept_filters[:n * n]

	margin = 5
	width = n * img_width + (n - 1) * margin
	height = n * img_height + (n - 1) * margin
	stitched_filters = np.zeros((width, height, 3))

	for i in range(n):
		for j in range(n):
			img, loss = kept_filters[i * n + j]
			stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
							 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
	return stitched_filters