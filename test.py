import settings as s
from style import *

import tensorflow as tf
from keras import backend as K
from scipy.misc import imsave
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

import PIL
from PIL import Image

base_model = VGG19(weights='imagenet')
intermediates = [name for name in 
	[layer.name for layer in base_model.layers] if "pool" in name]

filename = 'starry_night.jpg'	
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
combined_tensor = K.concatenate(model_input)

intermediate = intermediates[1]
intermediate_model = Model(input=base_model.input, 
	output=base_model.get_layer(intermediate).output)
# pool_features = intermediate_model.predict(x)

layers = intermediates[:1+1]
outputs = []
for layer in layers:
	output = base_model.get_layer(layer).output[0, :, :, :]
	outputs.append(K.permute_dimensions(output, (2, 1, 0)))

input_shape = base_model.get_layer(layers[0]).input.get_shape()
img_width   = input_shape[1]
img_height  = input_shape[2]

img = visualize_filters(base_model.get_layer(intermediate))



a = pool_features[0,:,:,0]
b = pool_features[0,:,:,1]
gram = 0
for i in range(112):
	for j in range(112):
		gram += a[i][j] * b[i][j]

filter_output = pool_features[0,:,:,3]
img = Image.fromarray(filter_output)
img = img.resize((224,224), PIL.Image.ANTIALIAS).convert('RGB')
	
img.save('test_pil.png')
imsave('test_sci.png', filter_output)

# ======================================================
def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# ======================================================
# we build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer = base_model.get_layer(intermediate)
filter_index = 1
input_img = base_model.input

loss_1 = K.mean(layer.output[:, :, :, filter_index])
grads_1 = normalize(K.gradients(loss_1, input_img)[0])

loss_2 = K.mean(pool_features[:, :, :, filter_index])
grads_1 = normalize(K.gradients(loss_1, inp)[0])

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
# ======================================================