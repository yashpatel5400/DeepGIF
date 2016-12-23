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
	
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

intermediate = intermediates[0]
intermediate_model = Model(input=base_model.input, 
	output=base_model.get_layer(intermediate).output)
pool_features = intermediate_model.predict(x)

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