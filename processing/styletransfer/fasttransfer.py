"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Does fast neural style transfer from:
https://github.com/yusuketomoto/chainer-fast-neuralstyle
"""
from __future__ import print_function
import settings as s

import os
import numpy as np
from PIL import Image, ImageFilter
import time

import chainer
from chainer import cuda, Variable, serializers
from net import FastStyleNet

# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

def fast_stylize_image(content_img, model, use_gpu=False):
    filename   = content_img.split(".")[0]
    img_name   = "{}{}".format(s.INPUT_CONTENT_DIR, content_img)
    model_name = "{}{}".format(s.FAST_MODEL_CACHE, model)

    if not os.path.exists(model_name):
        print("Note: A pre-trained model is REQUIRED for fast style transfer). \n"
        + "Please clone: https://github.com/gafr/chainer-fast-neuralstyle-models and transfer "
        + "contents of 'model' folder into the 'cache' folder")
        return

    model = FastStyleNet()
    serializers.load_npz(model_name, model)
    if use_gpu:
        cuda.get_device(1).use()
        model.to_gpu()
    xp = np if (not use_gpu) else cuda.cupy

    start = time.time()
    original = Image.open(img_name).convert('RGB')

    image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    image = np.pad(image, [[0, 0], [0, 0], [s.PADDING, s.PADDING], [s.PADDING, s.PADDING]], 'symmetric')
    image = xp.asarray(image)
    
    x = Variable(image)
    y = model(x)
    result = cuda.to_cpu(y.data)

    result = result[:, :, s.PADDING:-s.PADDING, s.PADDING:-s.PADDING]
    result = np.uint8(result[0].transpose((1, 2, 0)))

    med = Image.fromarray(result)
    med = med.filter(ImageFilter.MedianFilter(s.MEDIAN_FILTER))
    print(time.time() - start, 'sec')
    med.save("{}{}.jpg".format(s.OUTPUT_FINAL_DIR, filename))
    return med

if __name__ == "__main__":
    fast_stylize_image("buildings.jpg", "candy.model")