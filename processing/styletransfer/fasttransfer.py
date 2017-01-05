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

def fast_stylize_image(content_img, model):
    filename   = content_img.split(".")[0]
    modelname  = model.split(".")[0]

    img_name   = "{}{}".format(s.INPUT_CONTENT_DIR, content_img)
    model_name = "{}{}".format(s.FAST_MODEL_CACHE, model)

    if not os.path.exists(model_name):
        print("Note: A pre-trained model is REQUIRED for fast style transfer). \n"
        + "Please clone: https://github.com/gafr/chainer-fast-neuralstyle-models and transfer "
        + "contents of 'model' folder into the 'cache' folder")
        return

    model = FastStyleNet()
    serializers.load_npz(model_name, model)

    start = time.time()
    original = Image.open(img_name).convert('RGB')

    image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    image = np.pad(image, [[0, 0], [0, 0], [s.PADDING, s.PADDING], [s.PADDING, s.PADDING]], 'symmetric')
    image = np.asarray(image)
    
    x = Variable(image)
    y = model(x)
    result = cuda.to_cpu(y.data)

    result = result[:, :, s.PADDING:-s.PADDING, s.PADDING:-s.PADDING]
    result = np.uint8(result[0].transpose((1, 2, 0)))

    med = Image.fromarray(result)
    med = med.filter(ImageFilter.MedianFilter(s.MEDIAN_FILTER))
    print(time.time() - start, 'sec')
    med.save("{}{}-{}.jpg".format(s.OUTPUT_FINAL_DIR, modelname, filename))
    return med

if __name__ == "__main__":
    contents = ["bagend.jpg", "city.jpg"]
    models = ["candy.model", "cubist.model", "starry.model", "seurat.model", "kanagawa.model"]
    for content in contents:
        for model in models:
            fast_stylize_image(content, model)