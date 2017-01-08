"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Does fast neural style transfer from:
https://github.com/yusuketomoto/chainer-fast-neuralstyle
"""
from __future__ import print_function
import settings as s
from preprocess import preprocess_gif

import os
import numpy as np
from PIL import Image, ImageFilter
import time
import imageio 

import cv2
import chainer
from chainer import cuda, Variable, serializers
from net import FastStyleNet

def fast_stylize_image(content_img, model, cache_dir=s.FAST_MODEL_CACHE, 
    input_dir=s.INPUT_CONTENT_DIR, output_dir=s.OUTPUT_FINAL_DIR):

    filename   = content_img.split(".")[0]
    file_extension = content_img.split(".")[-1]
    modelname  = model.split(".")[0]

    img_name   = "{}{}".format(input_dir, content_img)
    model_name = "{}{}".format(cache_dir, model)
    final_name = "{}{}-{}.png".format(output_dir, modelname, filename)

    if file_extension == "gif":
        # preprocess_gif(content_img)
        gif_input_dir  = "{}{}/".format(input_dir, filename)
        gif_output_dir = "{}{}/".format(output_dir, filename)
        gif_imgs = os.listdir(gif_input_dir)

        styled_frames = []
        for frame in gif_imgs:
            styled_frame = fast_stylize_image(frame, model, cache_dir=cache_dir, 
                input_dir=gif_input_dir, output_dir=gif_output_dir)
            styled_frames.append(styled_frame)
            print("Completed frame {}".format(frame))

        num_frames = len(os.listdir(gif_output_dir))
        frames = [imageio.imread("{}{}-{}.png".format(gif_output_dir, modelname, frame))
            for frame in range(num_frames)]

        gif_name = "{}{}.gif".format(output_dir, filename)
        print(gif_name)
        imageio.mimsave(gif_name, frames)
        return styled_frames

    if not os.path.exists(model_name):
        print("Note: A pre-trained model is REQUIRED for fast style transfer). \n"
        + "Please clone: https://github.com/gafr/chainer-fast-neuralstyle-models and transfer "
        + "contents of 'model' folder into the 'cache' folder")
        return
    
    model = FastStyleNet()
    serializers.load_npz(model_name, model)

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
    med.convert('P', colors=256).save(final_name)
    return med

if __name__ == "__main__":
    contents = ["bagend.jpg", "city.jpg"]
    models = ["candy.model", "cubist.model", "starry.model", "seurat.model", "kanagawa.model"]
    for content in contents:
        for model in models:
            fast_stylize_image(content, model)