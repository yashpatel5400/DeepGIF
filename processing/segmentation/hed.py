"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Does edge segmentation using HED (pre-trained) model
"""

import settings as s
from preprocess import preprocess_gif, download_model, add_padding, remove_padding

import os
import numpy as np
import caffe
from scipy.misc import imsave
from PIL import Image

def segment_edges(imgs, cache_dir=s.MODEL_CACHE, 
    input_dir=s.INPUT_DIR, output_dir=s.OUTPUT_DIR, save_output=True):
    """
    Given a list of images, returns the edge segmentations of the images. They will
    be returned corresponding to the order inputted and can be saved to the default
    directory if so desired (primarily for debugging)
    
    :param imgs: The filenames (iterable of strings) of the images. Note that these
        MUST be stored in the "input" directory under "segmentation"
    :return: List of numpy arrays corresponding to the edge segmentations of input
    """
    pretrained_meta = '{}{}'.format(cache_dir, s.MODEL_META)
    pretrained_model = '{}{}'.format(cache_dir, s.MODEL_FILENAME)

    if not os.path.exists(pretrained_model):
        print("Downloading the pre-trained model")
        download_model()

    edges = []
    net = caffe.Net(pretrained_meta, pretrained_model, caffe.TEST)
    for (i, img) in enumerate(imgs):
        filename = img.split(".")[0]
        file_extension = img.split(".")[-1]
        if file_extension == "gif":
            preprocess_gif(img, input_dir=input_dir, output_dir=output_dir)
            gif_input_dir  = "{}{}/".format(input_dir, filename)
            gif_output_dir = "{}{}/".format(output_dir, filename)
            gif_imgs = os.listdir(gif_input_dir)

            gif_edges = segment_edges(gif_imgs, cache_dir=cache_dir, 
                input_dir=gif_input_dir, output_dir=gif_output_dir, 
                save_output=save_output)
            edges.append(gif_edges)

        else:
            print("Processing image {}".format(i + 1))

            im = Image.open("{}{}".format(input_dir, img))
            pad_im  = add_padding(im)
            img_arr = np.array(pad_im, dtype=np.float32)
            img_arr = img_arr[:,:,::-1]
            img_arr -= np.array((104.00698793,116.66876762,122.67891434))
            
            img_arr = img_arr.transpose((2,0,1))
            net.blobs['data'].reshape(1, *img_arr.shape)
            net.blobs['data'].data[...] = img_arr

            net.forward()

            # all outputs provided if desired for experimenting
            out = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
            if save_output:
                output_img = "{}{}.png".format(output_dir, filename)
                imsave(output_img, out)
                remove_padding(output_img)
            edges.append(out)
    return edges

if __name__ == "__main__":
    segment_edges(["banana.gif"])