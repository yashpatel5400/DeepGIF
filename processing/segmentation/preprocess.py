"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Preprocessing code used for set of for segmentation.
Mostly an assortment of miscellaneous image cleanup and transformation methods
"""

import settings as s

import urllib2
import shutil
import os

import cv2
from PIL import Image

def preprocess_gif(gif, input_dir=s.INPUT_DIR, output_dir=s.OUTPUT_DIR):
    """
    Breaks a GIF file into constituent frame images and saves them in the
    specified directory. If no directory specified, saves to default input/frames.
    :return: Filename of output directory
    """
    filename = gif.split(".")[0]
    
    frame_dir = "{}{}".format(input_dir, filename)
    output_dir = "{}{}".format(output_dir, filename)

    print(frame_dir)
    print("===================================")
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv2.VideoCapture("{}/{}".format(input_dir, gif))
    success, image = vidcap.read()
    count = 0
    while success:
        print('Read a new frame: {}'.format(success))
        cv2.imwrite("{}/{}.jpg".format(frame_dir, count), image)
        success, image = vidcap.read()
        count += 1

def download_model(cache_dir=s.MODEL_CACHE):
    """
    Downloads the pre-trained HED model from Berkeley website. Should display 
    progress as download taking place
    :return: None
    """
    file_name = s.MODEL_SITE.split('/')[-1]
    urllib.urlretrieve(s.MODEL_SITE, file_name)
    shutil.move("./{}".format(file_name), "{}{}".format(cache_dir, file_name))

def add_padding(im):
    (width, height) = im.size
    padded_width  = int(width * 1.25)
    padded_height = int(height * 1.25)

    delta_width = (padded_width - width)/2
    delta_height = (padded_height - height)/2

    padded_im = Image.new("RGB", (padded_width, padded_height))
    padded_im.paste(im, (delta_width, delta_height))
    return padded_im

def remove_padding(img_file):
    """
    Removes the grey artifact left from running the HED neural net on an image,
    overwriting the specified image with cropped version
    
    :param img_file: The filename (string) of image (WITH corresponding directory)
    :return: None
    """
    print("Postprocessing image")
    img = Image.open(img_file)
    (width, height) = img.size
    cropped = img.crop((int(.10 * width), int(.10 * height), width, height))
    cropped.save(img_file)
