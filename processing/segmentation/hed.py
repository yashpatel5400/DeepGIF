"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Does edge segmentation using HED (pre-trained) model
"""

import settings as s

import os
import cv2
import numpy as np
import Image
import caffe
from scipy.misc import imsave

def download_model():
    file_name = s.MODEL_SITE.split('/')[-1]
    u = urllib2.urlopen(s.MODEL_SITE)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: {} Bytes: {}".format(file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status)+1)
        print(status)
    
    f.close()
    shutil.move("./{}".format(file_name), "{}{}".format(s.MODEL_CACHE, file_name))

def normalize_img(img_file):
    img  = cv2.imread(img_file)
    dest = np.zeros(img.shape)
    norm_img = cv2.normalize(img, dest, alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_img

def segment_edges(imgs):
    pretrained_meta = '{}{}'.format(s.MODEL_CACHE, s.MODEL_META)
    pretrained_model = '{}{}'.format(s.MODEL_CACHE, s.MODEL_FILENAME)

    if not os.path.exists(pretrained_model):
        print("Downloading the pre-trained model")
        download_model()

    net = caffe.Net(pretrained_meta, pretrained_model, caffe.TEST)
    for (i, img) in enumerate(imgs):
        filename = img.split(".")[0]
        print("Processing image {}".format(i + 1))

        im = Image.open("{}{}".format(s.INPUT_DIR, img))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        
        in_ = in_.transpose((2,0,1))
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        net.forward()

        # all outputs provided if desired for experimenting
        out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
        out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
        out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
        out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
        out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
        imsave("{}{}.jpg".format(s.OUTPUT_DIR, filename), out2)

if __name__ == "__main__":
    segment_edges(["0.jpg", "buildings.jpg", "italy.png"])