"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Does edge segmentation using HED (pre-trained) model
"""

import numpy as np
import Image
import caffe
from scipy.misc import imsave

def segment_edges(imgs):
    for (i, img) in enumerate(imgs):
        print("Processing image {}".format(i))

        im = Image.open(img)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        
        in_ = in_.transpose((2,0,1))
        net = caffe.Net('./cache/deploy.prototxt', './cache/hed_pretrained_bsds.caffemodel', caffe.TEST)

        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        net.forward()
        # other outputs only provided if desired for debugging
        out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
        out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
        out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
        out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
        out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

        imsave("{}-edges.jpg".format(img), fuse)

if __name__ == "__main__":
    segment_edges(["2092.jpg"])