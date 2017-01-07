"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""
from segmentation import segment, mask_imgs

import cv2
import numpy as np

edges = []
for i in range(2):
    edges.append(cv2.imread('imgs/Pooh/'+str(i)+'.jpg', flags=cv2.IMREAD_GRAYSCALE))

segmented = np.array([segment(img) for img in edges])
masks = mask_imgs(segmented)

import matplotlib.pyplot as plt
for mask in masks:
    print mask
    imgplot = plt.imshow(mask)
    imgplot.set_cmap('spectral')
    plt.show()


from styletransfer import fast_stylize_image