"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Function to convert the edges of an image to its segmentation.
"""

import matplotlib
import matplotlib.pyplot as plt

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
import numpy as np

from skimage import measure
import cv2

def segment(edges):
    """
    Segments the greyscale edgemap image, img.
    
    :param edges: The greyscale edgemap
    :return: The labeled segmented image
    """

    # Perform local Otsu segmentation on the image
    radius = 15
    selem = disk(radius)
    local_otsu = rank.otsu(edges, selem)
    threshold_global_otsu = threshold_otsu(edges)
    img = 10*(edges >= local_otsu).astype(np.uint8)

    # Get connected components
    all_labels = measure.label(img)
    blobs_labels = measure.label(img, background=-1)

    # Identify the most populous label as the background label
    counts = np.bincount(blobs_labels.flatten())
    backgroundLabel = np.argmax(counts)

    # Divide connected components into two groups - background components and foreground components
    separated = 255*(blobs_labels != backgroundLabel).astype(np.uint8)

    # Perform connected components segmentation again
    blobs_labels = measure.label(separated, background=0)

    """
    Uncomment the following code to show the segmented image
    """
    matplotlib.rcParams['font.size'] = 9
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(blobs_labels, cmap='spectral')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return blobs_labels

def test():
    img = cv2.imread('r2d2/1.jpg-edges.jpg', flags = cv2.IMREAD_GRAYSCALE)
    segmentedImg = segment(img)

if __name__ == "__main__":
    test()