"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""
from segmentation import segment, mask_imgs, hed
from styletransfer import fasttransfer

import cv2
import numpy as np
import os

from PIL import Image

import imageio
import math

# Initialize directories
content_dir = 'input/content/pooh/'
frames_dir = 'input/frames/pooh/'
edges_dir = 'input/edges/pooh/'
styledFrames_dir = 'results/frames/pooh/'

(width, height) = cv2.imread(frames_dir+'0.jpg', flags=cv2.IMREAD_GRAYSCALE).shape

# Get the frames
frames = [cv2.imread(frames_dir+str(i)+'.jpg')for i in range(len(os.listdir(edges_dir)))]

# Initialize the styles we want to used
styles = os.listdir('styletransfer/cache')[1:]

# Detect the edges of the frames
imgFrames = [str(i)+'.jpg' for i in range(len(os.listdir(frames_dir)))]
hed.segment_edges(imgFrames, input_dir = frames_dir, output_dir = edges_dir)

# Load the edges of the frames
edges = [cv2.imread(edges_dir+str(i)+'.png', flags=cv2.IMREAD_GRAYSCALE) for i in range(len(os.listdir(edges_dir)))]

# Unpad edges
dw = (edges[0].shape[0] - width)/2.0
dh = (edges[0].shape[1] - height)/2.0
for i in range(len(edges)):
    img = edges[i]
    edges[i] = img[math.ceil(dw):-math.floor(dw), math.ceil(dh):-math.floor(dh)]

# Segment the edges and create the masks
segmented = np.array([segment(img) for img in edges])
masks = mask_imgs(segmented)

# Show the segmented images
import matplotlib.pyplot as plt
for mask in masks[0:1]:
    print mask
    imgplot = plt.imshow(mask)
    imgplot.set_cmap('spectral')
    plt.show()

# Apply the Style Transfer
styled_imgs = [[] for _ in range(len(imgFrames))]
for (frame, filename) in zip(styled_imgs, imgFrames):
    for style in styles:
        frame.append(fasttransfer.fast_stylize_image(filename, style, cache_dir = 'styletransfer/cache/', input_dir = frames_dir, output_dir = styledFrames_dir))

# Load the styled images
styled_imgs = [[Image.open(styledFrames_dir+style.split('.')[0]+'-'+str(i)+'.png').convert('RGB') for style in styles] for i in range(len(os.listdir(frames_dir)))]

# Make the images numpy arrays
styled_arr = [[np.array(img) for img in frame] for frame in styled_imgs]

# The resulting images from the style transfer algorithm are all 
# not the same dimension as the original images so we need to pad
# each image by some column of 0s
for frame in styled_arr:
    for idx in range(len(frame)):
        paddedImg = np.zeros((width, height, 3))
        dw = width - frame[idx].shape[0]
        dh = height - frame[idx].shape[1]
        paddedImg[dw:, dh:, :] = frame[idx]
        frame[idx] = paddedImg

# Choose the label to style mapping
numLabels = len(np.unique(masks[0]))
corrStyle = [2 for _ in range(numLabels)]
corrStyle[0] = 5

# Mask the frames and styles
final_frames = []
for (frame, mask) in zip(styled_arr, masks):
    dup_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    dup_mask[:, :, 0] = mask
    dup_mask[:, :, 1] = mask
    dup_mask[:, :, 2] = mask
    mask = dup_mask
    final_img = np.zeros(mask.shape)
    for idx in range(numLabels):
        masked_img = np.multiply(frame[corrStyle[idx]], mask == idx)
        final_img += masked_img
    final_frames.append(final_img)

# Convert the final images to PIL Images
for idx in range(len(final_frames)):
    final_frames[idx] = Image.fromarray(np.uint8(final_frames[idx]))

# Show final images
for img in final_frames:
    img.show()

# Convert the final frames to uin8
for idx in range(len(final_frames)):
    final_frames[idx] = np.uint8(final_frames[idx])
	
# Generate the final gif
imageio.mimsave('NYC.gif', final_frames, fps = 10)