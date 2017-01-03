"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level) for segmentation
"""

# -------------------- Segnet Global Variables ------------------------------#
# size of convolution kernels
KERNEL = 8

# paddings for the convolutions
PAD = 1

# side length for the max pooling window
POOL_SIZE = 2

# -------------------- Input Img Global Variables ---------------------------#
# training images split (in corresponding input directory)
TRAIN = 'train/'

# test images split (in corresponding input directory)
TEST = 'test/'

# validation images split (in corresponding input directory)
VAL = 'val/'

# -------------------- Input Directory Global Variables ---------------------#
# number of segmentations for each raw image
RAW_SEGMENTATION_SIZE = 4

# raw images and matrix files
RAW_INPUT_DIR = './input/raw/'

# edge files (i.e. for training N4 and UNet)
EDGE_INPUT_DIR = './input/edges/'

# segmented files (i.e. for training SegNet)
SEGMENTS_INPUT_DIR = './input/segments/'

# -------------------- Input Data File Extensions ---------------------------#
# extensions for how the truths are provided in DB
INPUT_MAT = 'mat'