"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level) for SegNet
"""

# -------------------- Simulation Global Variables ---------------------------#
# size of convolution kernels
KERNEL = 8

# paddings for the convolutions
PAD = 1

# side length for the max pooling window
POOL_SIZE = 2

# number of training epochs
NB_EPOCH = 100

# batch size for training input
BATCH_SIZE = 14

# ----------------------- Simulation Input Image -----------------------------#
# input/output image width
WIDTH = 320

# input/output image height
HEIGHT = 480

# ----------------------- Network Training Parameters ------------------------#
# number of output classes (softmax over 12 classifications)
NUM_CLASSES = 12

# number of training images
TRAINING_INP_SIZE = 367

# training/testing/validation data directory
DATA_DIRECTORY = './CamVid/'

# ignores the root directory when obtaining training data
IGNORE_ROOT = 7

# weights caching (after training)
OUTPUT_FILE = 'model_weights.hdf5'