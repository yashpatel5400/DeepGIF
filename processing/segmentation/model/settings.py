"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level) for segmentation
"""

# -------------------- Input Img Global Variables ---------------------------#
# training images split (in corresponding input directory)
TRAIN = 'train/'

# test images split (in corresponding input directory)
TEST = 'test/'

# -------------------- Input Directory Global Variables ---------------------#
# raw images and matrix files
INPUT_DIR = './input/'

# -------------------- Output Directory Variables ---------------------------#
# website where the pre-trained model can be downloaded
MODEL_SITE = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'

# directory for models cache
MODEL_CACHE = "./model/cache/"

# metadata about the model needed to load pretrained Caffe models
MODEL_META = "deploy.prototxt"

# cached model filename
MODEL_FILENAME = "hed_pretrained_bsds.caffemodel"

# directory for final output images
OUTPUT_DIR = "./results/"

# -------------------- Input Data File Extensions ---------------------------#
# extensions for how the truths are provided in DB
INPUT_MAT = 'mat'