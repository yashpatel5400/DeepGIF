"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level) for processing
"""

# ------------------------- Input Global Variables ---------------------------#
# location where the content inputs are saved
INPUT_CONTENT_DIR = './input/content/'

# location where the style inputs are saved
INPUT_STYLE_DIR = './input/style/'

# location where a video file is broke into its constituent frames
INPUT_FRAME_DIR = './input/frames/'

# -------------------- Pre-trained Model Variables ---------------------------#
# cached model filename
MODEL_FILENAME = "hed_pretrained_bsds.caffemodel"

# directory for models cache
SEGMENT_MODEL_CACHE = "./segmentation/cache/"

# location where pre-trained models for the fast style transfer are stored
TRANSFER_MODEL_CACHE = './styletransfer/cache/'

# -------------------- Output Directory Variables ---------------------------#
# directory for final output images
OUTPUT_DIR = "./results/"

# location where analyzed/stylized video frames are saved
OUTPUT_FRAME_DIR = './results/frames/'

# location where the overall (final) style transfers are saved
OUTPUT_FINAL_DIR = './results/final/'