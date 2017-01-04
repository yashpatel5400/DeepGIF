"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level)
"""

# -------------------- Default Transfer Parameters ---------------------------#
# weighting of the style
STYLE_WEIGHT = .75

# weighting of the content
CONTENT_WEIGHT = 0.025

# weighting of the coherence variable
COHERENCE_WEIGHT = 1.0

# layer of the network used for the content loss (4th layer in original paper)
CONTENT_FEATURE_LAYER = 3 

# gradiet ascent number of iterations (producing image)
NUM_ITERATIONS = 10

# gradient ascent step size
STEP_SIZE = 1.0

# ----------------------- Fast Transfer Parameters ---------------------------#
# median filter (same as content feature layer)
MEDIAN_FILTER = 3

# padding added to final output image: compensates for cutoff from convolutions
PADDING = 50

# ------------------------- Input Global Variables ---------------------------#
# location where the content inputs are saved
INPUT_CONTENT_DIR = './input/content/'

# location where the style inputs are saved
INPUT_STYLE_DIR = './input/style/'

# location where a video file is broke into its constituent frames
INPUT_FRAME_DIR = './input/frames/'

# location where pre-trained models for the fast style transfer are stored
FAST_MODEL_CACHE = './cache/'

# ------------------------- Output Global Variables --------------------------#
# location where analyzed/stylized video frames are saved
OUTPUT_FRAME_DIR = './results/frames/'

# location where the overall (final) style transfers are saved
OUTPUT_FINAL_DIR = './results/final/'