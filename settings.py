"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Global variable declarations (module-level)
"""

# ------------------------- Input Global Variables ---------------------------#
# assumed size of the inputs: resized to these if not provided
WIDTH  = 224
HEIGHT = 224

# layer of the network used for the content loss: 4th layer per original paper
CONTENT_FEATURE_LAYER = 3 

# location where the content inputs are saved
INPUT_CONTENT_DIR = './input/content/'

# location where the style inputs are saved
INPUT_STYLE_DIR = './input/style/'

# ------------------------- Output Global Variables --------------------------#\
# location where the content reconstructions are saved
OUTPUT_CONTENT_DIR = './results/content/'

# location where the style reconstructions are saved
OUTPUT_STYLE_DIR = './results/style/'

# location where the overall (final) style transfers are saved
OUTPUT_FINAL_DIR = './results/final/'