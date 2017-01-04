"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Client for using the segmentation portion of the
program, including doing the segmentation and watershed on input images.
"""

import click
from model import segment_edges, download_model