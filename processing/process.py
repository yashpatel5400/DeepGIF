"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""

from styletransfer import stylize_image, stylize_video
from styletransfer import segment_edges, download_model, segment

process