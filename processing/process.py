"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""

from styletransfer import stylize_image, stylize_video
from segmentation import segment_edges, mask_imgs

def process_imgs(contents, styles, mask_map):
	edges = [segment_edges(img) for img in contents]
	masks = mask_imgs(edges)

	stylized_imgs = []
	for (img, mask) in zip(imgs, masks):
		cur_img_masks = []
		for style in styles:
			stylized_img = stylize_image(content=img, style=style)
			style_mask   = mask[style]
			masked_img   = np.multiply(style_mask, style_mask)
			cur_img_masks.append(masked_img)