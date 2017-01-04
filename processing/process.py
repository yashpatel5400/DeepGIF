"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""

from styletransfer import stylize_image, stylize_video
from segmentation import segment_edges, segment, mask_imgs, submask

def process_imgs(contents, styles, mask_map):
	# construct all the edge maps for the images passed in
	edges = [segment_edges(img) for img in contents]

	# segment just the first image, since this is where user input specified
	reference = segment(edges[0])

	# obtain user input for specifying the desired segments and corresponding
	# stylizations (dictionary): uses WEB interface
	# gonna be something that depends on "reference" variable
	class_to_style = {
		0: 'bamboo.jpg',
		1: 'bokeh.jpg'
	}

	# construct corresponding mask for the first image (from the desired segment):
	# remains CONSTANT between the different input images
	classes = class_to_style.keys()
	submasked_input = submask(reference, classes)
	
	# construct full sequence of masks
	segmented = [submasked_input] + [segment(edge) for edge in edges[1:]]
	masks = mask_imgs(segmented)

	# iterate through each of the images, each with their corresponding masks
	stylized_imgs = []
	for (img, mask) in zip(imgs, masks):

		# iterate through each of the styles and apply them to the current image
		cur_img_masks = []
		for style in styles:
			stylized_img = stylize_image(content=img, style=style)
			style_mask   = mask[style]
			masked_img   = np.multiply(style_mask, style_mask)
			cur_img_masks.append(masked_img)

		# squash all the masked image into a single output

	return stylized_imgs