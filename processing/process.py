"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Final processing file used for fully stylizing an
image with multiple masks and tracking.
"""

import settings as s

from styletransfer import fast_stylize_image
from segmentation import segment_edges, segment, mask_imgs, submask

def process_imgs(contents, styles, mask_map):
	# construct all the edge maps for the images passed in
	edges = [segment_edges(img) for img in contents]

	# segment just the first image, since this is where user input specified
	reference = segment(edges[0])

	# obtain user input for specifying the desired segments and corresponding
	# stylizations (dictionary): uses WEB interface
	# gonna be something that depends on "reference" variable
    # Also construct the stylez that are will be used...
	class_to_style = ['bamboo.jpg','bokeh.jpg']

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
		for idx in range(len(class_to_style)):
			stylized_img = stylize_image(content=img, style=class_to_style[idx])
			masked_img = np.multiply(stylized_img, mask == idx)
			cur_img_masks.append(masked_img)
        
		# squash all the masked image into a single output
		stylized_imgs.append(sum(cur_img_masks))
		
	return stylized_imgs

if __name__ == "__main__":
    contents = ["banana.gif"]
    models = ["candy.model", "cubist.model"]
    #segment_edges(contents, cache_dir=s.SEGMENT_MODEL_CACHE, 
    #	input_dir=s.INPUT_CONTENT_DIR, output_dir=s.OUTPUT_FRAME_DIR, save_output=True)
    fast_stylize_image(content_img=contents[0], model=models[0],
    	cache_dir=s.TRANSFER_MODEL_CACHE, input_dir=s.INPUT_CONTENT_DIR, 
    	output_dir=s.OUTPUT_FRAME_DIR)