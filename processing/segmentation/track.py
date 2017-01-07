"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Functions to track classes of pixels from one image to another.
"""
import numpy as np
import itertools

def trackGreedy(img1, img2):
    '''
    Track segmented masks from img1 to img2 using a greedy approach.
    :param img1: First image
    :param img2: Second image
    :param numClasses: Total number of classes in the image
    :return: Return the mapping from the class labels in img1 that correspond to class labels in img2
    '''
    img1Classes = np.max(img1) + 1
    img2Classes = np.max(img2) + 1
    count = np.zeros((img1Classes, img2Classes));

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            label1 = img1[x][y]
            label2 = img2[x][y]
            count[label1][label2] += 1

    map = np.zeros(img2Classes)
    map[0] = 0 # Background label is always 0

    total = np.sum(count, axis = 0)
    indices = np.argsort(total)

    sortedArgsCount = np.argsort(count, axis = 0)

    for i in range(indices.shape[0]-1, -1, -1):
        label2 = indices[i]
        if label2 != 0:
            label1 = sortedArgsCount[-1][label2]
            if label1 == 0 and img1Classes > 1:
                label1 = sortedArgsCount[-2][label2]
            map[label2] = label1

    return map.astype(int)

def submask(masked_img, classes):
    """
    Given an image mask (2D numpy array) and a list of classes, produces a 
    submask of the image where only those pixels with entry being one of
    the ones specified in the classes list are retained
    
    :param masked_img: the numpy array corresponding to a mask
    :param classes: list of classes to be retained from masked_img
    :return: numpy array of same dimensions as masked image, with only the
        values from classes
    """
    submask = masked_img.copy()
    shape = imgs.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            super_mask_val = masked_img[i][j]
            submask[i][j]  = super_mask_val * int(super_mask_val in classes)
    return submask

def mask_imgs(imgs):
    '''
    Given imgs, a numpy array of images that have already been segmenetd, return a numpy array of images where the
    segmented labels correspond to the first image's labels.
    :param imgs: the numpy array of images that have already been segmented
    :return: the altered image segment labels
    '''
    for idx in range(1, imgs.shape[0]):
        mapping = trackGreedy(imgs[idx-1], imgs[idx])
        imgs[idx] = mapping[imgs[idx]]
    return imgs

def test():
    img1 = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0]])
    img2 = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]])
    n = 2
    print trackExhaustive(img1, img2, n)
    print trackGreedy(img1, img2, n)


if __name__ == "__main__":
    test()