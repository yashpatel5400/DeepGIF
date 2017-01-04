"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Functions to track classes of pixels from one image to another.
"""
import numpy as np
import itertools

def trackGreedy(img1, img2, numClasses):
    """
    Track segmented masks from img1 to img2 using a greedy approach.
    
    :param img1: First image
    :param img2: Second image
    :param numClasses: Total number of classes in the image
    :return: Return the mapping from the class labels in img1 that correspond to class labels in img2
    """
    count = np.zeros((numClasses, numClasses));

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            label1 = img1[x][y]
            label2 = img2[x][y]
            count[label1][label2] += 1

    total = np.sum(count, axis = 1)
    indices = np.argsort(total)

    map = np.zeros(numClasses)
    assignedLabels = [];

    sortedArgsCount = np.argsort(count)

    for i in range(indices.shape[0]):
        label1 = indices[-i]
        j = 1
        label2 = sortedArgsCount[label1][-j]
        while label2 in assignedLabels:
            j += 1
            label2 = sortedArgsCount[label1][-j]
        assignedLabels.append(label2)
        map[label1] = label2

    return map.astype(int)

def trackExhaustive(img1, img2, numClasses):
    """
    Track segmented masks from img1 to img2 trying all possible mappings

    :param img1: First image
    :param img2: Second image
    :param numClasses: Total number of classes in the image
    :return: Return the mapping from the class labels in img1 that correspond to class labels in img2
    """
    mapping = range(numClasses)
    perms = itertools.permutations(mapping)
    maxHits = 0

    img1 = img1.flatten()
    img2 = img2.flatten()

    bestMapping = None

    for p in perms:
        # Traverse the two images, counting the number of correct matches
        p = np.array(p)
        img1mapped = p[img1]
        matches = (img1mapped == img2)
        hits = np.sum(matches)
        if (hits >= maxHits):
            maxHits = hits
            bestMapping = p

    return bestMapping.astype(int)

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
    """
    Given imgs, a numpy array of images that have already been segmenetd, 
    return a numpy array of images where the segmented labels 
    correspond to the first image's labels.
    
    :param imgs: the numpy array of images that have already been segmented
    :return: the altered image segment labels
    """
    numClasses = np.max(imgs) + 1
    for idx in range(1, imgs.shape[0]):
        mapping = trackExhaustive(imgs[idx-1], imgs[idx], numClasses)
        reverseMapping = [0 for _ in range(numClasses)]
        for i in range(len(mapping)):
            reverseMapping[mapping[i]] = i
        imgs[idx] = reverseMapping[imgs[idx]]
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