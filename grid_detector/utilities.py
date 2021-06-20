import numpy as np
import cv2 as cv

def normL2(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_counts(arr):
    counts = {}
    for x in arr:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def apply_mask(image, contour, thickness=1, debug=False):
    mask = np.zeros((image.shape[0:2]), np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    cv.drawContours(mask, [contour], 0, 255 , thickness) # border
    if debug:
        cv.imshow("mask", mask)
        cv.waitKey(0)
    
    out = np.zeros_like(image)
    out[mask == 255] = image[mask == 255]
    return out
