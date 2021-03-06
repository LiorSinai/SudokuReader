import numpy as np
import cv2 as cv


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


def padRectangle(max_shape, top, bottom,  left, right, pad):
    height, width = max_shape[0], max_shape[1]
    top = max(0, int(top - pad))
    bottom = min(height, int(bottom + pad))
    left = max(0, int(left - pad))
    right = min(width, int(right + pad))
    return top, bottom, left, right