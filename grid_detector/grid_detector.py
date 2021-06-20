# tutorials 
# -  https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
# - https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
# - https://aishack.in/tutorials/sudoku-grabber-opencv-detection/

import cv2 as cv
import numpy as np

from .transforms import fitParallelogram, fitRectangle, fourPointTransform, warpPoints, resize, projectGrid
from .utilities import apply_mask

def preprocess(image, boxsize=5, debug=False):
    # remove noise
    image = cv.GaussianBlur(image, (boxsize, boxsize), 0)
    if debug:
        cv.imshow("gray", image)

    # threshold
    image = cv.adaptiveThreshold(image, 255,  cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, boxsize, 2)
    if debug:
        cv.imshow("thresh", image)

    return image


def detectRectangle(image, boxsize=5, debug=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # remove background with the largest contour
    thresholded = gray.copy()
    thresholded = preprocess(thresholded, boxsize=boxsize)
    if debug:
        cv.imshow("01-thresholded", thresholded)
        cv.waitKey(0)
    # get largest contour -> assume to be grid
    contours, hierarchy = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    gray = apply_mask(gray, max_contour, thickness=5)
    if debug:
        cv.imshow("02-masked gray", gray)
        cv.waitKey(0)
    
    parallelogram = fitParallelogram(max_contour.reshape(-1, 2))
    if debug:
        image_with_borders = image.copy()
        rect = fitRectangle(max_contour.reshape(-1, 2))
        cv.polylines(image_with_borders, np.int32([parallelogram]), True, (0,255,255))
        cv.polylines(image_with_borders, np.int32([rect]), True, (0,255,255)) 
        cv.imshow("03-borders", image_with_borders) 
        cv.waitKey(0)
    return parallelogram


if __name__ == '__main__':
    img_path = "images/sudoku_aishack.jpg"
    image_orig  = cv.imread(img_path)
    if image_orig is None:
        print("file not found at ", img_path)
        exit()
    image = image_orig.copy()
    image = resize(image, max_dim=840)
    cv.imshow("Image", image)
    cv.waitKey(0)

    parallelogram = detectRectangle(image, debug=True)

    warped, M = fourPointTransform(image, parallelogram)
    cv.imshow("warped", warped)
    cv.waitKey(0)

    Minv = np.linalg.inv(M)
    image_with_grid = image.copy()
    projectGrid(image_with_grid, warped.shape[1], warped.shape[0], Minv)
    cv.imshow("warped grid", image_with_grid)
    cv.waitKey(0)

    ## rewarping -> distortion because of interpolation
    image = image.copy()
    orig_shape = image.shape
    warped_shape = warped.shape
    nwarps = 100
    for i in range(nwarps):
        image = cv.warpPerspective(image, M, (warped_shape[1], warped_shape[0]))
        image = cv.warpPerspective(image, Minv, (orig_shape[1], orig_shape[0]))

    cv.imshow("warped {}x".format(nwarps), image)
    cv.waitKey(0)





    
