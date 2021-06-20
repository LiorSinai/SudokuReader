# tutorials 
# -  https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
# - https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
# - https://aishack.in/tutorials/sudoku-grabber-opencv-detection/

import cv2 as cv
import numpy as np

from .transforms import fitParallelogram, fitRectangle, fourPointTransform, warpPoints
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


def projectGrid(image, width, height, M, thickness=1, gridsize=3, color=(0, 255, 255)):    
    lines = [((0, 0), (width, 0)), ((0, 0), (0, height))]
    for i in range(gridsize + 1):
        h = i * height/gridsize
        lines.append(((0, h), (width, h)))
        w = i * width / gridsize
        lines.append(((w, 0), (w, height)))
    
    for line in lines:
        warped_line = warpPoints(line, M)
        cv.polylines(image, [warped_line], False, color, thickness=thickness)


def detectRectangle(image, boxsize=5, 
        blocksize=3, ksize=3, k=0.04, thres_corners=0.001,
        debug=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # remove background with the largest contour
    thresholded = gray.copy()
    thresholded = preprocess(thresholded, boxsize=boxsize)
    if debug:
        cv.imshow("01-thresholded", thresholded)
        cv.waitKey(0)
    # get largest contour -> assume to be grid
    contours, hierarchy = cv.findContours(thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)
    gray = apply_mask(gray, max_contour, thickness=5)
    if debug:
        cv.imshow("02-masked gray", gray)
        cv.waitKey(0)

    # get corners
    corners = cv.cornerHarris(gray, blocksize, ksize, k)
    if debug:
        image_with_corners = image.copy()
        image_with_corners[corners > thres_corners * corners.max()] = (0, 0, 255)
        cv.imshow("03-corners image", image_with_corners)  
        cv.waitKey(0)
    ret, corners = cv.threshold(corners, thres_corners * corners.max(), 255, 0)
    corners = np.uint8(corners)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners)
    
    parallelogram = fitParallelogram(centroids.reshape(-1, 2))
    if debug:
        image_with_borders = image.copy()
        rect = fitRectangle(centroids.reshape(-1, 2))
        cv.polylines(image_with_borders, np.int32([parallelogram]), True, (0,255,255))
        cv.polylines(image_with_borders, np.int32([rect]), True, (0,255,255)) 
        cv.imshow("04-borders", image_with_borders) 
        cv.waitKey(0)
    return parallelogram



if __name__ == '__main__':
    img_path = "images/sudoku_aishack.jpg"
    image_orig  = cv.imread(img_path)
    if image_orig is None:
        print("file not found at ", img_path)
        exit()
    shape = image_orig.shape
    cv.imshow("Image", image_orig)
    cv.waitKey(0)

    image = image_orig.copy()

    image = image_orig.copy()
    parallelogram = detectRectangle(image, debug=True)

    warped, M = fourPointTransform(image, parallelogram)
    cv.imshow("warped", warped)
    cv.waitKey(0)

    Minv = np.linalg.inv(M)
    image_with_grid = image_orig.copy()
    projectGrid(image_with_grid, warped.shape[1], warped.shape[0], Minv)
    cv.imshow("warped grid", image_with_grid)
    cv.waitKey(0)

    ## rewarping -> distortion because of interpolation
    image = image_orig.copy()
    warped_shape = warped.shape
    nwarps = 100
    for i in range(nwarps):
        image = cv.warpPerspective(image, M, (warped_shape[1], warped_shape[0]))
        image = cv.warpPerspective(image, Minv, shape[:2])

    cv.imshow("warped {}x".format(nwarps), image)
    cv.waitKey(0)





    
