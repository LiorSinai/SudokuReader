
"""
See tutorials at:
- https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
- https://aishack.in/tutorials/sudoku-grabber-opencv-plot/
"""

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from grid_detector.grid_detector import detectRectangle
from grid_detector.transforms import fourPointTransform, resize, projectText, projectGrid
from grid_detector.utilities import padRectangle
from number_detector.train_recogniser import load_CNN_model


def predictNumber(image, model):
    image = cv.resize(image, (28, 28))
    image = image.reshape((1, 28, 28, 1)) /255
    confidences = model.predict(image)
    max_conf = np.max(confidences)
    pred = np.argmax(confidences)
    return pred, max_conf


def extractDigit(image, min_area_ratio=0.01, min_centre_overlap_ratio=0.1, radius_ratio=0.25, pad_ratio=0.2, debug=False):
    ret, im_thres = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(im_thres, 8)
    height, width = im_thres.shape[0], im_thres.shape[1]  

    RoI = None
    centre = None
    for idx in range(1, len(centroids)):
        if stats[idx, cv.CC_STAT_AREA]/(width * height) < min_area_ratio:
            continue        
        img_label = np.zeros(image.shape)
        img_label[labels == idx] = 255
        overlap = calcCentreOverlap(img_label, radius_ratio=radius_ratio)
        if debug:
            print("area: {:.2f}%".format(stats[idx, cv.CC_STAT_AREA]/(width * height) * 100))
            print("centroid at: ({:.2f}, {:.2f})".format(centroids[idx][0]/width, centroids[idx][1]/height))
            print("overlap: {:.2f}%".format(overlap*100))
        if overlap > min_centre_overlap_ratio:
            ## exact digit
            top_left = (stats[idx, cv.CC_STAT_LEFT], stats[idx, cv.CC_STAT_TOP])
            width_label = stats[idx, cv.CC_STAT_WIDTH]
            height_label = stats[idx, cv.CC_STAT_HEIGHT]
            pad = int(pad_ratio * max(width_label, height_label))
            ## make square
            centre = (top_left[0] + width_label//2, top_left[1] + height_label//2)
            #centre = (int(centroids[idx][0]), int(centroids[idx][1]))
            length = max(width_label, height_label)
            top, bottom, left, right = padRectangle(
                (height, width), 
                centre[1] - length//2, centre[1] + length//2,
                centre[0] - length//2, centre[0] + length//2, 
                pad)
            # remove all other objects and enhance constrast
            im_thres[labels != idx] = 0 
            im_thres[labels == idx] = 255  
            RoI = im_thres[top:bottom, left:right]
            break
    return RoI, centre


def calcCentreOverlap(image, radius_ratio=0.25):
    """
    Calculate the amount in middle based on overlap with a circle
    """
    height, width = image.shape[0], image.shape[1]
    kernel = np.zeros((height, width))
    radius = int(min(width, height) * radius_ratio)
    kernel = cv.circle(kernel, (width//2, height//2), radius, 255, -1)
    overlap = np.sum((kernel * image) > 0) / (np.pi * radius * radius)
    return overlap


def collateImages(images, cell_shape, grid_shape):
    for i, im in enumerate(images):
        if im.shape != cell_shape:
            images[i] = cv.resize(im, cell_shape)
    im_rows = []
    n_col = grid_shape[1]
    for i in range(grid_shape[0]):
        row = np.concatenate(images[(i * n_col): ((i + 1) * n_col)], axis=1)
        im_rows.append(row)
    out = np.concatenate(im_rows, axis=0)
    return out


def sudokuDetector(image_in, model, confidence_thres=0.9, grid_color=(0, 255, 255), pad_ratio=0.2, debug=True):
    image = image_in.copy()
    GRID_SIZE = 9
    ## variables to return
    grid = np.zeros((GRID_SIZE, GRID_SIZE), np.int32)
    confidences = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    # extract grid outline
    parallelogram = detectRectangle(image, debug=False)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    warped, M = fourPointTransform(gray, parallelogram)
    if np.linalg.det(M) == 0: # singular matrix
        return image, grid, confidences
    # display on original image
    Minv = np.linalg.inv(M)
    height, width = warped.shape[0], warped.shape[1]
    projectGrid(image, width, height, Minv, gridsize=3, color=grid_color, thickness=2)
    if debug:
        cv.imshow("image with grid", image)
        cv.imshow("warped", warped)
        cv.waitKey(0)
    # predict numbers and dispaly
    step_size_w = width / GRID_SIZE
    step_size_h = height / GRID_SIZE
    pad = pad_ratio * min(width, height) / GRID_SIZE
    img_extracted = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            top, bottom, left, right = padRectangle(
                (height, width), 
                step_size_h * i, step_size_h * (i + 1), 
                step_size_w * j, step_size_w * (j + 1),
                pad)
            search_area = warped[top:bottom, left:right]
            search_area, centre = extractDigit(search_area, debug=False)
            if debug:
                if search_area is not None:
                    img_extracted.append(search_area)
                else:
                    img_extracted.append(np.zeros((28, 28)))
            if search_area is not None:
                predicted_num, confidence = predictNumber(search_area, model)
                print("({:d}, {:d}) predicted {:d} with confidence {:.2f}%".format(i, j, predicted_num, confidence*100))
                if confidence < confidence_thres:
                    continue
                centre = (left + centre[0], top + centre[1])
                projectText(image, str(predicted_num), centre, Minv, color=grid_color)   
                grid[i][j] = predicted_num
                confidences[i][j] = confidence

    if debug:
        img_extracted = collateImages(img_extracted, (28, 28), (9, 9))
        cv.imshow("extracted digits", img_extracted)
        cv.waitKey(0)
    return image, grid, confidences


if __name__ == '__main__':
    img_path = "images/sudoku_thinkfun.jpg"
    image_orig  = cv.imread(img_path)
    if image_orig is None:
        print("file not found at ", img_path)
        exit()
    image = image_orig.copy()
    image = resize(image, max_dim=840)  # GaussianBlur and adaptiveThreshold are scale dependent
    cv.imshow("Image", image)
    cv.waitKey(0)
    confidence_thres = 0.85
    grid_color = (0, 255, 255)

    print("loading model ...")
    model = load_CNN_model('number_detector/models/CNN_4_74k/')
    print("model loaded!")
    
    image, grid, confidences = sudokuDetector(
        image, model, 
        confidence_thres=confidence_thres, 
        grid_color=grid_color, 
        debug=False)
    cv.imshow("image with numbers", image)

    name, ext = os.path.splitext(img_path)
    cv.imwrite(name + "_detected"  + ext, image)
    cv.waitKey(0)
    

    ## using video
    # capture = cv.VideoCapture(0)
    # if not capture.isOpened():
    #     raise IOError("Cannot open webcam")
    # while True:
    #     ret, frame = capture.read()
    #     img_sudoku, grid, confidences = sudokuDetector(frame, model, confidence_thres=confidence_thres, debug=False)
    #     cv.imshow('input', frame)
    #     if np.sum(grid) > 0:
    #         cv.imshow('detected', img_sudoku)

    #     c = cv.waitKey(1)
    #     if c == 27:
    #         break

    