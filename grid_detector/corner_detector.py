import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from .utilities import get_counts
from .transforms import fitRectangle, fitParallelogram, fourPointTransform


def detect_grid_corners(image_in, blocksize=2, ksize=3, k=0.04, n_outlier=1, debug=False):
    image = image_in.copy()
    gray = image if (len(image.shape)==2) else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.cornerHarris(gray, blocksize, ksize, k)
    retval, corners = cv.threshold(corners, corners.max() * 0.01, 255, cv.THRESH_BINARY)
    corners = np.uint8(corners)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners)

    if debug:
        if len(image.shape) == 3:
            image[corners > 0.01 * corners.max()] = [0, 0, 255]
        else:
            image[corners > 0.01 * corners.max()] = 255
        cv.imshow("all corners", image)
        cv.waitKey(0)

    # throw away outliers: https://towardsdatascience.com/outlier-detection-python-cd22e6a12098
    radius = max(gray.shape)
    model = DBSCAN(eps=0.03*radius, min_samples=1).fit(centroids)
    if debug:
        fig, ax = plt.subplots()
        x, y = list(zip(*centroids))
        ax.scatter(x, y, c=model.labels_)
    counts = get_counts(model.labels_)
    n = len(centroids)
    labels = model.labels_.copy()
    for i in range(n-1, -1, -1):
        label = model.labels_[i]
        if counts[label] <= n_outlier:
            centroids = np.delete(centroids, i, axis=0)
            labels = np.delete(labels, i, axis=0)

    par = fitParallelogram(centroids.reshape(-1, 2))
    if debug:
        rect = fitRectangle(centroids.reshape(-1, 2))
        ax = plt.gca()
        x, y = list(zip(*rect))
        x += (x[0], )
        y += (y[0], )
        ax.plot(x, y, 'r-')
        x, y = list(zip(*par))
        x += (x[0],)
        y += (y[0],)
        ax.plot(x, y, 'r-')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.pause(1)  # note: interferes with opencv

    return par


if __name__ == '__main__':
    filename = "../images/sudoku-original.jpg" 
    image_orig = cv.imread(filename)
    if image_orig is None:
        print("file not found at ", filename)
        exit()

    image = image_orig.copy()
    cv.imshow("original", image)
    cv.waitKey(0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #gray = cv.GaussianBlur(gray, (5, 5), 0)

    corners = detect_grid_corners(gray, debug=True)
    for c in corners:
        cv.drawMarker(image, (int(c[0]), int(c[1])), (0, 255, 0), cv.MARKER_CROSS, markerSize=10, thickness=3)
    cv.waitKey(0)
