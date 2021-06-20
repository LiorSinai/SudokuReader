import cv2 as cv
import numpy as np


def fitRectangle(points):
    # return corners in top-left, top-right, bottom-right, bottom-left
    max_x = max([p[0] for p in points])
    min_x = min([p[0] for p in points])
    max_y = max([p[1] for p in points])
    min_y = min([p[1] for p in points])

    corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    return np.array(corners)


def fitParallelogram(points):
	 # return corners in top-left, top-right, bottom-right, bottom-left
	rect = fitRectangle(points)
	
	corners = rect.copy()
	dists = [np.inf, np.inf, np.inf, np.inf]
	
	for point in points:
		for i in range(4):
			d = abs(point[0] - rect[i][0]) + abs(point[1] - rect[i][1])
			if d < dists[i]:
				corners[i] = point
				dists[i] = d
	
	return corners


def areaParallelogram(points):
	base = abs(points[1][0] - points[0][0])
	height = abs(points[3][1] - points[0][1])
	return base * height


def orderPoints(pts):
	# order points: top-left, top-right, bottom-right, bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def fourPointTransform(image, pts):
	# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	rect = orderPoints(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image
	widthA = (br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2
	widthB = (tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2
	maxWidth = int(np.sqrt(max(widthA, widthB)))

	# compute the height of the new image
	heightA = (tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2
	heightB = (tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2
	maxHeight = int(np.sqrt(max(heightA, heightB)))	

	destination = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32"
        )

	# compute the perspective and apply transform matrix
	M = cv.getPerspectiveTransform(rect, destination)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped, M


def warpPoints(points, M):
	out = []
	for pt in points:
		x, y = pt
		denom = (M[2][0] * x + M[2][1] * y + M[2][2])
		xt = (M[0][0] * x + M[0][1] * y + M[0][2]) / denom
		yt = (M[1][0] * x + M[1][1] * y + M[1][2]) / denom
		out.append((round(xt), round(yt)))
	return np.array(out, np.int32)


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


def projectText(image, text, centre, M, fontScale=1, fontThickness=2, color=(0, 255, 255)):
    centre_warped = warpPoints([centre], M)
    textSize = cv.getTextSize(
		text=text, 
        fontFace=cv.FONT_HERSHEY_SIMPLEX, 
        fontScale=fontScale, 
        thickness=fontThickness)
    pos = (centre_warped[0][0] - textSize[0][0]//2, centre_warped[0][1] + textSize[0][1]//2)
    cv.putText(image, text, pos, cv.FONT_HERSHEY_SIMPLEX, fontScale, color, 
        thickness=fontThickness)


def resize(image, max_dim=1080):
	height, width = image.shape[0], image.shape[1]
	if height > max_dim:
		width = int((max_dim/height) * width)
		height = max_dim
	elif width > max_dim:
		height = int((max_dim/width) * height)
		width = max_dim
	else:
		return image
	out = cv.resize(image, (width, height))
	return out