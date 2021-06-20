import cv2 as cv
import numpy as np

from .utilities import normL2

def merge_line_cluster(cluster):
    x, y = [], []
    for line in cluster:
        for point in line:
            x.append(point[0])
            y.append(point[1])
    x = np.array(x)
    y = np.array(y)

    # least squares best fit line: X*theta = Y
    n = x.shape[0]
    X = np.hstack([x.reshape(n, 1), np.ones((n, 1))])
    Y = y.reshape(n, 1)
    theta = np.matmul(np.linalg.pinv(X), Y)

    x_min = min(x)
    x_max = max(x)

    line = ((x_min, x_min * theta[0] + theta[1]), (x_max, x_max * theta[0] + theta[1]))
    return line



def greedy_line_cluster(lines, thres_parallel=0.0174532, thres_distance=10):
    n = len(lines)
    
    angles = []
    for line in lines:
        theta = np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
        angles.append(theta)

    # greedily cluster lines
    clustered = [False] * n
    clusters = []
    for i in range(n):
        if clustered[i]:
            continue
        clustered[i] = True
        cluster = [lines[i]]
        for j in range(i + 1, n):
            if (not clustered[j]) and (abs(angles[i] - angles[j]) < thres_parallel) and \
                    (find_shortest_distance_segments(lines[i], lines[j]) < thres_distance):
                cluster.append(lines[j])
                clustered[j] = True
        clusters.append(cluster)

    return clusters



def merge_batch_lines(lines, thres_parallel=0.0174532, thres_distance=10, max_iters=10):
    merging = True
    iters = 0
    merged_lines = lines
    while merging and iters < max_iters:
        merging = False
        iters += 1
        print("merging iter {:2d}".format(iters))
        lines = merged_lines
        merged_lines = []
        ## lines = (x1, y1, x2, y2)
        lines_ = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in lines]

        clusters = greedy_line_cluster(lines_, thres_parallel=thres_parallel, thres_distance=thres_distance)

        merged_lines = []
        for cluster in clusters:
            if len(cluster) > 1:
                merging = True
                line = merge_line_cluster(cluster)
            else:
                line = cluster[0]
            line = (line[0][0], line[0][1], line[1][0], line[1][1])
            line = list(map(int, line))
            merged_lines.append([line])               

    return merged_lines


def find_shortest_distance_segments(seg1, seg2):
    # segment => point1(x, y), point2(x, y)
    if do_intersect(seg1, seg2):
        return 0
    else:
        d1_20 = shortest_distance_point_seg(seg1, seg2[0])
        d1_21 = shortest_distance_point_seg(seg1, seg2[1])
        d10_2 = shortest_distance_point_seg(seg2, seg1[0])
        d11_2 = shortest_distance_point_seg(seg2, seg1[1])
        return min(d1_20, d1_21, d10_2, d11_2)


def shortest_distance_point_seg(seg, point):
    if (seg[0][0] != seg[1][0]) and (seg[0][1] != seg[1][1]):
        # this is along the perpendicular line, y=-(1/m)x+c
        m, c = get_equation_line(seg[0], seg[1])
        x_int = ((1 / m) * point[0] + point[1] - c) / (m + 1 / m)  # == (c2 - c) / (m - -1/m)
        if min(seg[0][0], seg[1][0]) <= x_int <= max(seg[0][0], seg[1][0]):
            return normL2(point, (x_int, m * x_int + c))
    elif (seg[0][0] == seg[1][0]):  # vertical line
        if min(seg[0][1], seg[1][1]) <= point[1] <= max(seg[0][1], seg[1][1]):
            return abs(seg[0][0] - point[0])
    elif (seg[0][1] == seg[1][1]):  # horizontal line
        if min(seg[0][0], seg[1][0]) <= point[0] <= max(seg[0][0], seg[1][0]):
            return abs(seg[0][1] - point[1])
    # perpendicular line not on segment, find distance to edges instead
    return min(normL2(point, seg[0]), normL2(point, seg[1]))


def get_equation_line(point1, point2):
    # equation of line: y= grad * x + y_int
    grad = (point1[1] - point2[1]) / (point1[0] - point2[0]) if point1[0] != point2[0] else np.inf
    y_int = point1[1] - grad * point1[0]  # == point2[1] - grad * point2[0]
    return grad, y_int


def do_intersect(seg1, seg2):
    # equation of line: y=m*x+c
    m1, c1 = get_equation_line(seg1[0], seg1[1])
    m2, c2 = get_equation_line(seg2[0], seg2[1])

    if m1 == m2:  # parallel lines
        no_overlap = \
            (max(seg1[0][1], seg1[1][1]) < min(seg2[0][1], seg2[1][1])) or \
            (max(seg2[0][1], seg2[1][1]) < min(seg1[0][1], seg1[1][1])) or \
            (max(seg1[0][0], seg1[1][0]) < min(seg2[0][0], seg2[1][0])) or \
            (max(seg2[0][0], seg2[1][0]) < min(seg1[0][0], seg1[1][0]))
        return not no_overlap and (c1 == c2)

    # vertical lines
    if m1 == np.inf:
        x_int = seg1[0][0]
        if (x_int > max(seg2[0][0], seg2[1][0])) or (x_int < min(seg2[0][0], seg2[1][0])):
            return False
        y_int = m2 * x_int + c2
        return min(seg1[0][1], seg1[1][1]) < y_int < max(seg1[0][1], seg1[1][1])
    elif m2 == np.inf:
        x_int = seg2[0][0]
        if (x_int > max(seg1[0][0], seg1[1][0])) or (x_int < min(seg1[0][0], seg1[1][0])):
            return None
        y_int = m1 * x_int + c1
        return min(seg2[0][1], seg2[1][1]) < y_int < max(seg2[0][1], seg2[1][1])

    # at intersection, x1=x1 and y1=y1:
    x_int = (c2 - c1) / (m1 - m2)
    if \
            (x_int > max(seg1[0][0], seg1[1][0])) \
            or (x_int < min(seg1[0][0], seg1[1][0])) \
            or (x_int > max(seg2[0][0], seg2[1][0])) \
            or (x_int < min(seg2[0][0], seg2[1][0])):
        return False
    # y_int = m1 * x_int + c1 # == m2 * x_int + c2
    return True


if __name__ == '__main__':
    img_path = "../images/sudoku-original.jpg"
    image_orig = cv.imread(img_path)
    if image_orig is None:
        print("file not found at ", img_path)
        exit()

    image = image_orig.copy()
    cv.imshow("original", image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #gray = cv.GaussianBlur(gray, (3, 3), 0)

    ### -------------------------- lines -------------------------- ###
    edges = cv.Canny(gray, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    edges = cv.dilate(edges, kernel, iterations=1)
    edges = cv.erode(edges, kernel, iterations=1)
    #edge = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)
    cv.imshow("edges", edges)
    cv.waitKey(0)

    min_length = max(image.shape) / 100
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=min_length, maxLineGap=1)
    img_lines = image.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("lines-P", img_lines)
    cv.waitKey(0)

    print("merging lines")
    print("number of lines before merging: {:d}".format(len(lines)))
    merged_lines = merge_batch_lines(lines, thres_parallel=5*np.pi/180, thres_distance=10, max_iters=10)
    print("number of lines after merging: {:d}".format(len(merged_lines)))
    print("done")

    img_lines = image.copy()
    for i in range(0, len(merged_lines)):
        l = merged_lines[i][0]
        cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("merged lines", img_lines)
    cv.waitKey(0)