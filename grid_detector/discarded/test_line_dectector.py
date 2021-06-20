import matplotlib.pyplot as plt
from .line_detector import do_intersect, get_equation_line
import numpy as np

def test_do_intersect():
    segments = [
        (((1, 1), (5, 5)), ((4, 2), (2, 6)), True),  # intersect lines
        (((1, 1), (2, 7)), ((4, 2), (2, 6)), False), # too short
        (((1, 1), (2, 5)), ((4, 2), (2, 6)), False), # too short
        (((4, 2), (2, 6)), ((1, 1), (2, 5)), False),
        (((1, 1), (5, 5)), ((1, 3), (5, 7)), False), # parallel,
        (((1, 1), (5, 5)), ((2, 2), (6, 6)), True),  # parallel
        (((1, 1), (5, 6)), ((1, 3), (5, 7)), False), # just off parallel
        (((1, 1), (5, 1)), ((1, 3), (5, 7)), False), # horizontal
        (((1, 4), (5, 4)), ((1, 3), (5, 7)), True), # horizontal
        (((1, 1), (5, 5)), ((4, 3), (4, 7)), True), # vertical
        (((1, 1), (1, 5)), ((4, 4.5), (4, 7)), False), # vertical
        (((4, 1), (4, 5)), ((4, 4.5), (4, 7)), True), # vertical
        (((4, 1), (4, 4)), ((4, 4.5), (4, 7)), False), # vertical
        (((1, 1), (8, 1)), ((4, 4.5), (4, 7)), False), # vertical + horizontal,
        (((1, 6), (8, 6)), ((4, 4.5), (4, 7)), True), # vertical + horizontal
        (((4, 4.5), (4, 7)), ((1, 10), (8, 10)), False), # vertical + horizontal,
        (((4, 4), (4, 4)), ((1, 4), (8, 10)), False), # point
        (((4, 4), (4, 4)), ((1, 5), (1, 5)), False), # point - point,
        (((1, 1), (5, 1)), ((2, 4), (5, 4)), False),  # horizontal - horizontal
        (((1, 1), (4, 1)), ((2, 1), (5, 1)), True),  # horizontal - horizontal
    ]
    results = []
    for seg1, seg2, ans in segments:
        fig, ax = plt.subplots()
        ax.plot([seg1[0][0], seg1[1][0]], [seg1[0][1], seg1[1][1]], 'b-', marker='x')
        ax.plot([seg2[0][0], seg2[1][0]], [seg2[0][1], seg2[1][1]], 'r-', marker='x')
        result = do_intersect(seg1, seg2)
        int_point = find_intersection(seg1, seg2)
        print(ans == result, ans)
        results.append(ans == result)
        ax.set_title("intersects: " + str(result))
        if int_point:
            ax.plot(int_point[0], int_point[1], marker='d')
        shortest_line = find_shortest_line_segments(seg1, seg2)
        ax.plot(
            [shortest_line[0][0], shortest_line[1][0]], 
            [shortest_line[0][1], shortest_line[1][1]], 
            'k--x')
        ax.set_aspect('equal')
    print("{:d} tests passed, {:d} failed".format(results.count(True), results.count(False)))


def find_shortest_line_segments(seg1, seg2):
    if do_intersect(seg1, seg2):
        point = find_intersection(seg1, seg2) 
        return (point, point)
    else:
        d1_20 = shortest_line_point_seg(seg1, seg2[0])
        d1_21 = shortest_line_point_seg(seg1, seg2[1])
        d10_2 = shortest_line_point_seg(seg2, seg1[0])
        d11_2 = shortest_line_point_seg(seg2, seg1[1])
        shortest =  min(d1_20, d1_21, d10_2, d11_2, key=lambda x: x[1])
        return shortest[0]
    

def normL2(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def shortest_line_point_seg(seg, point):
    if (seg[0][0] != seg[1][0]) and (seg[0][1] != seg[1][1]):
        # this is along the perpendicular line, y=-(1/m)x+c
        m, c = get_equation_line(seg[0], seg[1])
        x_int = ((1/m) * point[0] + point[1] - c) / (m + 1/m)  # == (c2 - c) / (m - -1/m)
        if min(seg[0][0], seg[1][0]) <= x_int <= max(seg[0][0], seg[1][0]):
            y_int = m * x_int + c
            return(point, (x_int, y_int)), normL2(point, (x_int, y_int))
    elif (seg[0][0] == seg[1][0]): # vertical line
        if min(seg[0][1], seg[1][1]) <= point[1] <= max(seg[0][1], seg[1][1]):
            return (point, (seg[0][0], point[1])), abs(seg[0][0] - point[0])
    elif (seg[0][1] == seg[1][1]): # horizontal line
        if min(seg[0][0], seg[1][0]) <= point[0] <= max(seg[0][0], seg[1][0]):
            return (point, (point[0], seg[0][1])), abs(seg[0][1] - point[1])
    # perpendicular line not on segment, find distance to edges instead
    d0 = normL2(point, seg[0])
    d1 = normL2(point, seg[1])
    if d0 < d1:
        return (point, seg[0]), d0
    else:
        return (point, seg[1]), d1


def find_intersection(seg1, seg2):
    # equation of line: y=m*x+c
    m1, c1 = get_equation_line(seg1[0], seg1[1])
    m2, c2 = get_equation_line(seg2[0], seg2[1])

    if m1 == m2:  # parallel lines
        no_overlap = \
            (max(seg1[0][1], seg1[1][1]) < min(seg2[0][1], seg2[1][1])) or \
            (max(seg2[0][1], seg2[1][1]) < min(seg1[0][1], seg1[1][1])) or \
            (max(seg1[0][0], seg1[1][0]) < min(seg2[0][0], seg2[1][0])) or \
            (max(seg2[0][0], seg2[1][0]) < min(seg1[0][0], seg1[1][0]))
        if no_overlap or (c1 != c2):
            return None
        # many points overlap, return midpoint of overlap
        y_int_max = min(max(seg1[0][1], seg1[1][1]), max(seg2[0][1], seg2[1][1]))
        y_int_min = max(min(seg1[0][1], seg1[1][1]), min(seg2[0][1], seg2[1][1]))
        x_int_max = min(max(seg1[0][0], seg1[1][0]), max(seg2[0][0], seg2[1][0]))
        x_int_min = max(min(seg1[0][0], seg1[1][0]), min(seg2[0][0], seg2[1][0]))
        return (x_int_min + x_int_max)/2, (y_int_min + y_int_max)/2

    # vertical lines
    if m1 == np.inf:
        x_int = seg1[0][0]
        if (x_int > max(seg2[0][0], seg2[1][0])) or (x_int < min(seg2[0][0], seg2[1][0])):
            return None
        y_int = m2 * x_int + c2
        if min(seg1[0][1], seg1[1][1]) < y_int < max(seg1[0][1], seg1[1][1]):
            return (x_int, y_int)
        return None
    elif m2 == np.inf:
        x_int = seg2[0][0]
        y_int = m1 * x_int + c1
        if min(seg2[0][1], seg2[1][1]) < y_int < max(seg2[0][1], seg2[1][1]) :
            return (x_int, y_int)
        return None

    if m1 == m2:  # parallel lines
        if m1 == 0:  # both horizontal
            if seg1[0][1] != seg2[0][1] or \
                    (max(seg1[0][0], seg1[1][0]) < min(seg2[0][0], seg2[1][0])) or \
                    (max(seg2[0][0], seg2[1][0]) < min(seg1[0][1], seg1[1][0])):
                return None
            y_int = seg1[0][1]
            x_int = min(max(seg1[0][0], seg1[1][0]), max(seg2[0][0], seg2[1][0]))
            return (x_int, y_int)

        return None

    # at intersection, x1=x1 and y1=y1:
    x_int = (c2 - c1) / (m1 - m2)
    if (x_int > max(seg1[0][0], seg1[1][0])) \
        or (x_int < min(seg1[0][0], seg1[1][0])) \
        or (x_int > max(seg2[0][0], seg2[1][0])) \
        or (x_int < min(seg2[0][0], seg2[1][0])):
        return None
    # y_int = m1 * x_int + c1 # == m2 * x_int + c2
    return (x_int,  m1 * x_int + c1)


if __name__ == '__main__':
    test_do_intersect()

    plt.show()
    