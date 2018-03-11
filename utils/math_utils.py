from math import sqrt, pow, fabs

DBL_MAX = 1.7976931348623158e+308   #taken from Visual C++

DBL_EPS = 0.0000000000001


def euclidean_distance_coords(x1, y1, x2, y2):
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def euclidean_distance_points(p1, p2):
    return euclidean_distance_coords(p1.x(), p1.y(), p2.x(), p2.y())


def euclidean_distance_arrays(x, y):
    hx, wx = x.shape
    hy, wy = y.shape

    if wx != 1 or wy != 1 or hx != hy:
        return DBL_MAX

    result = 0.0

    for i in range(hx):
        xi = x.item(i, 0)
        yi = y.item(i, 0)

        result += pow(xi - yi, 2)

    return sqrt(result)


def fit_parabola_coords(x1, y1, x2, y2, x3, y3):
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)

    #avoiding zero division
    if d == 0:
        return -1, -1, -1

    A = (x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3 - y2) ) / d
    B = (x3*x3*(y1 - y2) + x2*x2*(y3 - y1) + x1*x1*(y2 - y3) ) / d
    C = (x2*x3*(x2 - x3)*y1 + x3*x1*(x3 - x1)*y2 + x1*x2*(x1 - x2)*y3 ) / d

    return A, B, C


def fit_parabola_points(p1, p2, p3):
    x1 = p1.x()
    y1 = p1.y()

    x2 = p2.x()
    y2 = p2.y()

    x3 = p3.x()
    y3 = p3.y()

    return fit_parabola_coords(x1, y1, x2, y2, x3, y3)


def in_parabola_coords(A, B, C, x, y):
    is_convex = (A > 0)
    yp = A*x*x + B*x + C

    if is_convex:
        return y > yp
    else:
        return y < yp


def in_parabola_point(A, B, C, p):
    x1 = p.x()
    y1 = p.y()

    return in_parabola_coords(x1, y1)


def is_between_parabolas_coords(parabola_1, parabola_2, x, y):
    if parabola_1 is None and parabola_2 is None:
        return 1

    if parabola_1 is None:
        A2, B2, C2 = parabola_2
        return in_parabola_coords(A2, B2, C2, x, y)

    if parabola_2 is None:
        A1, B1, C1 = parabola_1
        return in_parabola_coords(A1, B1, C1, x, y)

    #both parabolas are valid and we assume one is convex and the other is concave
    A1, B1, C1 = parabola_1
    A2, B2, C2 = parabola_2
    return in_parabola_coords(A1, B1, C1, x, y) and in_parabola_coords(A2, B2, C2, x, y)


def is_between_parabolas_point(parabola_1, parabola_2, p):
    x = p.x()
    y = p.y()

    return is_between_parabolas_coords(parabola_1, parabola_2, x, y)


def too_near(x1, y1, x2, y2):
    return fabs(x1 - x2) < DBL_EPS and fabs(y1 - y2) < DBL_EPS


def compute_circle_center_coords(x1, y1, x2, y2, x3, y3):
    if too_near(x1, y1, x2, y2) or too_near(x2, y2, x3, y3) or too_near(x3, y3, x1, y1):
        return -1, -1

    d = 2 * (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2)

    h = ( (pow(x1,2.0) + pow(y1,2.0))*(y2 - y3) + (pow(x2,2.0) + pow(y2,2.0))*(y3 - y1) + (pow(x3,2.0) + pow(y3,2.0))*(y1 - y2) ) / d
    k = ( (pow(x1,2.0) + pow(y1,2.0))*(x3 - x2) + (pow(x2,2.0) + pow(y2,2.0))*(x1 - x3) + (pow(x3,2.0) + pow(y3,2.0))*(x2 - x1) ) / d

    return int(round(h, 0)), int(round(k, 0))


def compute_circle_center_points(p1, p2, p3):
    x1 = p1.x()
    y1 = p1.y()

    x2 = p2.x()
    y2 = p2.y()

    x3 = p3.x()
    y3 = p3.y()

    return compute_circle_center_coords(x1, y1, x2, y2, x3, y3)

