import numpy.linalg as alg


def euclidean_distance(x, y):
    size_x = len(x)
    size_y = len(y)

    # if the size is not the same, return None
    if size_x != size_y:
        return None

    return alg.norm(x - y)
