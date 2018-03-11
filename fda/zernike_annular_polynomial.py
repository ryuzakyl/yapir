from cmath import sqrt as c_sqrt
from math import factorial as fact
from math import sqrt, ceil, cos, sin

import fda.zernike_circular_polynomial as zern_circ


#ToDo: Python code here. Optimize it with cython or anything like that.
def c_Rmn(m, n, rho):
    # Rnm uses the absolute value of m
    m = abs(m)

    value = 0.0
    ub_s = (n - m) // 2

    s = 0
    while s <= ub_s:
        num = fact(n - s)
        den = fact(s) * fact((n + m) // 2 - s) * fact((n - m) // 2 - s)
        tmp = (num / den) if s % 2 == 0 else (-num / den)

        power = n - 2 * s

        value += tmp * pow(rho, power).real

        s += 1

    return value


#ToDo: Python code here. Optimize it with cython or anything like that.
def h(m, j, eps):
    if m == 0:
        return (1 - pow(eps, 2)) / (2 * (2 * j + 1))

    num = 2 * (2 * j + 2 * m - 1) * Q(m - 1, j + 1, 0, eps) * h(m - 1, j, eps)
    den = (j + m) * (1 - pow(eps, 2)) * Q(m - 1, j, 0, eps)

    return -(num / den)


#ToDo: Refactor code because it is written for debug purposes
#ToDo: Python code here. Optimize it with cython or anything like that.
def Q(m, j, u, eps):
    if m == 0:
        new_rho = c_sqrt((u - pow(eps, 2)) / (1 - pow(eps, 2)))
        return c_Rmn(0, 2 * j, new_rho)

    h_val = h(m - 1, j, eps)
    q_val = Q(m - 1, j, 0, eps)

    num = 2 * (2 * j + 2 * m - 1) * h_val
    den = (j + m) * (1 - pow(eps, 2)) * q_val

    result = num / den

    cum_sum = 0.0
    i = 0
    while i <= j:
        q_val_1 = Q(m - 1, i, 0, eps)
        q_val_2 = Q(m - 1, i, u, eps)

        h_val_1 = h(m - 1, i, eps)

        cum_sum += (q_val_1 * q_val_2) / h_val_1

        i += 1

    return result * cum_sum


#ToDo: Refactor code because it is written for debug purposes
#ToDo: Python code here. Optimize it with cython or anything like that.
def Rmn(m, n, rho, eps):
    # case:	eps = 0
    if eps == 0:
        return zern_circ.Rmn(m, n, rho)     # evaluating in the Rmn of the circular polynomial

    # Rnm uses the absolute value of m
    m = abs(m)

    # case: m = 0 and n even
    if m == 0 and n % 2 == 0:
        num = (pow(rho, 2) - pow(eps, 2))
        den = (1 - pow(eps, 2))

        new_rho = sqrt(num / den)
        return zern_circ.Rmn(0, n, new_rho)

    # case: n = m
    if n == m:
        cum_sum = 0.0

        i = 0
        while i <= n:
            cum_sum += pow(eps, 2 * i)
            i += 1

        return pow(rho, n) / sqrt(cum_sum)

    # general case
    j = (n - m) // 2
    u = pow(rho, 2)

    q_val = Q(m, j, u, eps)
    h_val = h(m, j, eps)

    rad = (1 - pow(eps, 2)) / (2 * (2 * j + m + 1) * h_val)

    return sqrt(rad) * pow(rho, m) * q_val


def ZernikeAnnularSingle(j, rho, theta, eps_lb, eps_ub):
    n = int(ceil((-3 + sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)

    return ZernikeAnnularDouble(n, m, rho, theta, eps_lb, eps_ub)


def ZernikeAnnularDouble(n, m, rho, theta, eps_lb, eps_ub):
    if rho < eps_lb or rho > eps_ub:
        return 0.0

    # computing normalization constant
    nnm = zern_circ.Nnm(n, m)

    # computing radial contribution
    rnm = Rmn(m, n, rho, eps_lb)

    # computing azimuthal contribution
    if m == 0:
        azim = 1
    else:
        azim = cos(m * theta) if m > 0 else -sin(m * theta)

    #returning zernike circular polynomial evaluation
    return nnm * rnm * azim
