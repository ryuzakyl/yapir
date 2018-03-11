from math import sqrt, ceil, cos, sin
from math import factorial as fact


#ToDo: Python code here. Optimize it with cython or anything like that.
def Nnm(n, m):
    sig_nm = 1 if m == 0 else 0
    return sqrt(2 * (n + 1.0) / (1.0 + sig_nm))


#ToDo: Python code here. Optimize it with cython or anything like that.
def Rmn(m, n, rho):
    # Rnm uses the absolute value of m
    m = abs(m)

    value = 0.0
    ub_s = (n - m) // 2

    s = 0
    while s <= ub_s:
        num = fact(n - s) * pow(rho, n - 2 * s)
        den = fact(s) * fact((n + m) // 2 - s) * fact((n - m) // 2 - s)
        value += (num / den) if s % 2 == 0 else (-num / den)

        s += 1

    return value


#ToDo: Python code here. Optimize it with cython or anything like that.
def ZernikeCircularSingle(j, rho, theta):
    n = int(ceil((-3 + sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    return ZernikeCircularDouble(n, m, rho, theta)


#ToDo: Python code here. Optimize it with cython or anything like that.
def ZernikeCircularDouble(n, m, rho, theta):
    if rho < 0 or rho > 1:
        return 0

    # computing normalization factor
    nnm = Nnm(n, m)

    # computing radial contribution
    rnm = Rmn(m, n, rho)

    # computing azimuthal contribution
    azim = cos(m * theta) if m >= 0 else -sin(m * theta)

    #returning zernike circular polynomial evaluation
    return nnm * rnm * azim
