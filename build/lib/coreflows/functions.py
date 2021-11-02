import numpy as _np
import scipy.misc as _ms

def hermite(x, l):
    """compute the hermite basis function of degree l at location x

    :param x:
    :param l:
    :return:
    """
    c = _np.zeros(40)
    c[l] = 1.
    return (2 ** l * _ms.factorial(l) * _np.pi ** 0.5) ** -0.5 * _np.exp(-x ** 2 / 2) * _np.polynomial.hermite.hermval(
        x, c)

def hermite_sum(x, coeffs, delta_x):
    """compute sum of set of hermite basis functions

    :param x: locations x  - np.array((Nx))
    :param coeffs: coefficients of basis functions  - np.array((Nc))
    :param delta_x: width parameter of basis functions  - [np.array((Nc)) or float]
    :return: function along x  - np.array((Nx))
    """
    out = _np.zeros_like(x)
    for l in range(len(coeffs)):
        out += coeffs[l] * hermite(x / delta_x, l)
    return out

def hermite_fit(x, fit_c):
    """define the hermite fit function to use

    :param x: locations - np.array((Nx))
    :param fit_c: containing coefficients [:Nc] and width parameter [-1] - np.array((Nc+1))
    :return:
    """
    return sum(x, fit_c[:-1], fit_c[-1])

def square(x,dx=30):
    """Square wave 1 for -dx < x < dx, 0 else

    :param x:
    :param dx:

    :return: function of 1 or 0
    """
    y = _np.ones_like(x)
    y[x<-dx] = 0.
    y[x>dx] = 0.
    return y

def double_sigmoid(x, dx=30, tau=1):
    """ smoothed square wave 1 for -dx << x << dx, 0 for x << -dx or x >> dx

    :param x: coordinate vector
    :param dx: coordinate where transition occurs
    :param tau: parameter to control how sharp the cutoff is
    :return:
    """
    return 1/(1+2**((x-dx)/tau))/(1+2**((-x-dx)/tau))

def symmetric_empirical_wavepower(x, delta_x):
    """empirical fit to signal produced by symmetric wave of width dth

    :param x:
    :param dth:
    :return:
    """
    return hermite(x/(delta_x+2), 0)

def asymmetric_empirical_wavepower(x, delta_x):
    """empirical fit to signal produced by asymmetric wave of width dth

    :param x:
    :param delta_th:
    :return:
    """
    return _np.abs(hermite(x/(delta_x*1.2), 0)) + _np.abs(hermite(x/(delta_x*1.2), 1))

def empirical_wavepower(x, delta_x, l):
    if l == 0:
        return symmetric_empirical_wavepower(x, delta_x)
    elif l == 1:
        return asymmetric_empirical_wavepower(x, delta_x)
    else:
        raise ValueError('l must be 0 or 1')