import numpy as np


def lin_interpolate(x: np.ndarray, y: np.ndarray, z: float):
    if x[0] < 0:
        raise ValueError('x values must be non-negative')
    if z > x[-1]:
        raise ValueError('z must be within x range')
    for i in range(len(x)):
        if x[i - 1] == x[i]:
            raise ValueError('all values in x must be unique')
        if z < x[0]:
            if x[0] != 0:
                return y[0]*z/x[0]
            else:
                raise ValueError('z must be non-negative')
        if z == x[i]:
            return y[i]
        else:
            if x[i - 1] < z < x[i]:
                return y[i - 1]*(x[i] - z) + y[i]*(z - x[i - 1])