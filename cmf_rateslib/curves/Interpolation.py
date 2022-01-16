import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

x = [0, 2, 5, 10]
y = [0.01, 0.02, 0.03, 0.04]
z = 2.5

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
                return ((y[i - 1] - y[i])*z + (x[i-1]*y[i] - x[i]*y[i-1]))/-(x[i] - x[i-1])

inter = lin_interpolate(x, y, z)
print(inter)
plt.figure(figsize = (10,8))
plt.plot(x, y, '-ob')
plt.plot(z, inter, 'ro')
plt.title('Linear Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Addition with library
#def lin_interpolate2(x: np.ndarray, y: np.ndarray, z: float):
    #if x[0] < 0:
        #raise ValueError('x values must be non-negative')
    #if z > x[-1]:
        #raise ValueError('z must be within x range')
    #f = interp1d(x, y)
    #return f(z)