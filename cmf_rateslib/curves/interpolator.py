from unittest.signals import removeResult
import numpy as np


class Interpolator(object):
    __xp: np.ndarray
    __yp: np.ndarray
    __mode: str
    #what to do if value is out of board
    _left: str
    _right: str

    #splines coefficients
    __a: np.ndarray
    __b: np.ndarray
    __c: np.ndarray
    #list of spline functions
    __S: list
    #template for interpolation method
    def __interp_method(self, x):
        pass

    def get_nodes(self):
        return self.__xp.copy(), self.__yp.copy()
    def get_method(self):
        return self.__mode

    def __init__(self, x: np.ndarray, y: np.ndarray, mode: str):
        if (len(x.shape) != 1):
            raise ValueError("x must be 1d ndarray!")
        if (len(y.shape) != 1):
            raise ValueError("y must be 1d ndarray!")
        if (x.shape[0] != y.shape[0]):
            raise ValueError("x and y must be on the same length!")
        if (x.shape[0] < 2):
            raise ValueError("x is to consist of at least two points")

        self.__xp = x.copy()
        self.__yp = y.copy()
        self.__mode = mode
        self.__set_interp_method(mode)

    def __set_interp_method(self, mode):
        if (mode == "linear"):
            self.__interp_method = self._linear
        elif (mode == "log_linear"):
            self.__interp_method = self._log_linear
        elif (mode == "quadratic"):
            self._fill_spline(self.__xp, self.__yp)
            self.interp = self._piecewise_quadratic
        elif (mode == "log_quadratic"):
            self._fill_spline(self.__xp, np.log(self.__yp))
            self.interp = self._log_piecewise_quadratic
        else:
            raise Exception("Error! Unknown interpolation method!")

 
    def _fill_spline(self, x, y):
        #required to calculate spline coefficients
        beta = (y[1] - y[0]) / (x[1] - x[0])
        sz = x.shape[0] - 1

        #spline coefficients
        self.__a = np.zeros(sz + 2)
        self.__b = np.zeros(sz + 2)
        self.__c = np.zeros(sz + 2)
        #splines
        self.__S = [None] * (sz + 2)

        #splines functions and coefficients calculation
        for i in range(1, sz + 1):
            dx = x[i] - x[i - 1]
            self.__c[i] = (y[i] - y[i - 1]) / (dx * dx) - beta / dx
            self.__b[i] = beta - 2 * self.__c[i] * x[i - 1]
            self.__a[i] = y[i - 1] - self.__c[i] * x[i - 1] * x[i - 1] - self.__b[i] * x[i - 1]

            beta = self.__b[i] + 2 * self.__c[i] * x[i]
            self.__S[i] = lambda t, i=i: self.__a[i] + self.__b[i] * t + self.__c[i] * t * t

        self.__a[0] = self.__a[1]
        self.__a[-1] = self.__a[-2]
        self.__b[0] = self.__b[1]
        self.__b[-1] = self.__b[-2]
        self.__c[0] = self.__c[1]
        self.__c[-1] = self.__c[-2]
        self.__S[0] = self.__S[1]
        self.__S[-1] = self.__S[-2]
        

    def interp(self, x):
        if not (isinstance(x, np.ndarray)):
            x = np.array([x])
        return self.__interp_method(x)
 

    def _linear_int_ext(self, x, xp, yp):
        left_outer_ind = x < xp[0]
        right_outer_ind = x > xp[-1]
        res = np.interp(x, xp, yp)

        # if we out of bounds then extrapolate 
        k = (yp[1] - yp[0]) / (xp[1] - xp[0])
        b = yp[0] - k * xp[0]
        res[left_outer_ind] = x[left_outer_ind] * k + b

        k = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        b = yp[-1] - k * xp[-1]
        res[right_outer_ind] = x[right_outer_ind] * k + b

        return res
       
    def _linear(self, x):
        return self._linear_int_ext(x, self.__xp, self.__yp)

    def _log_linear(self, x):
        return np.exp(self._linear_int_ext(x, self.__xp, np.log(self.__yp)))

    def _piecewise_quadratic(self, x):
        sz = self.__xp.shape[0]
        xl = np.insert(self.__xp, 0, -np.inf)
        xr = np.append(self.__xp, np.inf)
        X, XL = np.meshgrid(x, xl)
        _, XR = np.meshgrid(x, xr)
        cond = ((XL <= X) & (X < XR)).reshape(-1, x.shape[0])

        res = np.piecewise(x, cond, self.__S)
        return res

    def _log_piecewise_quadratic(self, x):
        return np.exp(self._piecewise_quadratic(x))



























