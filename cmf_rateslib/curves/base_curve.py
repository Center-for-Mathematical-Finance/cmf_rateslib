import datetime
import numpy as np


class BaseZeroCurve(object):

    _maturities: np.ndarray
    _rates: np.ndarray

    def __init__(self, maturities, rates):
        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must be os the same length")

        self._maturities = np.array(maturities)
        self._rates = np.array(rates)

    def df(self, expiry):
        return np.exp(- self.zero_rate(expiry) * expiry)

    def zero_rate(self, expiry):
        return np.interp(expiry, self._maturities, self._rates)
    
    def fwd_rate(self, expiry: float, tenor: float, n=None):
        fwd_rate = -np.log((self.df(expiry)/self.df(expiry + tenor))) / tenor
        if n:
            return n * (np.exp(fwd_rate/n) - 1)
        else:
            return fwd_rate

    def bump(self, shift):
        return BaseZeroCurve(self._maturities, self._rates + shift)

    def __add__(self, other):
        if isinstance(other, BaseZeroCurve):
            return BaseZeroCurve(self._maturities, self._rates + other.zero_rate(self._maturities))
        else:
            raise TypeError("'other' must be an instance of a BaseZeroCurve")

    def __sub__(self, other):
        if isinstance(other, BaseZeroCurve):
            return BaseZeroCurve(self._maturities, self._rates - other.zero_rate(self._maturities))
        else:
            raise TypeError("'other' must be an instance of a BaseZeroCurve")