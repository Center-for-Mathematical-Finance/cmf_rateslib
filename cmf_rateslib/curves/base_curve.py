import datetime
import numpy as np


class BaseZeroCurve(object):

    _maturities: np.ndarray
    _rates: np.ndarray

    def __init__(self, maturities, rates):
        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must be os the same length")

        maturities = np.array(maturities) #for maturities to be sorted ascending
        sorted_args = maturities.argsort()
        self._maturities = maturities[sorted_args]
        self._rates = np.array(rates)[sorted_args]

    def df(self, expiry):
        return np.exp(- self.zero_rate(expiry) * expiry)

    def zero_rate(self, expiry):
        return np.interp(expiry, self._maturities, self._rates)
    
    def fwd_rate(self, expiry: float, tenor: float):
        return -np.log((self.df(expiry)/self.df(expiry + tenor))) / tenor

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