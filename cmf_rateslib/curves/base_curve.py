import datetime
import numpy as np
import scipy
from scipy.interpolate import interp1d


class BaseZeroCurve(object):
    _maturities: np.ndarray
    _rates: np.ndarray

    def __init__(self, maturities, rates, interp_method='L-R'):
        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must be os the same length")

        self._maturities = np.array(maturities)
        self._rates = np.array(rates)
        self._interp_method = interp_method.split('-')

        if self._maturities[0] != 0:
            self._maturities = np.insert(self._maturities, 0, [0])
            self._rates = np.insert(self._rates, 0, self._rates.min())

        self._discounts_cont = self.df(self._maturities)
        self._forward_start = self._maturities[:-1]
        self._forward_end = self._maturities[1:]
        self._forward_tenor = self._forward_end - self._forward_start

        self.tenor_to_periods = {'1D': 1 / 360, '3M': 0.25, '6M': 0.5, '1Y': 1.0}

        self._forward_cont = self.fwd_rate(self._forward_start, self._forward_tenor)
        # L-R-
        if len(self._interp_method) == 3:
            self.values = \
                {
                    'R': [self._maturities, self._rates],
                    'LDF': [self._maturities, np.log(self._discounts_cont)],
                    'F': [self.fwd_rate(self._forward_start, self.tenor_to_periods[self._interp_method[2]])]
                    #        else self.fwd_rate(self._forward_start, self.tenor_to_periods[self._interp_method[2]],
                    #                           int(self._interp_method[3]))]
                }
        elif len(self._interp_method) == 2:
            self.values = \
                {
                    'R': [self._maturities, self._rates],
                    'LDF': [self._maturities, np.log(self._discounts_cont)],
                    'F': [self.fwd_rate(self._forward_start, 0.25, 0.25)]
                    #        else self.fwd_rate(self._forward_start, self.tenor_to_periods[self._interp_method[2]],
                    #                           int(self._interp_method[3]))]
                }
        self.interp_type = {'L': 'linear', 'Q': 'quadratic'}
        self.result = interp1d(*self.values[self._interp_method[1]],
                               self.interp_type[self._interp_method[0]])

    def df(self, expiry):
        return np.exp(- self.zero_rate(expiry) * expiry)

    def zero_rate(self, expiry):
        return np.interp(expiry, self._maturities, self._rates)

    def fwd_rate(self, expiry: float, tenor: float, m: int = None):
        forward_rate = - np.log((self.df(expiry) / self.df(expiry + tenor))) / tenor
        if m is None:
            return forward_rate
        return m * (np.exp(forward_rate / m) - 1)

    def interpolate(self, expiry: list):

        if self._interp_method[1] == 'R':
            return self.result(expiry)
        elif self._interp_method[1] == 'LDF':
            return -1*self.result(expiry) / expiry
        else:
            interp_rates = []
            for date in expiry:
                forward_values = np.concatenate(
                    [
                        self.values[self._interp_method[1]][0][self.values[self._interp_method[1]][1] < date],
                        np.array(self.result(date))
                    ]
                )
                interp_rates.append(forward_values.mean())

            return interp_rates

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
