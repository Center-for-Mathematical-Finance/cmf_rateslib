import datetime
import numpy as np
import scipy
from scipy.interpolate import interp1d


class BaseZeroCurve(object):

    _maturities: np.ndarray
    _rates: np.ndarray

    def __init__(self, maturities, rates, interp_method='L_R', rate_100yr=0.05):
        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must be os the same length")

        self._maturities = np.array(maturities.append(100.0))
        self._rates = np.array(rates.append(rate_100yr))
        self._interp_method = interp_method.split('-')

        self.tenor_to_periods = {'3M': 0.25, '6M': 0.5, '1Y': 1.0}
        self.interp_type = {'L': 'linear', 'Q': 'quadratic'}

    @staticmethod
    def to_discrete(rate: Float, m: Int):
        return m * (np.exp(rate / m) - 1)

    @staticmethod
    def ssq(k):
        return k * (k + 1) * (2 * k + 1) / 6

    def df(self, expiry):
        return np.exp(- self.zero_rate(expiry) * expiry)

    def zero_rate(self, expiry, m: int = None):
        rate = self.interpolate(expiry)
        if m is None:
            return rate
        return self.to_discrete(rate, m)

    def fwd_rate(self, expiry: float, tenor: float, m: int = None):
        rates = self.interpolate([expiry + tenor, expiry])
        forward_rate = ((expiry + tenor) * rates[0] - expiry * rates[1]) / tenor
        if m is None:
            return forward_rate
        return self.to_discrete(forward_rate, m)

    def prepare_forwards(self, tenor=0.25, type='L'):
        forwards = [self._rates[-1]]
        params = {self._maturities: (forwards[0], 0)}
        if type == 'Q':
            derivs = [0]
            params = {self._maturities: (forwards[0], 0, 0)}

        for i in range(len(self._rates) - 1, 0, -1):
            first_rate, second_rate = self_rates[i - 1:i + 1]
            first_mat, second_mat = self._maturities[i - 1:i + 1]
            n = (second_mat - first_mat) / tenor - 1
            if type == 'L':
                a, b = np.linalg.solve(
                    [
                        [
                            n,
                            (first_mat + second_mat - tenor) / 2 * n
                        ],
                        [
                            1,
                            second_mat
                        ]
                    ],
                    [
                        (second_mat * second_rate - first_mat * first_rate) / tenor,
                        forwards[-1]
                    ])
                params[first_mat] = (a, b)
                forwards.append(a + b * first_mat)
            elif type == 'Q':
                a, b, c = np.linalg.solve(
                    [
                        [
                            n,
                            (first_mat + second_mat - tenor) / 2 * n,
                            (ssq(second_rate / tenor - 1) - ssq(first_rate / tenor - 1)) * tenor ** 2],
                        [
                            1,
                            second_mat,
                            second_mat ** 2],
                        [
                            0,
                            1,
                            2 * second_mat]
                    ],
                    [
                        (second_mat * second_rate - first_mat * first_rate) / tenor,
                        forwards[-1],
                        derivs[-1]
                    ])
                params[first_mat] = (a, b, c)
                forwards.append(a + b * first_mat, c * first_mat ** 2)
                derivs.append(b + 2 * c * first_mat)

        forwards = np.array(reversed(forwards))
        interp_func = interp1d(self._maturities, forwards, self.interp_type[self._interp_method[0]])
        return forwards, interp_func, params

    def compute_rates(self, expiry, tenor, params):
        rates = []
        expiry = np.array(expiry)
        ind_base = self._rates[np.searchsorted(self._maturities, expiry - tenor, side='right')]

        for i in range(len(expiry)):
            ind = ind_base[i]
            first_mat, second_mat = self._maturities[ind:], expiry[i]
            first_rate = self._rates[ind]

            n = (second_mat - first_mat) / tenor - 1
            if self._interp_method[0] == 'L':
                a, b = params[first_mat]
                rate = first_rate * first_mat + tenor * (a * n + b * (first_mat + second_mat - tenor) / 2 * n)
                rate /= second_mat
            else:
                a, b, c = params[first_mat]
                rate = first_rate * first_mat + tenor * (a * n + b * (first_mat + second_mat - tenor) / 2 * n)
                rate += (ssq(second_rate / tenor - 1) - ssq(first_rate / tenor - 1)) * tenor ** 3
                rate /= second_mat
            rates.append(rate)

        return np.array(rates)

    def interpolate(self, expiry: List):

        if self._interp_method[1] == 'R':
            interp_func = interp1d(self._maturities, self._rates,
                                   self.interp_type[self._interp_method[0]])
            return interp_func(expiry)
        elif self._interp_method[1] == 'LDF':
            interp_func = interp1d(self._maturities, np.log(self._discounts_cont),
                                   self.interp_type[self._interp_method[0]])
            return - interp_func(expiry) / expiry
        else:
            _, _, params = self.prepare_forwards(self.tenor_to_periods[self._interp_method[2]], self._interp_method[0])
            return self.compute_rates(expiry, self.tenor_to_periods[self._interp_method[2]], params)

    def create_from_existing_curve(self, interp_method):
        return ZeroCurve(self._maturities, self._rates, interp_method)

    def bump(self, shift, exposure, type='parallel', standpoint=None):
        if type == 'parallel' or (type == 'all' and len(shift) == len(self._maturities)):
            return BaseZeroCurve(self._maturities, self._rates + shift)
        elif type == 'linear':
            if standpoint is None:
                standpoint = np.median(self._maturities)
            return BaseZeroCurve(self._maturities, self._rates + shift + exposure * (self._maturities - standpoint))
        else:
            raise ArgumentError('False Parameters!')

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