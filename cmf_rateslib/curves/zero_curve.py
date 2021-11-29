from cmf_rateslib.curves.base_curve import BaseZeroCurve
import numpy as np


def quad_interp(expiry, maturities, rates):
    x = maturities
    y = rates
    if x != sorted(x):
        raise ValueError('x must be sorted ascending')
    for i in range(len(x) - 1):
        if x[i - 1] <= expiry <= x[i + 1]:
            a2 = (y[i + 1] - y[i - 1]) / ((x[i + 1] - x[i - 1]) * (x[i + 1] - x[i])) - (y[i] - y[i - 1]) / (
                    (x[i] - x[i - 1]) * (x[i + 1] - x[i]))
            a1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]) - a2 * (x[i] + x[i - 1])
            a0 = y[i - 1] - a1 * x[i - 1] - a2 * (x[i - 1] ** 2)
            interpolate = a0 + a1 * expiry + a2 * (expiry ** 2)

            return interpolate


class ZeroCurve(BaseZeroCurve):

    def __init__(self, maturities, rates):
        super().__init__(maturities, rates)

    def interpolation(self, expiry, interp_type, tenor=None):
        interp_types = (['lin_zero_rate', 'lin_log_df', 'lin_fwd_rate', 'quad_zero_rate',
                         'quad_log_df', 'quad_fwd_rate'])
        if interp_type not in interp_types:
            raise ValueError('Unknown interpolation type')
        if interp_type == 'lin_zero_rate':
            return self.zero_rate(expiry)
        elif interp_type == 'lin_log_df':
            log_df = - self._maturities * self._rates
            log_df_interp = np.interp(expiry, self._maturities, log_df)
            return log_df_interp / expiry
        # elif interp_type == 'lin_fwd_rate':
        #     forward_rate = self.fwd_rate(expiry, tenor)
        #     forward_rate_interp = np.interp(expiry, self._maturities, forward_rate)
        #     return
        elif interp_type == 'quad_zero_rate':
            return quad_interp(expiry, self._maturities, self._rates)
        elif interp_type == 'quad_log_df':
            log_df = - self._maturities * self._rates
            log_df_interp = quad_interp(expiry, self._maturities, log_df)
            return log_df_interp / expiry
        # elif type == 'quad_fwd_rate':
        #     return (-np.log((np.exp(- quad_interp(expiry, self._maturities, self._rates) * expiry)
        #                      / np.exp(- quad_interp(expiry + tenor, self._maturities, self._rates) * (expiry + tenor)))) / tenor)
