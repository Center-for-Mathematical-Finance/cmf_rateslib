import numpy as np
import matplotlib.pyplot as plt
from ..curves.base_curve import BaseZeroCurve


class ZeroCurve(BaseZeroCurve):

    def __init__(self, maturities, rates, interp_method):
        super().__init__(maturities, rates, interp_method)

    def create_from_existing_curve(self, interp_method):
        return ZeroCurve(self._maturities, self._rates, interp_method)

    def roll(self, t: float):
        if t < 0:
            raise ValueError('invalid roll period value')
        # new_maturities = []
        # for i in range(1, int(self._maturities[-1] * 12)):
        #    new_maturities.append(i / 12)
        # new_maturities = np.array(new_maturities)
        new_rates = []
        for i in range(len(self._maturities)):
            new_rates.append(-1 * self.fwd_rate(t + self._maturities[i], t))
        # self._maturities = new_maturities
        new_rates = np.array(new_rates)

        return ZeroCurve(self._maturities, new_rates, "-".join(str(x) for x in self._interp_method))

    def plot(self, tenors=None):
        if tenors is None:
            tenors = list(np.linspace(1 / 252, self._maturities[-1]))
        plt.plot(tenors, self.interpolate(tenors))
        plt.scatter(list(self._maturities), list(self._rates), marker='o')
        plt.show()
