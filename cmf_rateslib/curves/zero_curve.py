import numpy as np
import matplotlib.pyplot as plt
from ..curves.base_curve import BaseZeroCurve


class ZeroCurve(BaseZeroCurve):

    def lin_interpolate(self, interpolation: str, tenor=1/12):
        new_maturities = []
        for i in range(1, int(self._maturities[-1]*12)):
            new_maturities.append(i/12)
        new_maturities = np.array(new_maturities)
        new_rates = []
        if interpolation == 'zero_rates':
            for i in range(len(new_maturities)):
                new_rates.append(np.interp(new_maturities[i], self._maturities, self._rates))
            self._maturities = new_maturities
            self._rates = np.array(new_rates)
        elif interpolation == 'log_discount_factors':
            df = self._maturities * self._rates
            new_rates.append(self._rates[0])
            for i in range(1, len(new_maturities)):
                new_rates.append(np.interp(new_maturities[i], self._maturities, df)/new_maturities[i])
            self._maturities = new_maturities
            self._rates = np.array(new_rates)
        elif interpolation == 'forward_rates':
            forward_rates = []
            for i in range(len(new_maturities)):
                forward_rates.append(self.fwd_rate((i+1)/12, tenor))
            new_rates.append(self._rates[0])
            for i in range(1, len(new_maturities)):
                new_rates.append((forward_rates[i-1]+12*new_maturities[i-1]*new_rates[i-1])/(1+12*new_maturities[i-1]))
            self._maturities = new_maturities
            self._rates = np.array(new_rates)
        else:
            raise ValueError('unknown interpolation type')

    def roll(self, t: float):
        if t < 0:
            raise ValueError('invalid roll period value')
        new_maturities = []
        for i in range(1, int(self._maturities[-1] * 12)):
            new_maturities.append(i / 12)
        new_maturities = np.array(new_maturities)
        new_rates = []
        for i in range(len(new_maturities)):
            new_rates.append(self.fwd_rate(t, t+new_maturities[i]))
        self._maturities = new_maturities
        self._rates = np.array(new_rates)

    def plot(self):
        plt.plot(self._maturities, self._rates)
        plt.show()
