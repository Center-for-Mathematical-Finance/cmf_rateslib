import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def sq_interpolate(x: list, y: list, z: float):
    if x != sorted(x):
        raise ValueError('x must be sorted ascending')
    for i in range(len(x) - 1):
        if (int(x[i - 1]) <= int(z) <= int(x[i + 1])):
            a2 = (y[i + 1] - y[i - 1]) / ((x[i + 1] - x[i - 1]) * (x[i + 1] - x[i])) - (y[i] - y[i - 1]) / (
                        (x[i] - x[i - 1]) * (x[i + 1] - x[i]))
            a1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1]) - a2 * (x[i] + x[i - 1])
            a0 = y[i - 1] - a1 * x[i - 1] - a2 * (x[i - 1] ** 2)
            interpolate = a0 + a1 * z + a2 * (z ** 2)

    return interpolate

# проверим на точках с https://www.moex.com/ru/marketdata/indices/state/g-curve/

maturities = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 10]
rates = [0.0751, 0.0763, 0.0773, 0.079, 0.0804, 0.0819, 0.0838, 0.0838, 0.0825, 0.0808]
dates = [0.25, 0.4, 0.7, 1.7, 2.5, 4, 6, 7, 8, 9]

int_rates = []
for date in dates:
    int_rates.append(sq_interpolate(maturities, rates, date))

print(int_rates)

# сравним со встроенной интерполяцией
f = interpolate.interp1d(maturities, rates, kind='quadratic')
pyth_int_rates = f(dates)

fig = plt.figure(figsize=(12, 6))

plt.plot(maturities, rates)
plt.plot(dates, int_rates, color='green')
plt.plot(dates, pyth_int_rates, color='red')
plt.title("Кривая бескупонной доходности", weight="bold")
plt.legend(['Кривая с moex', 'Интерполированная кривая', 'Кривая интеполированная встроенной функцией'], fontsize=10)
plt.grid(True)
plt.show()