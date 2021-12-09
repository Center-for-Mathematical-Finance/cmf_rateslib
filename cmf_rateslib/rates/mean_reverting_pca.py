
from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np


class MeanRevertingPCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols, theta):
        super().__init__()

        maturities = np.array(maturities)
        maturity_loadings = np.array(maturity_loadings)
        factor_vols = np.array(factor_vols)
        theta = np.array(theta)

        if maturities.shape[0] != maturity_loadings.shape[0]:
            raise ValueError("Shapes of maturities and maturity_loadings don't match")

        if factor_vols.shape[0] != maturity_loadings.shape[1]:
            raise ValueError("Shapes of factor_vols and maturity_loadings don't match")

        self.params['maturities'] = maturities
        self.params['loadings'] = maturity_loadings
        self.params['num_factors'] = maturity_loadings.shape[1]
        self.params['factor_vols'] = factor_vols
        self.params['theta'] = theta

    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):

        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta = self.params['theta']


        maturities = self.params['maturities']

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)
        dX = []

        starting_rates = start_curve.zero_rate(maturities)

        X = starting_rates[0]

        for d in dW:
            dX.append((np.random.randn(num_factors) * theta - X) * dt + sigma * d)
            X += dX[-1]

        dX = np.array(dX)

        # increments of zero rates
        dZ = U.dot(dX.T)

        all_rates = np.cumsum(np.concatenate([[starting_rates], dZ.T]), axis = 0)[1:, :]

        return [ZeroCurve(maturities, rates) for rates in all_rates]


    def create_new(self, num_periods: int, dt: float):

        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta = self.params['theta']


        maturities = self.params['maturities']

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)
        dX = []

        X = 0

        for d in dW:
            dX.append((np.random.randn(num_factors) * theta - X) * dt + sigma * d)
            X += dX[-1]

        dX = np.array(dX)

        # increments of zero rates
        dZ = U.dot(dX.T)

        return [ZeroCurve(maturities, rates) for rates in dZ.T]

    def create_new_1(self, dt: float):
        sigma = self.params['factor_vols']
        theta = self.params['theta']
        maturities = self.params['maturities']
        alpha = 4
        curves = []
        for m in maturities:
            for t in theta:
                for s in sigma:
                    u = np.random.randn(m * 253)
                    dW = u * np.sqrt(dt)
                    r_0 = t
                    r_i = r_0
                    r_t_mc = np.zeros(m * 253)
                    r_t_mc[0] = r_0
                    for i in range(1, m * 253):
                        r_i += alpha * (t - r_i) * dt + s * dW[i]
                        r_t_mc[i] = r_i
                    curves.append(r_t_mc)

        return curves





    def fit(self, curves):
        # dZ = UdX. Кривые задаются матрицей Z.
        # в силу спектрального разложения матрицы  - мы можем раложить матрицу ковариации dZ с dZ в прозведение матрицы
        # из лоадингов и диагональной матрицы  с собственными значениями, которые будут являться нашей волатильностью.
        # Следовательно, необоходимо вывести матрицу лоадингов, собственные значения и собственные вектора
        U = self.params['loadings']
        curves_rates = []
        for curve in curves:
            curves_rates.append(curve._rates)
        curves_rates = np.array(curves_rates)
        Cov = np.cov(curves_rates)
        eigvalues, eigvectors = np.linalg.eig(Cov)
        return eigvalues, eigvectors, U





