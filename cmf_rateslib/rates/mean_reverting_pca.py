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

    def evolve_zero_curve(self, start_curve: ZeroCurve, curves_num: int, dt: float):

        maturities = self.params['maturities']
        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta = self.params['theta']

        starting_rates = start_curve.zero_rate(maturities)

        X = starting_rates[0]
        dW = np.random.randn(curves_num, num_factors) * np.sqrt(dt)
        dX = np.array([(np.random.randn(num_factors) * theta - X) * dt + sigma * w for w in dW])
        # из simple_pca.py
        dZ = U.dot(dX.T)
        starting_rates = start_curve.zero_rate(maturities)
        all_rates = np.concatenate(starting_rates, dZ.T).cumsum()[1:, :]
        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def create_new(self, curves_num: int, dt: float):

        maturities = self.params['maturities']
        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta = self.params['theta']

        dW = np.random.randn(curves_num, num_factors) * np.sqrt(dt)
        dX = np.array([(np.random.randn(num_factors) * theta) * dt + sigma * w for w in dW])

        # из simple_pca.py
        dZ = U.dot(dX.T)
        all_rates = dZ.T
        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def fit(self, fitting_curves, num_factor: int = -1):
        rates = np.array([curve._rates for curve in fitting_curves])
        cov_matrix = np.cov(rates)

        eig_val, eig_vec = np.linalg.eig(cov_matrix)[:num_factor]
        # наверно стоит возвращать self.params['loadings'] иначе не понятно,
        # почему это метод класса SimplePCAModel, а не BaseModel
        return eig_val, eig_vec, self.params['loadings']
