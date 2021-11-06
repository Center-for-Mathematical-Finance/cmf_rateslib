
from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np


class SimplePCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols):
        super().__init__()

        maturities = np.array(maturities)
        maturity_loadings = np.array(maturity_loadings)
        factor_vols = np.array(factor_vols)

        if maturities.shape[0] != maturity_loadings.shape[0]:
            raise ValueError("Shapes of maturities and maturity_loandings don't match")

        if factor_vols.shape[0] != maturity_loadings.shape[1]:
            raise ValueError("Shapes of factor_vols and maturity_loandings don't match")

        self.params['maturities'] = maturities
        self.params['loadings'] = maturity_loadings
        self.params['num_factors'] = maturity_loadings.shape[1]
        self.params['factor_vols'] = factor_vols

    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):

        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']

        maturities = self.params['maturities']

        dW = np.random.randn(num_periods, num_factors) * dt

        # increments of PCA factors
        dX = sigma * dW

        # increments of zero rates
        dZ = U.dot(dX)

        starting_rates = start_curve.zero_rate(maturities)

        all_rates = np.concatenate(starting_rates, dZ.T).cumsum()[1:, :]

        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def myfunction(self):
        pass




