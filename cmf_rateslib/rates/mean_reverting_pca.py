
from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np


class MeanRevertingPCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols, theta_vol):
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
        self.params['theta_vol'] = theta_vol

    def evolve_zero_curve(self, X, start_curve: ZeroCurve, num_periods: int, dt: float):

        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta_vol = self.params['theta_vol']


        maturities = self.params['maturities']

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt) # 5x3
        dX = []

        for delta in dW:
            dX.append((np.random.randn(num_factors) * theta_vol - X) * dt + sigma * delta)
            X += dX[-1]

        dX = np.array(dX)

        # increments of zero rates
        dZ = U.dot(dX.T) #4x3 3x5 -> 4x5

        starting_rates = start_curve.zero_rate(maturities)

        all_rates = np.cumsum(np.concatenate([[starting_rates], dZ.T]), axis = 0)[1:, :] #5x4

        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def create_new(self, curves_num: int):
        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        maturities = self.params['maturities']
        W = np.random.randn(curves_num, num_factors) # 5x3
        Z = U.dot(W.T) #4x3 3x5 -> 4x5
        return [ZeroCurve(maturities, rates) for rates in Z.T]

    def fit(self, fitting_curves):

        U = self.params['loadings']
        curves_rates = []
        for curve in fitting_curves:
            curves_rates.append(curve._rates)
        curves_rates = np.array(curves_rates)
        C = np.cov(curves_rates)
        eigvalues, eigvectors = np.linalg.eig(C)
        return eigvalues 








