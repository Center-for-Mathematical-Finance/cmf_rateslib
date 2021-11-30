
from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np
import scipy as sp
import pandas as pd


class SimplePCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols,real_rates:pd.DataFrame=None,):
        super().__init__()

        if maturity_loadings is None or factor_vols is None:
            maturity_loadings = self.create_pca(real_rates)[0]
            factor_vols = self.create_pca(real_rates)[1]
        else:
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

    @staticmethod
    def create_pca(real_rates: pd.DataFrame, number_pca: int = 3):
        real_rates -= real_rates.mean(axis=0)
        covariance_matrix = np.cov(real_rates, rowvar=False)
        eigenvals, eigenvecs = sp.linalg.eigh(covariance_matrix)
        eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]]
        eigenvecs = eigenvecs[:, :number_pca]
        return eigenvecs, np.std(eigenvecs, axis=1)

    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):

        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']

        maturities = self.params['maturities']

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)

        # increments of PCA factors
        dX = sigma * dW

        # increments of zero rates
        dZ = U.dot(dX.T)

        starting_rates = start_curve.zero_rate(maturities)

        all_rates = np.concatenate((starting_rates.reshape(1, starting_rates.shape[0]), dZ.T), axis=0).cumsum(axis=0)[1:, :]

        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def create_new(self, curves_num: int):
       pass
