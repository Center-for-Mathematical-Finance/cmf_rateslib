
from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np
import typing


class MeanRevertingPCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols, theta, alpha):
        super().__init__()

        maturities = np.array(maturities)
        maturity_loadings = np.array(maturity_loadings)
        factor_vols = np.array(factor_vols)

        if maturities.shape[0] != maturity_loadings.shape[0]:
            raise ValueError("Shapes of maturities and maturity_loandings don't match")

        if factor_vols.shape[0] != maturity_loadings.shape[1]:
            raise ValueError("Shapes of factor_vols and maturity_loandings don't match")

        if factor_vols.shape[0] != theta.shape[0]:
            raise ValueError("Shapes of factor_vols and theta don't match")

        if factor_vols.shape[0] != alpha.shape[0]:
            raise ValueError("Shapes of factor_vols and alpha don't match")

        self.params['maturities'] = maturities
        self.params['loadings'] = maturity_loadings
        self.params['num_factors'] = maturity_loadings.shape[1]
        self.params['factor_vols'] = factor_vols
        self.params['alpha'] = alpha
        self.params['theta'] = theta

    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):
        """ 
            Evolve given zero curve using loadings and mean-reverting PCA method.
        """ 
        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        maturities = self.params['maturities']
        theta = self.params['theta']
        alpha = self.params['alpha']

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)
        
        starting_rates = start_curve.zero_rate(maturities)

        # stochastic loop
        r_prev = starting_rates[0]
        r_list = []
        for i in dW:
            r_cur = alpha * (theta - r_prev) * dt + sigma * i
            r_list.append(r_cur)
            r_prev += r_cur

        # increments of PCA factors
        dX = np.array(r_list)

        # increments of zero rates
        dZ = U.dot(dX.T)

        all_rates = np.cumsum(np.concatenate([[starting_rates], dZ.T]), axis = 0)[1:, :]

        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def create_new(self, n_curves:int, dt:float, r_0:float=None):
        """ 
            Generate n zero curves using loadings and mean-reverting PCA method.
        """      
        U = self.params['loadings']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        maturities = self.params['maturities']
        theta = self.params['theta']
        alpha = self.params['alpha']

        dW = np.random.randn(n_curves, num_factors) * np.sqrt(dt)
        
        # stochastic loop
        if r_0 is None:
            r_prev = 0
        else:
            r_prev = r_0
        r_list = []
        for i in dW:
            r_cur = alpha * (theta - r_prev) * dt + sigma * i
            r_list.append(r_cur)
            r_prev += r_cur

        # increments of PCA factors
        dX = np.array(r_list)

        # increments of zero rates
        dZ = U.dot(dX.T)

        all_rates = dZ.T

        return [ZeroCurve(maturities, rates) for rates in all_rates]

    def fit(self, curve_list:list, n_components:int=0):
        """ 
            For given list of curves fit U and vol
        """ 
        rate_matrix = np.array([curve._rates for curve in curve_list])
        cov = np.cov(rate_matrix)
        vol, u = np.linalg.eig(cov)
        if (n_components == 0):
            n_components = self.params["num_factors"]
        ind = np.arsort(vol)[::-1]
        vols = vols[ind][:n_components]
        u = u[:, ind][:, n_components]

        return u, np.sqrt(vol) 




