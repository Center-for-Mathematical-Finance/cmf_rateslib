
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

        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)

        # increments of PCA factors
        dX = sigma * dW

        # increments of zero rates
        dZ = U.dot(dX.T)

        interp_method = '-'.join(start_curve._interp_method) # delete?

        starting_rates = start_curve.zero_rate(maturities)

        all_rates = np.cumsum(np.concatenate([np.array([starting_rates]), dZ.T]), axis = 0)[1:, :]

        return [ZeroCurve(maturities, rates, interp_method) for rates in all_rates]


    def fit(self, existing_curves: list, new_maturities: list):
        """
        Function for fitting Simple PCA Model

        Args
        ----
        existing_curves: list, required
            A list of BaseZeroCurve objects
        new_maturities: list required
            A list of new maturities which will be used to compose Z matrix of zero-rates

        Returns
        -------
        U: np.array
            Loadings matrix of size (len(new_maturities),self.params['num_factors']) 
        vols: np.array
            Volatility factors, an array of size (,self.params['num_factors'])

        """
        
        Z = np.array([curve.zero_rate(np.array(new_maturities)) for curve in existing_curves]) 
        # get zero rates at new maturities for each curve
        # rows - existing_curves
        # columns - new_maturities
        
        dZ = np.diff(Z, n=1, axis=0) # get zero rates increments
        C = np.dot(dZ.T,dZ) # compute covariance matrix

        eigenvalues, eigenvectors = np.linalg.eig(C) # get eigenvalues and eigenvectors
        largest_eigenvalues_indices = np.argsort(-eigenvalues) # get the indices of eigenvalues from largest to smallest
        vols = np.sqrt(eigenvalues[largest_eigenvalues_indices][:self.params['num_factors']]) # take top-n largest eigenvalues
        U = eigenvectors[:,largest_eigenvalues_indices][:,:self.params['num_factors']] # take their eigenvectors

        return U, vols



    def create_new(self):
        pass




