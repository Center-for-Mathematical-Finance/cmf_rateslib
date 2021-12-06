from ..rates.base_model import BaseRatesModel
from ..curves.zero_curve import ZeroCurve
import numpy as np
from scipy import optimize

class MeanRevertingPCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols, theta, kappa):
        super().__init__()

        maturities = np.array(maturities)
        maturity_loadings = np.array(maturity_loadings)
        factor_vols = np.array(factor_vols)
        theta = np.array(theta)
        kappa = np.array(kappa)

        if maturities.shape[0] != maturity_loadings.shape[0]:
            raise ValueError("Shapes of maturities and maturity_loandings don't match")

        if factor_vols.shape[0] != maturity_loadings.shape[1]:
            raise ValueError("Shapes of factor_vols and maturity_loandings don't match")
        
        if factor_vols.shape[0] != theta.shape[0]:
            raise ValueError("Shapes of factor_vols and theta don't match")
        
        if factor_vols.shape[0] != kappa.shape[0]:
            raise ValueError("Shapes of factor_vols and kappa don't match")

        self.params['maturities'] = maturities
        self.params['loadings'] = maturity_loadings
        self.params['num_factors'] = maturity_loadings.shape[1]
        self.params['factor_vols'] = factor_vols
        self.params['theta'] = theta
        self.params['kappa'] = kappa

    
    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):

        # =========================================================================
        # Function for computing X0 (factors at starting point)
        # z = Ux + c 
        # ||c|| -> min => x
        def findX0(z,U):

            def remainder_norm(x):
                return np.dot(z-U.dot(x),z-U.dot(x))

            result = optimize.minimize(remainder_norm,np.zeros((1,U.shape[1])) )

            return result['x']
        # =========================================================================

        U = self.params['loadings']
        maturities = self.params['maturities']
        num_factors = self.params['num_factors']
        sigma = self.params['factor_vols']
        theta = self.params['theta']
        kappa = self.params['kappa']

        
        z = start_curve.zero_rate(maturities) # get starting rates
        X0 = findX0(z,U) # find starting factors

        starting_rates = start_curve.zero_rate(maturities)
        dW = np.random.randn(num_periods, num_factors) * np.sqrt(dt)


        # increments of PCA factors
        # dX = kappa(X-theta)dt + sigma*dW
        dX_stochastic = sigma * dW
        dX_deterministic = np.zeros((num_periods+1,num_factors))
        dX_deterministic[0,:] = X0
        Xi = X0

        for i in range(1,num_periods+1):
            Xi += kappa.dot(theta - Xi) * dt
            dX_deterministic[i] = Xi

        dX = dX_deterministic[1:,:] + dX_stochastic

        # increments of zero rates
        dZ = U.dot(dX.T)

        interp_method = '-'.join(start_curve._interp_method) # delete?

        starting_rates = start_curve.zero_rate(maturities)

        all_rates = np.cumsum(np.concatenate([np.array([starting_rates]), dZ.T]), axis = 0)[1:, :]

        return [ZeroCurve(maturities, rates, interp_method) for rates in all_rates]