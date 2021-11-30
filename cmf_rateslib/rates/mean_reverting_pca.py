from ..rates.base_model import BaseRatesModel
import numpy as np


class MeanRevetignPCAModel(BaseRatesModel):

    def __init__(self, maturities, maturity_loadings, factor_vols, theta):
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
        self.params['theta'] = theta

    @staticmethod
    def calculate_next_rate(X_last, kappa, theta, sigma, dt):
        X_next = X_last * np.exp(-1 * kappa * dt) + theta * (1 - np.exp(-1 * kappa * dt)) \
        + (np.sqrt((sigma ** 2) * (1 - np.exp(-1 * kappa * dt) ** 2) / (2 * kappa))) * np.random.normal()
        return X_next

    def evolve_zero_curve(self, X_0, num_periods: int, dt: float, kappa, sigma, num_curves: int):
        simulation_curves = np.ndarray((num_periods, num_curves))
        for i in range(0, num_curves):
            simulation_curves[:, i] = self.create_new(X_0=X_0, num_periods=num_periods, kappa=kappa,
                                                      sigma=sigma, dt=dt)
        return simulation_curves

    def create_new(self, X_0, num_periods, kappa, sigma, dt):
        X = np.zeros(num_periods)
        X[0] = X_0
        for i in range(1, num_periods):
            X[i] = self.calculate_next_rate(X[i - 1], kappa=kappa, theta=self.params['theta'], sigma=sigma, dt=dt)
        return X

    def fit_parameters(self, fitting_curves, dt):
        n = len(fitting_curves)

        # maximum likelihood + OLS method
        Sx = sum(fitting_curves[0:(n-1)])
        Sy = sum(fitting_curves[1:n])
        Sxx = np.dot(fitting_curves[0:(n - 1)], fitting_curves[0:(n - 1)])
        Sxy = np.dot(fitting_curves[0:(n - 1)], fitting_curves[1:n])
        Syy = np.dot(fitting_curves[1:n], fitting_curves[1:n])

        theta = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx ** 2 - Sx * Sy))
        kappa = -np.log((Sxy - theta * Sx - theta * Sy + n * theta ** 2) / (Sxx - 2 * theta * Sx + n * theta ** 2)) / dt
        a = np.exp(-kappa * dt)
        sigmah2 = (Syy - 2 * a * Sxy + a ** 2 * Sxx - 2 * theta * (1 - a) * (Sy - a * Sx) + n * theta ** 2 * (
                    1 - a) ** 2) / n
        sigma = np.sqrt(sigmah2 * 2 * kappa / (1 - a ** 2))
        X0 = fitting_curves[n - 1]

        return [kappa, theta, sigma, X0]
