
from ..curves.base_curve import BaseZeroCurve
import numpy as np
from scipy.linalg import sqrtm

class ZeroCurve(BaseZeroCurve):

    _interpolation_mode: str

    _compounding_mode: str
    _annual_number_of_compounds: float

    def __init__(self, maturities, rates, interpolation_mode, compounding_mode, annual_number_of_compounds):
        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must be os the same length")
        if not (interpolation_mode in ['zero_rate', 'log_df', 'forward_rate_3month',
                         'forward_rate_6month', 'forward_rate_12month']) or not (compounding_mode in ['discrete', 'continuous']):
            raise ValueError("Incorrect interpolation or compounding mode")

        self._compounding_mode = compounding_mode
        self._annual_number_of_compounds = annual_number_of_compounds

        maturities = np.array(maturities)  # for maturities to be sorted ascending
        sorted_args = maturities.argsort()
        self._maturities = maturities[sorted_args]
        self._rates = np.array(rates)[sorted_args]
        self._interpolation_mode = interpolation_mode

    def from_existing_curve(self,curve,interpolation_mode):
        if not (interpolation_mode in ['zero_rate', 'log_df', 'forward_rate_3month',
                         'forward_rate_6month', 'forward_rate_12month']):
            raise ValueError("Incorrect interpolation mode")
        return ZeroCurve(self._maturities,self.rates,interpolation_mode,self._compounding_mode,self._annual_number_of_compounds)


    def df(self, expiry,pow=1):
        if(self._compounding_mode=='continuous'):
            return np.exp(- self.interpolate(expiry,pow) * expiry)
        else:
            return (1+self.zero_rate(expiry)/self._annual_number_of_compounds)**(-np.floor(self._annual_number_of_compounds*expiry))

    def fwd_rate(self, expiry,tenor,pow=1):

        if (self._compounding_mode == 'continuous'):
            return np.log((self.df(expiry,pow)/self.df(expiry + tenor,pow))) / tenor
        else:
            return (self.df(expiry)/self.df(expiry+tenor))**tenor - 1

    def roll(self,t):
        new_rates = np.array([])
        for maturity in self._maturities:
            new_rates = np.append(new_rates,self.fwd_rate(t, maturity))
        return ZeroCurve(self._maturities,new_rates,self._interpolation_mode, self._compounding_mode, self._annual_number_of_compounds)

    def bump_pca(self,historical_curves,bumped_values,bumped_positions):
        cov_mtx = np.cov(np.concatenate([historical_curves,self._rates[:,np.newaxis]],axis=1))
        L, U = np.linalg.eig(cov_mtx)
        L = np.diag(L)
        V = U @ sqrtm(L)

        p_num = cov_mtx.shape[0]
        I = -2 * np.eye(p_num)
        unbumped_positions = np.setdiff1d(np.arange(cov_mtx.shape[0]),bumped_positions)
        V_ = np.delete(V, unbumped_positions, axis=0)
        O = np.zeros((len(bumped_positions),len(bumped_positions)))
        A = np.concatenate([np.concatenate([I, V_]), np.concatenate([V_.T, O])], axis=1)

        b = np.zeros(p_num)
        b = np.concatenate([b, self._rates[bumped_positions]+bumped_values])
        x = np.linalg.solve(A, b)

        p = x[:5]
        y_ = V @ p

        return y_


    def _get_spline_coeffs(self,maturities,rates):
        beta = (rates[1] - rates[0]) / (maturities[1] - maturities[0])
        a = []
        b = []
        c = []
        for i in range(len(maturities)-1):
            c.append((rates[i + 1] - rates[i]) / (maturities[i + 1] - maturities[i]) ** 2 - beta / (maturities[i + 1] - maturities[i]))
            b.append(beta - 2 * c[i] * maturities[i])
            a.append(rates[i] - c[i] * maturities[i] ** 2 - b[i] * maturities[i])
            beta = b[i] + 2 * c[i] * maturities[i+1]
        return a,b,c

    def interpolate(self, expiries, pow =1):
        expiries=np.array(expiries)
        if not(pow in [1,2]):
            raise ValueError("Incorrect power")
        if np.any((expiries<self._maturities[0])*(expiries>self._maturities[-1])):
            raise ValueError("Expiry is not in maturities range. Use extrapolation")

        if pow==1:
            if(self._interpolation_mode=='zero_rate'):
                return self.zero_rate(expiries)
            if(self._interpolation_mode=='log_df'):
                return np.interp(expiries, self._maturities, -self._rates*self._maturities)/(-expiries)
            if(self._interpolation_mode=='forward_rate_3month'):
                pass

        if pow==2:
            out_rates = []
            n=np.zeros(len(expiries))

            if (self._interpolation_mode == 'zero_rate'):
                a,b,c = self._get_spline_coeffs(self._maturities,self._rates)

                for i in range(len(expiries)):
                    n=0
                    while (self._maturities[n] < expiries[i]):
                        n = n + 1
                    out_rates.append(a[n-1]+b[n-1]*expiries[i] + c[n-1]*expiries[i]**2)
                return np.array(out_rates)

            if (self._interpolation_mode == 'log_df'):
                a, b, c = self._get_spline_coeffs(self._maturities, -self._rates*self._maturities)

                for i in range(len(expiries)):
                    n=0
                    while (self._maturities[n] < expiries[i]):
                        n = n + 1
                    out_rates.append(((a[n - 1] + b[n - 1] * expiries[i] + c[n - 1] * expiries[i] ** 2))/(-expiries[i]))

                return np.array(out_rates)

    def extrapolate(self, expiries, pow =1):

        if not(pow in [1,2]):
            raise ValueError("Incorrect power")
        if np.any((expiries>=self._maturities[0])*(expiries<=self._maturities[-1])):
            raise ValueError("Expiry is in maturities range. Use interpolation")

        out_rates=[]

        if pow==1:
            if(self._interpolation_mode=='zero_rate'):
                for e in expiries:
                    if(e<self._maturities[0]):
                        out_rates.append(self._rates[0] + (self._rates[0] - self._rates[1]) /
                                            (self._maturities[0] - self._maturities[1]) * (e - self._maturities[0]))
                    else:
                        out_rates.append(self._rates[-1] + (self._rates[-1] - self._rates[-2]) /
                                     (self._maturities[-1] - self._maturities[-2]) * (e - self._maturities[-1]))
                return np.array(out_rates)

            if(self._interpolation_mode=='log_df'):
                for e in expiries:
                    if (e < self._maturities[0]):
                        out_rates.append(np.exp((self._rates[0] + (self._rates[0] - self._rates[1]) /
                                         (self._maturities[0] - self._maturities[1]) * (e - self._maturities[0]))))
                    else:
                        out_rates.append(np.exp((self._rates[-1] + (self._rates[-1] - self._rates[-2]) /
                                         (self._maturities[-1] - self._maturities[-2]) * (e - self._maturities[-1]))))
                return np.array(out_rates)

        if pow == 2:
            out_rates = []
            if (self._interpolation_mode == 'zero_rate'):
                a, b, c = self._get_spline_coeffs(self._maturities, self._rates)

                for i in range(len(expiries)):
                    if(expiries[i]<self._maturities[0]):
                        out_rates.append(a[0] + b[0] * expiries[i] + c[0] * expiries[i] ** 2)
                    else:
                        out_rates.append(a[-1] + b[-1] * expiries[i] + c[-1] * expiries[i] ** 2)
                return np.array(out_rates)

            if (self._interpolation_mode == 'log_df'):
                a, b, c = self._get_spline_coeffs(self._maturities, np.log(self._rates))

                for i in range(len(expiries)):
                    if(expiries[i]<self._maturities[0]):
                        out_rates.append(np.exp((a[0] + b[0] * expiries[i] + c[0] * expiries[i] ** 2)))
                    else:
                        out_rates.append(np.exp((a[-1] + b[-1] * expiries[i] + c[-1] * expiries[i] ** 2)))

                return np.array(out_rates)











