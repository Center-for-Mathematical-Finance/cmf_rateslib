from typing import List
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import interp
from base_curve import BaseZeroCurve
from interpolator import Interpolator
import numpy as np


class ZeroCurve(BaseZeroCurve):
    __interpolator: Interpolator
    __interp_value: str
    __compound: str
    __interp_method: str

    __df: np.ndarray

    def __init__(self, maturities, rates, interp_method="linear", compound="continuous",\
    rates_format="float"):
        if (rates_format == "percent"):
            rates /= 100

        if (compound == "continuous") or (compound == "discret"):
            self.__compound = compound
        else:
            raise ValueError("invalid compound format!")
        self.__interp_method = interp_method
       
        self.__df = self.__zero_rate_to_df(rates, maturities, compound)

        if (interp_method == "linear") or (interp_method == "quadratic"):
            self.__interp_value = "zero_rates"
            self.__interpolator = Interpolator(maturities, rates, interp_method)
        elif (interp_method == "log_linear") or (interp_method == "log_quadratic"):
            self.__interp_value = "df"
            self.__interpolator = Interpolator(maturities, self.__df, interp_method)
        else:
            raise ValueError("Incorrect interpolated value!")
        
        super().__init__(maturities, rates)

    def __add__(self, other):
        if (isinstance(other, ZeroCurve) or isinstance(other, BaseZeroCurve)):
            return ZeroCurve(self._maturities, self._rates + other.zero_rate(self._maturities),\
                             interp_method=self.__interp_method, compound=self.__compound)
        else:
            raise ValueError("other must be an instance of ZeroCurve or BaseZeroCurve")

    def __zero_rate_to_df(self, zero_rate, expiry, compound):
        if (compound == "continuous"):
            return np.exp(-zero_rate * expiry)
        elif (compound == "discret"):
            return (1 + zero_rate) ** (-expiry)
        else:
            raise ValueError("Invalid compound format!")

    def __df_to_zero_rate(self, df, expiry, compound):
        if (compound == "continuous"):
            return -np.log(df) / expiry
        elif (compound == "discret"):
            return df**(-1/expiry) - 1
        else:
            raise ValueError("Invalid compound format!")
    
    def zero_rate(self, expiry):
        ret = self.__interpolator.interp(expiry)
        if (self.__interp_value == "zero_rates"):
            return ret
        elif (self.__interp_value == "df"):
            return self.__df_to_zero_rate(ret, expiry, self.__compound)
    
    def df(self, expiry):
        ret = self.__interpolator.interp(expiry)
        if (self.__interp_value == "zero_rates"):
            return self.__zero_rate_to_df(ret, expiry, self.__compound)
        elif (self.__interp_value == "df"):
            return ret

    def fwd_rate(self, expiry: np.ndarray, tenor: np.ndarray):
        df1 = self.df(expiry)
        df2 = self.df(expiry + tenor)
        if (self.__compound == "continuous"): 
            return (np.log(df1) - np.log(df2)) / tenor
        elif (self.__compound == "discret"):
            return (df1 / df2) ** (1 / tenor) - 1
        else:
            raise ValueError("Invalid compound format!")

    def roll(self, t):
        return ZeroCurve(self._maturities + t, self._rates,\
             interp_method=self.__interpolator.get_method(), compound=self.__compound)

    def get_rates(self):
        return self._rates.copy()

    @staticmethod
    def get_cov_matrix(curves, maturities):
        mat = np.array([curve.zero_rate(maturities) for curve in curves])
        res = np.cov(mat)
        return res
    
    

    # adj - value at which zero eigenvalues will be corrected
    #dr - dictionary:
    #dr[i] = dr_i - change of i-th rate, 0 <= i <= n - 1
    def PCAbump(self, dr: dict, cov: np.ndarray, adj=0.):
        lmb, U = np.linalg.eigh(cov)
        lmb[lmb < 0] = 0
        z_ind = np.argwhere(lmb == 0).reshape(-1)
        to_change = set(dr.keys())

        doubt_ind = to_change & set(z_ind)
        if doubt_ind != 0:
            if (adj == 0):
                raise ValueError("Indexes {} could not be changed".format(doubt_ind))
            else:
                for i in doubt_ind:
                    lmb[i] = adj / len(doubt_ind)

        sqrtLmb = np.diag(np.sqrt(lmb))
        V = U @ sqrtLmb

        b = np.array(list(dr.values()))
        A = np.array([V[i, :] for i in dr.keys()])
        dy = V @ A.T @ np.linalg.inv(A @ A.T) @ b

        return self + BaseZeroCurve(self._maturities, dy)

    
    @staticmethod
    def changeInterp(zc, interp_method):
        return ZeroCurve(zc._maturities, zc._rates,\
             interp_method, zc.__compound)






