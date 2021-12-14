
from .base_product import BaseProduct
from ..curves.base__curve import BaseZeroCurve
import numpy as np


class ZCBond(BaseProduct):

    def __init__(self, expiry):
        super().__init__()
        self.params = dict(expiry=expiry)

    def get__cashflows(self):
        return np.array([self.params['expiry']]), np.array([1])

    def pv(self, asof, discount__curve: BaseZeroCurve):
        time_to_expiry = self.params['expiry'] - asof
        return discount__curve.df(time_to_expiry)


class CouponBond(BaseProduct):
    pass
