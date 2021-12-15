
from .base_product import BaseProduct
from ..curves.base_curve import BaseZeroCurve
import numpy as np


class ZCBond(BaseProduct):

    def __init__(self, expiry):
        super().__init__()
        self.params = dict(expiry=expiry)

    def get_cashflows(self):
        return np.array([self.params['expiry']]), np.array([1])

    def pv(self, asof, discount_curve: BaseZeroCurve):
        time_to_expiry = self.params['expiry'] - asof
        return discount_curve.df(time_to_expiry)


class CouponBond(BaseProduct):

    _expiry: float
    _coupon: float
    _freq: float

    def __init__(self, expiry: float, coupon: float, freq: int):
        super().__init__()
        self._expiry = expiry
        self._coupon = coupon
        self._freq = freq
