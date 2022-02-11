from ..curves.base_curve import BaseZeroCurve
import numpy as np


class ZeroCurve(BaseZeroCurve):

    def __init__(self, maturities, rates, interp_method):
        super().__init__(maturities, rates, interp_method)
