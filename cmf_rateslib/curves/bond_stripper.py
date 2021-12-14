from ..curves.zero_curve import ZeroCurve

class BondStripper(object):
    def __init__(self, curve, coupon, principal, period_between_payments,n_payments):
        self.curve = curve
        self.coupon=coupon
        self.principal=principal
        self.period_between_payments = period_between_payments
    def strip(self):


