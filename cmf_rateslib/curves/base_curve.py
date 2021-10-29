import datetime


class BaseCurve(object):

    ccy: str
    curve_type: str
    interp_method: str
    asof: datetime.datetime

    def df(self, expiry: float) -> float:
        raise NotImplementedError()

    def zero_rate(self, expiry: float, freq=None):
        raise NotImplementedError()

    def fwd_rate(self, expiry: float, tenor: float, freq=None):
        raise NotImplementedError()

    # TODO:
    def bump(self):
        raise NotImplementedError()