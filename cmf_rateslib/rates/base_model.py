
from ..curves.zero_curve import ZeroCurve

class BaseRatesModel(object):

    params: dict

    def __init__(self):
        self.params = {}

    def generate_zero_curves(self, **kwargs):
        raise NotImplementedError()

    def evolve_zero_curve(self, start_curve: ZeroCurve, num_periods: int, dt: float):
        raise NotImplementedError()




