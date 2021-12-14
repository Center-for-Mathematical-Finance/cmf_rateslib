
from ..curves.zero__curve import ZeroCurve

class BaseRatesModel(object):

    params: dict

    def __init__(self):
        self.params = {}

    def generate_zero__curves(self, **kwargs):
        raise NotImplementedError()

    def evolve_zero__curve(self, start__curve: ZeroCurve, num_periods: int, dt: float):
        raise NotImplementedError()




