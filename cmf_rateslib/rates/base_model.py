

class BaseRatesModel(object):

    params: dict

    def __init__(self):
        self.params = {}

    def generate_zero_curves(self, **kwargs):
        raise NotImplementedError()



