

class BaseProduct(object):

    params: dict = {}

    def __init__(self):
        pass

    def get__cashflows(self, *args, **kwargs):
        return None

    def pv(self, *args, **kwargs):
        return 0
