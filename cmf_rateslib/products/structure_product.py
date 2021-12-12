import typing as tp
from datetime import datetime

import numpy as np

from ..products.base_product import BaseProduct


class StructureProduct(BaseProduct):
    """
        Linear combination of base products.
    """

    def __init__(
        self,
        weights: np.array,
        products: list[BaseProduct],
    ) -> None:
        super().__init__()
        assert len(weights) == len(products)
        self.params = dict(
            weights=weights,
            products=products,
        )

    def __neg__(self) -> 'StructureProduct':
        return StructureProduct(weights=[-w for w in self.params['weights']], products=self.params['products'])

    def __add__(self, other: 'StructureProduct') -> 'StructureProduct':
        # TODO maybe add up the weights of equivalent products instead of duplicating them in products list.
        return StructureProduct(
            weights=self.params['weights'] + other.params['weights'],
            products=self.params['products'] + other.params['products'],
        )

    def __sub__(self, other: 'StructureProduct') -> 'StructureProduct':
        return self + -other

    def __mul__(self, coefficient: float) -> 'StructureProduct':
        assert isinstance(coefficient, float)
        return StructureProduct(
            weights=[coefficient * w for w in self.params['weights']],
            products=self.params['products'],
        )

    def __rmul__(self, coefficient: float) -> 'StructureProduct':
        return self * coefficient

    def get_cashflows(
        self,
        *args,
        **kwargs,
    ) -> tuple[np.array, np.array]:
        cashflows: dict[datetime, float] = {}
        for product_i, product in enumerate(self.params['products']):
            dates, flows = product.get_cashflows(*args, **kwargs)
            for date_i, date in enumerate(dates):
                cashflows[date] = (
                    cashflows.get(date, 0.)
                    + self.params['weights'][product_i] * flows[date_i]
                )
        return np.array(list(cashflows.keys())), np.array(list(cashflows.values()))

    def _weighted_sum(self, func: str, *args, **kwargs) -> float:
        return sum(
            self.params['weights'][product_i] * getattr(product, func)(*args, **kwargs)
            for product_i, product in enumerate(self.params['products'])
        )

    def dv01(self, *args, **kwargs) -> float:
        return self._weighted_sum('dv01')

    def modified_duration(self, *args, **kwargs) -> float:
        return self._weighted_sum('modified_duration')

    def convexity(self, *args, **kwargs):
        return self._weighted_sum('convexity')

    def rolldown(self, *args, **kwargs):
        return self._weighted_sum('rolldown')
