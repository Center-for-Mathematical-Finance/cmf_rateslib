from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from ..curves.base_curve import BaseZeroCurve

from .structure_product import StructureProduct


class BaseProduct(ABC):

    params: dict = {}

    def __init__(self):
        pass

    def __pos__(self) -> 'BaseProduct':
        return self

    def __neg__(self) -> StructureProduct:
        return StructureProduct(weights=[-1], products=[self])

    def __add__(self, other: 'BaseProduct') -> StructureProduct:
        return StructureProduct(weights=[1], products=[self]) + other

    def __sub__(self, other: 'BaseProduct') -> StructureProduct:
        return StructureProduct(weights=[1], products=[self]) + -other

    def __mul__(self, coefficient: float) -> StructureProduct:
        assert isinstance(coefficient, float)
        return StructureProduct(weights=[coefficient], products=[self])

    def __rmul__(self, coefficient: float) -> StructureProduct:
        return self * coefficient

    @abstractmethod
    def get_cashflows(self, *args, **kwargs) -> tuple[np.array, np.array]:
        return None

    def pv(
        self,
        asof: datetime,
        discount_curve: BaseZeroCurve,
    ) -> float:
        dates, flows = self.get_cashflows()
        assert len(dates) == len(flows)
        dfs = discount_curve.df(dates - asof)
        return sum(
            dfs[i] * flows[i] for i in range(len(dates))
        )

    # Add @abstractmethod when implemented in subclasses
    def dv01(self, *args, **kwargs):
        raise NotImplementedError()

    # Add @abstractmethod when implemented in subclasses
    def modified_duration(self, *args, **kwargs):
        raise NotImplementedError()

    # Add @abstractmethod when implemented in subclasses
    def convexity(self, *args, **kwargs):
        raise NotImplementedError()

    # Add @abstractmethod when implemented in subclasses
    def rolldown(self, *args, **kwargs):
        raise NotImplementedError()

    def stats(self, *args, **kwargs) -> dict[str, float]:
        return dict(
            pv=self.pv(*args, **kwargs),
            dv01=self.dv01(*args, **kwargs),
            modified_duration=self.modified_duration(*args, **kwargs),
            convexity=self.convexity(*args, **kwargs),
            rolldown=self.rolldown(*args, **kwargs),
        )
