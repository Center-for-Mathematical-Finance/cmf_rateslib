import numpy as np

from ..products.bonds import ZCBond
from ..products.base_product import BaseProduct
from ..products.structure_product import StructureProduct


class Swap(StructureProduct):
    """
        Interest Rate Swap. Cashflows are defined from the payer's point of view.
    """

    def __init__(
        self,
        fixed_leg: BaseProduct,
        float_leg: BaseProduct,
    ) -> None:
        super().__init__(
            weights=[1, -1],
            products=[fixed_leg, float_leg],
        )

    def macauley_duration(self, *args, **kwargs) -> float:
        raise RuntimeError('Macauley duration is meaningless for swaps.')


class Swaption(StructureProduct):
    """
        Swaption. Cashflows are defined from the position of the swaption purchaser.
    """

    def __init__(
        self,
        swap: Swap,
        premium: ZCBond,
        execution_probability: np.array,
    ) -> None:
        assert isinstance(premium, (BaseProduct, StructureProduct)), (
            type(premium), premium, getattr(premium, 'params', None)
        )  # not to confuse with float
        super().__init__(
            weights=[execution_probability, -1],
            products=[swap, premium],
        )
