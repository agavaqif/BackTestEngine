"""Slippage models — the price impact applied on top of the reference price.

Slippage is always adverse: it worsens the fill for whichever side is
transacting. Models return a per-unit adjustment, which the broker signs by
:attr:`OrderSide.sign` and records on the :class:`~sigmaloop.domain.order.Fill`
so cost attribution stays explicit in the trade log.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.bar import Bar
from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import Basis, OrderSide, Price, Quantity

__all__ = [
    "SlippageContext",
    "SlippageModel",
    "NoSlippageModel",
    "FixedBpsSlippageModel",
    "TickSlippageModel",
    "VolumeShareSlippageModel",
    "SpreadFractionSlippageModel",
]


@dataclass(frozen=True, slots=True)
class SlippageContext:
    """Inputs available to a slippage model for one prospective fill."""

    instrument: Instrument
    side: OrderSide
    quantity: Quantity
    reference_price: Price
    bar: Bar
    #: Realised volatility estimate (e.g. ATR) if the engine has one.
    volatility: float | None = None


class SlippageModel(ABC):
    """Computes adverse price impact per unit."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def slippage_per_unit(self, context: SlippageContext) -> Price:
        """Non-negative price degradation per share/contract."""
        raise NotImplementedError

    def fillable_quantity(self, context: SlippageContext) -> Quantity:
        """Cap on how much of the order can fill against this bar.

        Default is uncapped. :class:`VolumeShareSlippageModel` overrides it to
        enforce a participation limit, which is what prevents a backtest from
        "buying" more volume than the market printed.
        """
        raise NotImplementedError


class NoSlippageModel(SlippageModel):
    """Zero impact. Only honest when the spread model already carries the cost."""

    name: ClassVar[str] = "none"

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        raise NotImplementedError


class FixedBpsSlippageModel(SlippageModel):
    """Constant basis points of the reference price. The sane default."""

    name: ClassVar[str] = "fixed_bps"

    def __init__(self, bps: Basis = 1.0) -> None:
        raise NotImplementedError

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        raise NotImplementedError


class TickSlippageModel(SlippageModel):
    """A fixed number of ticks, independent of price level."""

    name: ClassVar[str] = "ticks"

    def __init__(self, ticks: float = 1.0) -> None:
        raise NotImplementedError

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        raise NotImplementedError


class VolumeShareSlippageModel(SlippageModel):
    """Square-root market impact plus a participation cap.

    Impact grows with the fraction of bar volume consumed:
    ``impact = coefficient * price * sqrt(quantity / bar_volume)``. Orders above
    ``max_volume_share`` of the bar are truncated (and the remainder either
    carries to the next bar or is cancelled, per broker policy).

    This is the model that keeps portfolio-mode results honest on small caps,
    where naive backtests routinely assume infinite liquidity.
    """

    name: ClassVar[str] = "volume_share"

    def __init__(self, coefficient: float = 0.1, max_volume_share: float = 0.025) -> None:
        raise NotImplementedError

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        raise NotImplementedError

    def fillable_quantity(self, context: SlippageContext) -> Quantity:
        raise NotImplementedError


class SpreadFractionSlippageModel(SlippageModel):
    """Slippage as a fraction of the prevailing spread.

    Natural for options, where the spread — not volume — is the binding cost.
    """

    name: ClassVar[str] = "spread_fraction"

    def __init__(self, fraction: float = 0.5) -> None:
        raise NotImplementedError

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        raise NotImplementedError
