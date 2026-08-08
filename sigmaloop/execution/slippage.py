"""Slippage models — the price impact applied on top of the reference price.

Slippage is always adverse: it worsens the fill for whichever side is
transacting. Models return a per-unit adjustment, which the broker signs by
:attr:`OrderSide.sign` and records on the :class:`~sigmaloop.domain.order.Fill`
so cost attribution stays explicit in the trade log.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.bar import Bar
from sigmaloop.domain.instrument import Instrument
from sigmaloop.errors import ValidationError
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

#: One basis point as a fraction.
_BPS = 1e-4


def _require_non_negative(value: float, name: str, owner: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(
            f"{owner}.{name} must be a non-negative, finite number; negative "
            "slippage is price improvement, which no honest backtest assumes.",
            **{name: value},
        )
    return value


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
        return context.quantity


class NoSlippageModel(SlippageModel):
    """Zero impact. Only honest when the spread model already carries the cost."""

    name: ClassVar[str] = "none"

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        return 0.0


class FixedBpsSlippageModel(SlippageModel):
    """Constant basis points of the reference price. The sane default."""

    name: ClassVar[str] = "fixed_bps"

    def __init__(self, bps: Basis = 1.0) -> None:
        self._bps = _require_non_negative(bps, "bps", type(self).__name__)

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        return abs(context.reference_price) * self._bps * _BPS


class TickSlippageModel(SlippageModel):
    """A fixed number of ticks, independent of price level."""

    name: ClassVar[str] = "ticks"

    def __init__(self, ticks: float = 1.0) -> None:
        self._ticks = _require_non_negative(ticks, "ticks", type(self).__name__)

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        return self._ticks * context.instrument.tick_size


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
        self._coefficient = _require_non_negative(coefficient, "coefficient", type(self).__name__)
        self._max_volume_share = _require_non_negative(
            max_volume_share, "max_volume_share", type(self).__name__
        )

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        volume = context.bar.volume
        price = abs(context.reference_price)
        if volume <= 0.0:
            # Nothing printed on this bar, so participation is undefined rather
            # than zero. :meth:`fillable_quantity` refuses the fill outright;
            # charging the full coefficient here keeps any caller that ignores
            # the cap from reading "no volume" as "no cost".
            return self._coefficient * price
        # Capped at the participation limit before taking the root: the model
        # only ever prices the slice it would actually let through, so an
        # oversized order is not charged for impact the broker then truncates.
        participating = min(abs(context.quantity), self._max_volume_share * volume)
        return self._coefficient * price * math.sqrt(participating / volume)

    def fillable_quantity(self, context: SlippageContext) -> Quantity:
        volume = context.bar.volume
        if volume <= 0.0:
            return 0.0
        return min(context.quantity, self._max_volume_share * volume)


class SpreadFractionSlippageModel(SlippageModel):
    """Slippage as a fraction of the prevailing spread.

    Natural for options, where the spread — not volume — is the binding cost.
    """

    name: ClassVar[str] = "spread_fraction"

    def __init__(self, fraction: float = 0.5) -> None:
        self._fraction = _require_non_negative(fraction, "fraction", type(self).__name__)

    def slippage_per_unit(self, context: SlippageContext) -> Price:
        """Zero when the bar carries no book — there is no spread to take a
        fraction of, and inventing one here would double-charge whatever the
        spread model already applied at pricing time. The broker attaches the
        quote it priced against (real or synthetic) to the bar it passes in, so
        this is only reached when the run has no spread model at all."""
        quote = context.bar.quote
        if quote is None:
            return 0.0
        return max(quote.spread, 0.0) * self._fraction
