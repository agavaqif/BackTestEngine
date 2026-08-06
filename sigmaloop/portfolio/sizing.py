"""Position sizing (Accounting requirement 4).

The engine never assumes a size. A strategy expresses *intent* via
:class:`~sigmaloop.domain.order.SizingRequest`; a :class:`PositionSizer` turns
that into a concrete quantity using live account state. This keeps sizing a
swappable policy — the same strategy can be run fixed-quantity for debugging
and percent-of-equity for evaluation without editing a line of strategy code.

All sizers must return a quantity that is:

* non-negative (direction lives on the order's side),
* a valid lot multiple for the instrument,
* expressed in **contracts** for options, not shares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import OrderIntent
from sigmaloop.portfolio.accounting import PortfolioView
from sigmaloop.types import Money, Price, Quantity, SizingMode

__all__ = [
    "SizingContext",
    "PositionSizer",
    "FixedQuantitySizer",
    "FixedNotionalSizer",
    "PercentEquitySizer",
    "RiskPercentSizer",
    "TargetWeightSizer",
    "CallableSizer",
    "CompositeSizer",
]


@dataclass(frozen=True, slots=True)
class SizingContext:
    """Inputs available when resolving one intent into a quantity."""

    intent: OrderIntent
    instrument: Instrument
    portfolio: PortfolioView
    #: Best available estimate of the fill price — usually the last close.
    #: The true fill price is not knowable at sizing time under next-bar-open,
    #: so sizing is intentionally approximate and the broker re-checks capital.
    reference_price: Price
    equity: Money
    available_cash: Money
    buying_power: Money
    #: Volatility estimate (ATR) when the strategy declared one; enables
    #: risk-based sizing without the strategy plumbing it through.
    volatility: float | None = None


class PositionSizer(ABC):
    """Resolves a :class:`SizingRequest` into a concrete quantity."""

    name: ClassVar[str] = "abstract"
    mode: ClassVar[SizingMode] = SizingMode.CUSTOM

    @abstractmethod
    def size(self, context: SizingContext) -> Quantity:
        """Return the quantity to trade; ``0.0`` suppresses the order."""
        raise NotImplementedError

    def apply_constraints(self, quantity: Quantity, context: SizingContext) -> Quantity:
        """Clamp to lot size, ``max_quantity``, and available buying power.

        Shared post-processing so every sizer honours the same rules; sizers
        should call this on their raw result rather than reimplementing it.
        """
        raise NotImplementedError


class FixedQuantitySizer(PositionSizer):
    """Trade exactly ``SizingRequest.value`` units."""

    name: ClassVar[str] = "fixed_quantity"
    mode: ClassVar[SizingMode] = SizingMode.FIXED_QUANTITY

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class FixedNotionalSizer(PositionSizer):
    """Trade a fixed cash notional: ``value / (price * multiplier)``."""

    name: ClassVar[str] = "fixed_notional"
    mode: ClassVar[SizingMode] = SizingMode.FIXED_NOTIONAL

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class PercentEquitySizer(PositionSizer):
    """Allocate a fraction of current equity: ``equity * pct / notional_per_unit``.

    Note this compounds — as equity grows, position sizes grow with it, which
    is usually intended but makes the equity curve path-dependent in a way
    fixed sizing is not.
    """

    name: ClassVar[str] = "percent_equity"
    mode: ClassVar[SizingMode] = SizingMode.PERCENT_EQUITY

    def __init__(self, use_buying_power: bool = False) -> None:
        raise NotImplementedError

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class RiskPercentSizer(PositionSizer):
    """Size so that hitting the stop loses exactly ``value`` of equity.

    ``qty = equity * risk_pct / (|entry - stop| * multiplier)``

    Requires either an explicit stop on the intent's bracket or a volatility
    estimate (``atr_multiple * ATR``); raises ``ConfigurationError`` if neither
    is available, rather than silently falling back to a different policy.
    """

    name: ClassVar[str] = "risk_percent"
    mode: ClassVar[SizingMode] = SizingMode.RISK_PERCENT

    def __init__(self, atr_multiple: float = 2.0) -> None:
        raise NotImplementedError

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class TargetWeightSizer(PositionSizer):
    """Trade the delta between current and target portfolio weight.

    The portfolio-mode rebalancing primitive: a strategy declares desired
    weights and the sizer emits only the difference, avoiding round-trip churn
    on names whose weight has not drifted.
    """

    name: ClassVar[str] = "target_weight"
    mode: ClassVar[SizingMode] = SizingMode.TARGET_WEIGHT

    def __init__(self, rebalance_threshold: float = 0.0) -> None:
        raise NotImplementedError

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class CallableSizer(PositionSizer):
    """Wraps an arbitrary user function — the "custom rule" escape hatch."""

    name: ClassVar[str] = "custom"
    mode: ClassVar[SizingMode] = SizingMode.CUSTOM

    def __init__(self, fn: Callable[[SizingContext], Quantity], name: str = "custom") -> None:
        raise NotImplementedError

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError


class CompositeSizer(PositionSizer):
    """Dispatches to the sizer matching the intent's declared mode.

    This is what the engine actually installs, so a single strategy can mix
    percent-of-equity entries with fixed-quantity hedges.
    """

    name: ClassVar[str] = "composite"

    def __init__(self, sizers: dict[SizingMode, PositionSizer] | None = None) -> None:
        raise NotImplementedError

    def register(self, mode: SizingMode, sizer: PositionSizer) -> None:
        raise NotImplementedError

    def size(self, context: SizingContext) -> Quantity:
        raise NotImplementedError
