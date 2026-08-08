"""Pre-trade risk checks and margin.

The risk layer sits between sizing and the broker. Every sized order passes
through it, and it is the single place that can veto a trade for a
non-market reason (capital, concentration, leverage, shorting rules).

Checks are ordered cheapest-first and short-circuit, because in portfolio mode
this runs on every candidate order on every bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import Order, Rejection
from sigmaloop.portfolio.accounting import PortfolioView
from sigmaloop.types import MarginModel, Money, Percent, Price, Quantity

__all__ = [
    "RiskContext",
    "RiskCheck",
    "CapitalCheck",
    "ShortingCheck",
    "ConcentrationCheck",
    "LeverageCheck",
    "MaxPositionsCheck",
    "RiskManager",
    "MarginCalculator",
    "RegTMarginCalculator",
]


@dataclass(frozen=True, slots=True)
class RiskContext:
    """State a risk check may consult."""

    order: Order
    instrument: Instrument
    portfolio: PortfolioView
    reference_price: Price
    estimated_cost: Money
    is_reducing: bool


class RiskCheck(ABC):
    """One veto-capable pre-trade rule."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def check(self, context: RiskContext) -> Rejection | None:
        """``None`` to allow; a :class:`Rejection` to veto."""
        raise NotImplementedError

    @property
    def applies_to_reducing_orders(self) -> bool:
        """Most checks skip position-reducing orders — closing risk is not new
        risk, and blocking an exit is worse than allowing it."""
        raise NotImplementedError


class CapitalCheck(RiskCheck):
    """Rejects (or flags) orders that exceed available capital.

    Accounting requirement #2. Whether a breach rejects the order or is merely
    recorded is a config decision, not this check's — it always reports, and
    the :class:`RiskManager` applies the policy.
    """

    name: ClassVar[str] = "capital"

    def check(self, context: RiskContext) -> Rejection | None:
        raise NotImplementedError


class ShortingCheck(RiskCheck):
    """Blocks shorts when disabled globally or the instrument is not shortable."""

    name: ClassVar[str] = "shorting"

    def __init__(self, allow_short: bool = True) -> None:
        raise NotImplementedError

    def check(self, context: RiskContext) -> Rejection | None:
        raise NotImplementedError


class ConcentrationCheck(RiskCheck):
    """Caps any single position's weight in the portfolio."""

    name: ClassVar[str] = "concentration"

    def __init__(self, max_weight: Percent = 0.25) -> None:
        raise NotImplementedError

    def check(self, context: RiskContext) -> Rejection | None:
        raise NotImplementedError


class LeverageCheck(RiskCheck):
    """Caps gross exposure relative to equity."""

    name: ClassVar[str] = "leverage"

    def __init__(self, max_leverage: float = 1.0) -> None:
        raise NotImplementedError

    def check(self, context: RiskContext) -> Rejection | None:
        raise NotImplementedError


class MaxPositionsCheck(RiskCheck):
    """Caps the count of concurrent open positions."""

    name: ClassVar[str] = "max_positions"

    def __init__(self, max_positions: int = 100) -> None:
        raise NotImplementedError

    def check(self, context: RiskContext) -> Rejection | None:
        raise NotImplementedError


class RiskManager:
    """Runs the configured checks in order and applies the breach policy."""

    __slots__ = ("_checks", "_flag_only", "_flagged")

    def __init__(self, checks: Sequence[RiskCheck], flag_only: bool = False) -> None:
        raise NotImplementedError

    def evaluate(self, context: RiskContext) -> Rejection | None:
        """First veto wins. Under ``flag_only`` the breach is recorded on the
        order's metadata and the order proceeds, so a run can surface how often
        a strategy would have over-traded without truncating its behaviour."""
        raise NotImplementedError

    def add_check(self, check: RiskCheck) -> None:
        raise NotImplementedError

    def flagged_orders(self) -> Sequence[tuple[Order, Rejection]]:
        """Breaches that were recorded rather than enforced."""
        raise NotImplementedError


class MarginCalculator(ABC):
    """Computes margin requirements for positions and prospective orders."""

    @abstractmethod
    def initial_margin(self, instrument: Instrument, quantity: Quantity, price: Price) -> Money:
        raise NotImplementedError

    @abstractmethod
    def maintenance_margin(self, instrument: Instrument, quantity: Quantity, price: Price) -> Money:
        raise NotImplementedError

    @abstractmethod
    def buying_power(self, equity: Money, margin_used: Money) -> Money:
        raise NotImplementedError


class RegTMarginCalculator(MarginCalculator):
    """US Reg-T: 50% initial / 25% maintenance on equities.

    Long options are paid in full (no margin). Short options use the standard
    naked-option formula — the larger of 20% of underlying less OTM amount, or
    10% of underlying — which is why short premium strategies can consume far
    more buying power than their premium suggests.
    """

    __slots__ = ("_initial_pct", "_maintenance_pct", "_model")

    def __init__(
        self,
        initial_pct: Percent = 0.5,
        maintenance_pct: Percent = 0.25,
        model: MarginModel = MarginModel.REG_T,
    ) -> None:
        raise NotImplementedError

    def initial_margin(self, instrument: Instrument, quantity: Quantity, price: Price) -> Money:
        raise NotImplementedError

    def maintenance_margin(self, instrument: Instrument, quantity: Quantity, price: Price) -> Money:
        raise NotImplementedError

    def buying_power(self, equity: Money, margin_used: Money) -> Money:
        raise NotImplementedError
