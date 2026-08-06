"""Order-construction helpers.

:class:`StrategyContext` exposes the common cases (``buy``, ``sell``, ``close``).
This module holds the full builder surface — multi-leg option structures,
brackets, and target-weight rebalancing — kept out of the context so that
protocol stays small and easy to reimplement (e.g. for a dry-run context in
tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sigmaloop.domain.bar import OptionChain
from sigmaloop.domain.order import BracketSpec, OrderIntent, SizingRequest
from sigmaloop.types import (
    InstrumentId,
    Money,
    OptionRight,
    OrderSide,
    OrderType,
    Price,
    Quantity,
    SizingMode,
    TimeInForce,
)

__all__ = ["OrderBuilder", "OptionLeg", "OptionStructure", "StrategyApi"]


@dataclass(slots=True)
class OrderBuilder:
    """Fluent builder for a single :class:`OrderIntent`.

    Exists so that unusual orders (GTD limit with a trailing stop and a tag)
    are expressible without a 12-argument function call.
    """

    instrument_id: InstrumentId
    side: OrderSide
    _sizing: SizingRequest | None = None
    _order_type: OrderType = OrderType.MARKET
    _limit_price: Price | None = None
    _stop_price: Price | None = None
    _tif: TimeInForce = TimeInForce.DAY
    _bracket: BracketSpec | None = None
    _tag: str = ""
    _metadata: dict[str, object] = field(default_factory=dict)

    def quantity(self, quantity: Quantity) -> OrderBuilder:
        raise NotImplementedError

    def notional(self, notional: Money) -> OrderBuilder:
        raise NotImplementedError

    def percent_equity(self, pct: float) -> OrderBuilder:
        raise NotImplementedError

    def risk_percent(self, pct: float) -> OrderBuilder:
        raise NotImplementedError

    def sizer(self, name: str, value: float = 0.0) -> OrderBuilder:
        """Route sizing to a named custom sizer."""
        raise NotImplementedError

    def limit(self, price: Price) -> OrderBuilder:
        raise NotImplementedError

    def stop(self, price: Price) -> OrderBuilder:
        raise NotImplementedError

    def bracket(
        self,
        *,
        stop_loss: Price | None = None,
        stop_loss_pct: float | None = None,
        take_profit: Price | None = None,
        take_profit_pct: float | None = None,
        trailing_pct: float | None = None,
    ) -> OrderBuilder:
        raise NotImplementedError

    def time_in_force(self, tif: TimeInForce) -> OrderBuilder:
        raise NotImplementedError

    def tag(self, tag: str) -> OrderBuilder:
        raise NotImplementedError

    def build(self) -> OrderIntent:
        """Validate and materialise. Raises ``ValidationError`` if incoherent."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OptionLeg:
    """One leg of a multi-leg option structure.

    ``ratio`` is relative, not absolute: the structure is sized once and each
    leg's quantity is ``structure_quantity * ratio``. That keeps a 1x2 ratio
    spread correct under any sizing mode.
    """

    right: OptionRight
    side: OrderSide
    ratio: float = 1.0
    #: Exactly one selector is used, checked at build time.
    target_delta: float | None = None
    strike: Price | None = None
    moneyness: float | None = None
    dte: int | None = None


@dataclass(frozen=True, slots=True)
class OptionStructure:
    """A named multi-leg structure resolved against a chain.

    Covers the requirement's examples (covered call, straddle) and the usual
    defined-risk structures. Resolution is atomic: if any leg cannot be found
    in the chain, the whole structure is skipped and a warning is recorded,
    rather than legging into a partial position the strategy never asked for.
    """

    name: str
    legs: tuple[OptionLeg, ...]
    #: Include a position in the underlying (e.g. +100 shares for a covered call).
    underlying_ratio: float = 0.0

    @classmethod
    def covered_call(cls, target_delta: float = 0.30, dte: int = 30) -> OptionStructure:
        raise NotImplementedError

    @classmethod
    def cash_secured_put(cls, target_delta: float = 0.30, dte: int = 30) -> OptionStructure:
        raise NotImplementedError

    @classmethod
    def straddle(cls, dte: int = 0, short: bool = False) -> OptionStructure:
        raise NotImplementedError

    @classmethod
    def strangle(cls, target_delta: float = 0.20, dte: int = 0, short: bool = False) -> OptionStructure:
        """The requirement's "SPY 0DTE 20-delta put and call"."""
        raise NotImplementedError

    @classmethod
    def vertical_spread(
        cls, right: OptionRight, short_delta: float, long_delta: float, dte: int = 30
    ) -> OptionStructure:
        raise NotImplementedError

    def resolve(self, chain: OptionChain) -> Sequence[tuple[InstrumentId, OrderSide, float]]:
        """Map legs onto concrete contracts: ``(instrument_id, side, ratio)``."""
        raise NotImplementedError


class StrategyApi(ABC):
    """Order-submission surface implemented by the engine's context."""

    @abstractmethod
    def order(self, instrument_id: InstrumentId, side: OrderSide) -> OrderBuilder:
        """Start a fluent order."""
        raise NotImplementedError

    @abstractmethod
    def submit_intent(self, intent: OrderIntent) -> OrderIntent:
        raise NotImplementedError

    @abstractmethod
    def submit_structure(
        self,
        structure: OptionStructure,
        chain: OptionChain,
        *,
        quantity: Quantity | None = None,
        sizing: SizingRequest | None = None,
        tag: str = "",
    ) -> Sequence[OrderIntent]:
        """Resolve and submit every leg atomically."""
        raise NotImplementedError

    @abstractmethod
    def target_weights(self, weights: Mapping[InstrumentId, float]) -> Sequence[OrderIntent]:
        """Emit only the deltas needed to reach the target weight vector."""
        raise NotImplementedError

    @abstractmethod
    def default_sizing(self) -> SizingRequest:
        """The run's configured default, used when an order omits sizing."""
        raise NotImplementedError

    @staticmethod
    def sizing(mode: SizingMode, value: float, **kwargs: object) -> SizingRequest:
        raise NotImplementedError
