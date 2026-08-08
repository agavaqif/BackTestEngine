"""Option expiry, exercise and assignment.

Options positions do not simply vanish at expiry — they resolve into cash, into
underlying shares, or into nothing, and each path has a different P&L and a
different follow-on position. Getting this wrong is the single largest source
of error in options backtests, so it is handled by a dedicated component rather
than buried in the broker.

Resolution at expiry, for a contract held to the close of its expiry date:

* OTM (and unassigned)  -> expires worthless; premium is the full P&L.
* ITM long, PHYSICAL    -> exercised: long call buys shares at strike,
                           long put sells shares at strike.
* ITM short, PHYSICAL   -> assigned (subject to :attr:`ExpiryPolicy.assignment_probability`).
* ITM, CASH settled     -> intrinsic value credited/debited.

American-style early assignment is approximated, not simulated tick-by-tick:
short ITM options are assigned with configurable probability, raised near
ex-dividend dates for short calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.instrument import OptionContract
from sigmaloop.domain.order import Fill
from sigmaloop.domain.position import Position
from sigmaloop.types import Money, Price, Quantity, TradeCloseReason, UtcDatetime

__all__ = ["ExpiryPolicy", "ExpiryOutcome", "ExpiryEngine", "StandardExpiryEngine"]


@dataclass(frozen=True, slots=True)
class ExpiryPolicy:
    """Configurable rules for how expiring options are resolved."""

    #: Close positions this many bars before expiry instead of letting them
    #: expire. Avoids modelling assignment at the cost of realism.
    close_before_expiry_bars: int | None = None
    #: Probability a short ITM American option is assigned early, per bar ITM.
    early_assignment_probability: float = 0.0
    #: Assign short ITM options at expiry with this probability (1.0 == always).
    expiry_assignment_probability: float = 1.0
    #: Auto-exercise long options ITM by at least this much per share.
    auto_exercise_threshold: Price = 0.01
    #: If False, physical settlement is converted to cash-equivalent instead of
    #: creating an underlying position (useful when no equity data is loaded).
    allow_physical_settlement: bool = True
    #: Charge this per contract on exercise/assignment.
    exercise_fee_per_contract: Money = 0.0


@dataclass(frozen=True, slots=True)
class ExpiryOutcome:
    """What happened to one expiring position."""

    contract: OptionContract
    quantity: Quantity
    reason: TradeCloseReason
    settlement_price: Price
    underlying_price: Price
    cash_impact: Money
    #: Non-zero when PHYSICAL settlement creates/removes an underlying position.
    underlying_quantity_delta: Quantity = 0.0
    fees: Money = 0.0
    #: Synthetic fills recorded so the trade log shows the full round trip.
    fills: tuple[Fill, ...] = ()


class ExpiryEngine(ABC):
    """Resolves expiring and assigned option positions each bar."""

    @abstractmethod
    def expiring_positions(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        """Positions whose contract expires at (or before) this timestamp."""
        raise NotImplementedError

    @abstractmethod
    def resolve(self, position: Position, snapshot: MarketSnapshot) -> ExpiryOutcome:
        raise NotImplementedError

    @abstractmethod
    def check_early_assignment(
        self, position: Position, snapshot: MarketSnapshot
    ) -> ExpiryOutcome | None:
        """Called every bar on short option positions; ``None`` if untouched."""
        raise NotImplementedError

    def positions_to_close_early(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        """Positions the policy wants flattened before expiry."""
        raise NotImplementedError


class StandardExpiryEngine(ExpiryEngine):
    """Reference implementation of the resolution rules documented above.

    Deterministic given a seed: assignment draws come from the run's seeded RNG
    so an identical config reproduces identical results, which is a hard
    requirement for parameter sweeps to be comparable.
    """

    __slots__ = ("_policy", "_rng", "_settlement_source")

    def __init__(self, policy: ExpiryPolicy, seed: int = 0) -> None:
        raise NotImplementedError

    def expiring_positions(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        raise NotImplementedError

    def resolve(self, position: Position, snapshot: MarketSnapshot) -> ExpiryOutcome:
        raise NotImplementedError

    def check_early_assignment(
        self, position: Position, snapshot: MarketSnapshot
    ) -> ExpiryOutcome | None:
        raise NotImplementedError

    def _settlement_price(self, contract: OptionContract, snapshot: MarketSnapshot) -> Price:
        """Underlying's official close on the expiry date."""
        raise NotImplementedError

    def _resolve_at(
        self, position: Position, underlying_price: Price, at: UtcDatetime
    ) -> ExpiryOutcome:
        raise NotImplementedError
