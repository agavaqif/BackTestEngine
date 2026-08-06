"""Simulated broker — the order book, matching loop and fill factory.

Responsibilities, in the order they run each bar:

1. Expire orders whose time-in-force has lapsed.
2. For each working order, ask the :class:`~sigmaloop.execution.models.ExecutionModel`
   whether it is eligible and whether the bar triggers it.
3. Resolve the reference price
   (:class:`~sigmaloop.execution.pricing.FillPriceModel`), apply slippage and
   commission, and emit a :class:`~sigmaloop.domain.order.Fill`.
4. Activate any bracket children once their parent fills.

The broker deliberately does NOT touch the ledger. It emits fills; the
:class:`~sigmaloop.portfolio.accounting.Portfolio` applies them. Keeping matching
and accounting apart is what makes both independently testable, and it is why
a rejected order can never leave a half-applied cash effect behind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.instrument import InstrumentRegistry
from sigmaloop.domain.order import Fill, Order, Rejection
from sigmaloop.execution.commission import CommissionModel
from sigmaloop.execution.models import ExecutionModel
from sigmaloop.execution.pricing import FillPriceModel
from sigmaloop.execution.slippage import SlippageModel
from sigmaloop.types import InstrumentId, OrderId, PriceSelection, UtcDatetime

__all__ = ["BrokerResult", "Broker", "SimulatedBroker"]


@dataclass(slots=True)
class BrokerResult:
    """Everything the broker produced during one bar."""

    fills: list[Fill] = field(default_factory=list)
    rejections: list[tuple[OrderId, Rejection]] = field(default_factory=list)
    expirations: list[OrderId] = field(default_factory=list)
    cancellations: list[OrderId] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError


class Broker(ABC):
    """Order intake and matching contract."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept an order into the working book (status -> ACCEPTED)."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, order_id: OrderId, at: UtcDatetime, reason: str = "") -> bool:
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self, at: UtcDatetime, instrument_id: InstrumentId | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def process_bar(self, snapshot: MarketSnapshot) -> BrokerResult:
        """Advance every working order against this bar."""
        raise NotImplementedError

    @abstractmethod
    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: OrderId) -> Order | None:
        raise NotImplementedError


class SimulatedBroker(Broker):
    """Bar-driven matching engine.

    Working orders are indexed by instrument so a bar only walks the orders
    that could possibly fill on it — O(orders_on_this_instrument) rather than
    O(all_open_orders). In portfolio mode with thousands of resting brackets,
    that difference is the run.
    """

    __slots__ = (
        "_execution_model",
        "_price_model",
        "_slippage_model",
        "_commission_model",
        "_registry",
        "_price_selection",
        "_orders",
        "_by_instrument",
        "_brackets",
        "_next_fill_seq",
    )

    def __init__(
        self,
        execution_model: ExecutionModel,
        price_model: FillPriceModel,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
        registry: InstrumentRegistry,
        price_selection: PriceSelection = PriceSelection.WORST,
    ) -> None:
        raise NotImplementedError

    def submit(self, order: Order) -> None:
        raise NotImplementedError

    def cancel(self, order_id: OrderId, at: UtcDatetime, reason: str = "") -> bool:
        raise NotImplementedError

    def cancel_all(self, at: UtcDatetime, instrument_id: InstrumentId | None = None) -> int:
        raise NotImplementedError

    def process_bar(self, snapshot: MarketSnapshot) -> BrokerResult:
        raise NotImplementedError

    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        raise NotImplementedError

    def get_order(self, order_id: OrderId) -> Order | None:
        raise NotImplementedError

    # ---- internals ---------------------------------------------------------- #

    def _build_fill(self, order: Order, snapshot: MarketSnapshot) -> Fill:
        """Price -> slippage -> commission -> :class:`Fill`."""
        raise NotImplementedError

    def _activate_brackets(self, parent: Order, fill: Fill) -> list[Order]:
        """Materialise stop-loss / take-profit children from the parent's fill.

        Children are OCO: filling one cancels its sibling.
        """
        raise NotImplementedError
