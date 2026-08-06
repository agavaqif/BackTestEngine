"""Engine events and the observer hook.

The engine is a bar loop, not a general event bus — a queue-based design would
cost an allocation and a dispatch per event per bar, which the performance NFR
does not permit. Events exist for *observation*: progress reporting, live
plotting, debugging and custom recorders subscribe without the loop paying for
them when nobody is listening.

Ordering within one bar is fixed and documented on
:class:`~sigmaloop.engine.core.BacktestEngine`; these types name the phases.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Protocol, runtime_checkable

from sigmaloop.domain.account import CorporateAction, EquityPoint
from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.order import Fill, Order, Rejection
from sigmaloop.domain.position import Trade
from sigmaloop.execution.expiry import ExpiryOutcome
from sigmaloop.types import RunId, RunState, UtcDatetime

__all__ = [
    "EventType",
    "Event",
    "RunStarted",
    "BarOpened",
    "OrderSubmitted",
    "OrderFilled",
    "OrderRejected",
    "TradeClosed",
    "OptionExpired",
    "CorporateActionApplied",
    "EquityUpdated",
    "BarClosed",
    "RunFinished",
    "EventListener",
    "EventBus",
]


class EventType(IntEnum):
    """Phase markers, ordered as they occur within a bar."""

    RUN_STARTED = auto()
    BAR_OPENED = auto()
    CORPORATE_ACTION_APPLIED = auto()
    ORDER_FILLED = auto()
    ORDER_REJECTED = auto()
    OPTION_EXPIRED = auto()
    TRADE_CLOSED = auto()
    EQUITY_UPDATED = auto()
    ORDER_SUBMITTED = auto()
    BAR_CLOSED = auto()
    RUN_FINISHED = auto()


@dataclass(frozen=True, slots=True)
class Event(ABC):
    """Base event. Carries the simulation timestamp, never wall-clock time."""

    timestamp: UtcDatetime

    @property
    def event_type(self) -> EventType:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    run_id: RunId
    total_bars_estimate: int


@dataclass(frozen=True, slots=True)
class BarOpened(Event):
    """A new snapshot has been read; nothing has been processed yet."""

    snapshot: MarketSnapshot
    bar_index: int


@dataclass(frozen=True, slots=True)
class CorporateActionApplied(Event):
    action: CorporateAction


@dataclass(frozen=True, slots=True)
class OrderFilled(Event):
    order: Order
    fill: Fill


@dataclass(frozen=True, slots=True)
class OrderRejected(Event):
    order: Order
    rejection: Rejection


@dataclass(frozen=True, slots=True)
class OptionExpired(Event):
    outcome: ExpiryOutcome


@dataclass(frozen=True, slots=True)
class TradeClosed(Event):
    trade: Trade


@dataclass(frozen=True, slots=True)
class EquityUpdated(Event):
    point: EquityPoint


@dataclass(frozen=True, slots=True)
class OrderSubmitted(Event):
    """The strategy raised an order and it passed sizing and risk."""

    order: Order


@dataclass(frozen=True, slots=True)
class BarClosed(Event):
    bar_index: int
    equity: float


@dataclass(frozen=True, slots=True)
class RunFinished(Event):
    run_id: RunId
    state: RunState
    bars_processed: int
    error: str | None = None


@runtime_checkable
class EventListener(Protocol):
    """Anything that wants to observe the run.

    Listeners must be cheap and must not mutate engine state; a slow listener
    directly extends every bar. Exceptions raised by a listener are logged and
    swallowed — an observer must never be able to fail a run.
    """

    def on_event(self, event: Event) -> None: ...


class EventBus:
    """Synchronous fan-out with a no-listener fast path.

    ``emit`` returns immediately when nothing is subscribed to that event type,
    so the instrumentation costs one dict lookup per phase in the common case.
    """

    __slots__ = ("_listeners", "_any_listeners")

    def __init__(self) -> None:
        raise NotImplementedError

    def subscribe(self, listener: EventListener, event_types: tuple[EventType, ...] = ()) -> None:
        """Subscribe to specific types, or to everything when ``event_types`` is empty."""
        raise NotImplementedError

    def unsubscribe(self, listener: EventListener) -> None:
        raise NotImplementedError

    def emit(self, event: Event) -> None:
        raise NotImplementedError

    @property
    def has_listeners(self) -> bool:
        raise NotImplementedError
