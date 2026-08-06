"""Execution models — when and whether a working order fills.

This is the lookahead firewall (Execution requirement 2). A signal computed
from bar *t*'s close cannot transact at bar *t*'s open, because that price was
already in the past when the signal existed. The default
:class:`NextBarOpenExecutionModel` enforces that by construction: an order
submitted during bar *t* becomes eligible only at bar *t+1*.

The model answers two questions per working order per bar:

1. *Is it eligible yet?* — :meth:`ExecutionModel.is_eligible`
2. *Does the bar's price action trigger it, and at what price?* —
   :meth:`ExecutionModel.try_fill`

Limit/stop triggering uses the bar's high/low, which is the standard OHLC
approximation. Its known weakness — intrabar path is unknown, so a bar that
touches both a stop and a target is ambiguous — is resolved pessimistically:
the stop is assumed to trigger first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.bar import Bar, OptionQuote
from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import Order
from sigmaloop.types import ExecutionTiming, Price, Quantity, UtcDatetime

__all__ = [
    "FillDecision",
    "ExecutionContext",
    "ExecutionModel",
    "NextBarOpenExecutionModel",
    "NextBarCloseExecutionModel",
    "SameBarCloseExecutionModel",
]


@dataclass(frozen=True, slots=True)
class FillDecision:
    """Outcome of evaluating one order against one bar."""

    should_fill: bool
    quantity: Quantity = 0.0
    #: Pre-slippage price; the broker applies slippage and commission after.
    reference_price: Price = 0.0
    is_partial: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Everything needed to evaluate one order at one timestamp."""

    order: Order
    instrument: Instrument
    timestamp: UtcDatetime
    bar: Bar | None
    option_quote: OptionQuote | None = None
    is_session_close: bool = False


class ExecutionModel(ABC):
    """Decides eligibility and fill outcome for working orders."""

    name: ClassVar[str] = "abstract"

    @property
    @abstractmethod
    def timing(self) -> ExecutionTiming:
        raise NotImplementedError

    @property
    def introduces_lookahead(self) -> bool:
        """True for models that let bar *t* signals fill at bar *t* prices.

        The engine emits a prominent warning into
        :attr:`~sigmaloop.results.result.BacktestResult.warnings` when this is
        True, so a lookahead-tainted result can never be mistaken for a clean
        one.
        """
        raise NotImplementedError

    @abstractmethod
    def is_eligible(self, context: ExecutionContext) -> bool:
        """True if the order may be considered for filling at this bar."""
        raise NotImplementedError

    @abstractmethod
    def try_fill(self, context: ExecutionContext) -> FillDecision:
        """Evaluate the order against the bar."""
        raise NotImplementedError

    def should_expire(self, context: ExecutionContext) -> bool:
        """Apply time-in-force: DAY expires at session close, GTD at its date."""
        raise NotImplementedError


class NextBarOpenExecutionModel(ExecutionModel):
    """DEFAULT. Market orders fill at the next bar's open.

    Limit and stop orders become eligible at the next bar and are then tested
    against each subsequent bar's range until filled, cancelled or expired:

    * Buy limit  fills if ``low <= limit``, at ``min(open, limit)``.
    * Sell limit fills if ``high >= limit``, at ``max(open, limit)``.
    * Buy stop   triggers if ``high >= stop``, at ``max(open, stop)``.
    * Sell stop  triggers if ``low <= stop``, at ``min(open, stop)``.

    Gap handling matters: when a bar opens through the level, the fill uses the
    open, not the level. Assuming the limit price on a gap is the most common
    way backtests manufacture free money.
    """

    name: ClassVar[str] = "next_bar_open"

    def __init__(self, allow_gap_improvement: bool = False) -> None:
        raise NotImplementedError

    @property
    def timing(self) -> ExecutionTiming:
        raise NotImplementedError

    @property
    def introduces_lookahead(self) -> bool:
        raise NotImplementedError

    def is_eligible(self, context: ExecutionContext) -> bool:
        raise NotImplementedError

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        raise NotImplementedError


class NextBarCloseExecutionModel(ExecutionModel):
    """Fills at the next bar's close — models a VWAP/TWAP-style working order."""

    name: ClassVar[str] = "next_bar_close"

    @property
    def timing(self) -> ExecutionTiming:
        raise NotImplementedError

    @property
    def introduces_lookahead(self) -> bool:
        raise NotImplementedError

    def is_eligible(self, context: ExecutionContext) -> bool:
        raise NotImplementedError

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        raise NotImplementedError


class SameBarCloseExecutionModel(ExecutionModel):
    """Fills at the close of the bar that produced the signal.

    LOOKAHEAD-PRONE and offered only for comparison against published results
    that use this convention. :attr:`introduces_lookahead` is True.
    """

    name: ClassVar[str] = "same_bar_close"

    @property
    def timing(self) -> ExecutionTiming:
        raise NotImplementedError

    @property
    def introduces_lookahead(self) -> bool:
        raise NotImplementedError

    def is_eligible(self, context: ExecutionContext) -> bool:
        raise NotImplementedError

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        raise NotImplementedError
