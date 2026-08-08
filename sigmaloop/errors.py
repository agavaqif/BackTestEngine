"""Exception hierarchy for SigmaLoop.

Every exception raised by the engine derives from :class:`SigmaLoopError`, so
callers can wrap an entire run in one ``except``. Each error carries structured
context (``context`` dict) in addition to a human-readable message, satisfying
the "clear error messages" NFR: the message states *what* failed, *where* in
the run (timestamp / instrument / order), and *what the user can do about it*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigmaloop.types import InstrumentId, OrderId, RejectReason, Symbol, UtcDatetime

__all__ = [
    "AccountingError",
    "ConfigurationError",
    "DataError",
    "DataIntegrityError",
    "DataNotAvailableError",
    "DataProviderError",
    "DuplicatePluginError",
    "EngineError",
    "EngineStateError",
    "ExecutionError",
    "IndicatorError",
    "InstrumentNotFoundError",
    "InsufficientCapitalError",
    "InsufficientHistoryError",
    "LookaheadViolationError",
    "MetricError",
    "OptionChainUnavailableError",
    "OrderRejectedError",
    "PluginError",
    "PluginNotFoundError",
    "PositionNotFoundError",
    "RunCancelledError",
    "SigmaLoopError",
    "StrategyContractError",
    "StrategyError",
    "ValidationError",
]


class SigmaLoopError(Exception):
    """Root of the SigmaLoop exception tree."""

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if not self.context:
            return self.message
        detail = ", ".join(f"{k}={v!r}" for k, v in self.context.items() if v is not None)
        return f"{self.message} ({detail})" if detail else self.message


# --------------------------------------------------------------------------- #
# Configuration / validation
# --------------------------------------------------------------------------- #


class ConfigurationError(SigmaLoopError):
    """A ``BacktestConfig`` (or sub-config) is internally inconsistent."""


class ValidationError(SigmaLoopError):
    """A dataclass invariant failed in ``__post_init__``."""


# --------------------------------------------------------------------------- #
# Plugins
# --------------------------------------------------------------------------- #


class PluginError(SigmaLoopError):
    """Base for plugin registry failures."""


class PluginNotFoundError(PluginError):
    """No plugin registered under the requested name in the given group."""

    def __init__(self, group: str, name: str, available: tuple[str, ...] = ()) -> None:
        listing = ", ".join(sorted(available)) if available else "none registered"
        super().__init__(
            f"No {group!r} plugin named {name!r}. Available: {listing}.",
            group=group,
            name=name,
            available=available,
        )
        self.group = group
        self.name = name
        self.available = available


class DuplicatePluginError(PluginError):
    """Two plugins claim the same name within one registry group."""


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


class DataError(SigmaLoopError):
    """Base for anything wrong with market data."""


class DataProviderError(DataError):
    """A provider failed to fetch (network, auth, rate limit, bad file)."""

    def __init__(self, provider: str, message: str, /, **context: Any) -> None:
        super().__init__(f"[{provider}] {message}", provider=provider, **context)
        self.provider = provider


class DataNotAvailableError(DataError):
    """The provider has no data for the requested symbol/range/timeframe."""

    def __init__(
        self,
        symbol: Symbol | InstrumentId,
        start: UtcDatetime | None = None,
        end: UtcDatetime | None = None,
        /,
        **context: Any,
    ) -> None:
        window = ""
        if start is not None or end is not None:
            window = f" in {start.isoformat() if start else '-inf'}..{end.isoformat() if end else '+inf'}"
        super().__init__(
            f"No data available for {symbol}{window}.",
            symbol=symbol,
            start=start,
            end=end,
            **context,
        )
        self.symbol = symbol
        self.start = start
        self.end = end


class InstrumentNotFoundError(DataError):
    """An ``InstrumentId`` referenced by a strategy or order is unknown."""


class OptionChainUnavailableError(DataError):
    """No option chain snapshot exists for the underlying at this timestamp."""


class DataIntegrityError(DataError):
    """Loaded data violates a structural invariant.

    Examples: non-monotonic timestamps, duplicate bars, ``high < low``,
    negative volume, or a bar outside the requested range.
    """


class LookaheadViolationError(DataError):
    """A component requested data at or after the current simulation clock.

    Raised by ``RunContext`` guards; this is a programming error in a strategy
    or indicator, never a data problem.
    """


# --------------------------------------------------------------------------- #
# Strategy / indicators
# --------------------------------------------------------------------------- #


class StrategyError(SigmaLoopError):
    """A user strategy raised, or misused the strategy API.

    The engine wraps user-code exceptions in this type, preserving ``__cause__``
    and annotating with the bar timestamp so failures are locatable.
    """


class StrategyContractError(StrategyError):
    """The strategy class does not satisfy its base-class contract.

    Examples: no ``on_bar`` override, a parameter missing from ``param_spec``,
    or a ``PortfolioStrategy`` that never declares a universe.
    """


class IndicatorError(SigmaLoopError):
    """An indicator failed to update or compute."""


class InsufficientHistoryError(IndicatorError):
    """An indicator's value was read before its warm-up period elapsed."""


# --------------------------------------------------------------------------- #
# Execution / accounting
# --------------------------------------------------------------------------- #


class ExecutionError(SigmaLoopError):
    """Base for simulated-broker failures."""


class OrderRejectedError(ExecutionError):
    """Raised only when ``AccountingConfig.raise_on_reject`` is True.

    By default rejections are non-fatal: the order is recorded with
    ``OrderStatus.REJECTED`` and delivered to ``Strategy.on_order_rejected``.
    """

    def __init__(self, order_id: OrderId, reason: RejectReason, message: str) -> None:
        # ``message`` is kept under a distinct name: ``SigmaLoopError.__str__``
        # renders ``self.message``, so rebinding it here would throw away the
        # order id and reason this class exists to put in front of the reader.
        super().__init__(
            f"Order {order_id} rejected ({reason.value}): {message}",
            order_id=order_id,
            reason=reason,
        )
        self.order_id = order_id
        self.reason = reason
        self.detail = message


class AccountingError(SigmaLoopError):
    """The ledger reached an impossible state (invariant breach)."""


class InsufficientCapitalError(AccountingError):
    """Cash or buying power would go negative and the policy is to raise."""


class PositionNotFoundError(AccountingError):
    """A close/reduce was applied to an instrument with no open position."""


# --------------------------------------------------------------------------- #
# Metrics / engine
# --------------------------------------------------------------------------- #


class MetricError(SigmaLoopError):
    """A metric could not be computed (e.g. fewer than two equity points)."""


class EngineError(SigmaLoopError):
    """Base for engine orchestration failures."""


class EngineStateError(EngineError):
    """An operation was attempted in the wrong ``RunState``."""


class RunCancelledError(EngineError):
    """The run was cancelled cooperatively via its cancellation token."""
