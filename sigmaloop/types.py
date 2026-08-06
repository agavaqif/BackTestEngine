"""Primitive type aliases and enumerations shared across SigmaLoop.

This module is dependency-free by design: every other package may import it,
and it must never import from them. It is the bottom of the dependency graph.

Numeric policy
--------------
Prices, quantities and cash are ``float`` (IEEE-754 float64), not ``Decimal``.
The performance NFR (vectorised indicators, millions of bars, parallel sweeps)
makes ``Decimal`` untenable, and float64 carries ~15 significant digits, which
is ample for simulated cash balances. Rounding to a currency's minor unit
happens once, at reporting boundaries, via ``sigmaloop.utils.money.round_money``.

Time policy
-----------
Every ``datetime`` crossing a public API is timezone-aware and normalised to
UTC. Columnar storage uses ``numpy.datetime64[ns]`` / int64 epoch-nanoseconds.
Bars are RIGHT-labelled: ``Bar.timestamp`` is the instant the bar CLOSED.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum, IntEnum, StrEnum, auto
from typing import NewType, TypeAlias

__all__ = [
    # aliases
    "Symbol",
    "InstrumentId",
    "OrderId",
    "IntentId",
    "FillId",
    "TradeId",
    "RunId",
    "StrategyId",
    "Price",
    "Quantity",
    "Money",
    "Percent",
    "Basis",
    "EpochNanos",
    "UtcDatetime",
    "ParamValue",
    "ParamDict",
    # enums
    "AssetClass",
    "OptionRight",
    "OptionStyle",
    "SettlementType",
    "Timeframe",
    "OrderSide",
    "PositionSide",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "RejectReason",
    "ExecutionTiming",
    "PriceSelection",
    "FillLiquidity",
    "StrategyMode",
    "SizingMode",
    "TradeCloseReason",
    "CorporateActionType",
    "MarginModel",
    "RunState",
    "Currency",
]

# --------------------------------------------------------------------------- #
# Identifier aliases (NewType => zero runtime cost, checked statically)
# --------------------------------------------------------------------------- #

Symbol = NewType("Symbol", str)
"""Human ticker, e.g. ``"SPY"``. Not unique across asset classes."""

InstrumentId = NewType("InstrumentId", str)
"""Canonical, globally unique instrument key.

Equity:  ``"EQ:SPY"``
Option:  ``"OPT:SPY:20250117:C:00500000"`` (OCC-derived: root, expiry, right, strike*1000)
"""

OrderId = NewType("OrderId", str)
IntentId = NewType("IntentId", str)
FillId = NewType("FillId", str)
TradeId = NewType("TradeId", str)
RunId = NewType("RunId", str)
StrategyId = NewType("StrategyId", str)

# --------------------------------------------------------------------------- #
# Numeric aliases (documentation-grade; all resolve to float)
# --------------------------------------------------------------------------- #

Price: TypeAlias = float
"""Per-unit price in the instrument's quote currency. Never multiplier-scaled."""

Quantity: TypeAlias = float
"""Signed for positions (negative == short), unsigned for orders/fills.

Options are counted in CONTRACTS, not shares. Multiply by
``Instrument.multiplier`` to obtain notional.
"""

Money: TypeAlias = float
"""Cash amount in the account's base currency."""

Percent: TypeAlias = float
"""Fractional, not per-hundred: 0.075 == 7.5%."""

Basis: TypeAlias = float
"""Basis points: 1.0 == 0.01%."""

EpochNanos: TypeAlias = int
"""UTC nanoseconds since the Unix epoch — the columnar time representation."""

UtcDatetime: TypeAlias = datetime
"""A timezone-aware ``datetime`` guaranteed to be in UTC."""

ParamValue: TypeAlias = int | float | bool | str | None
ParamDict: TypeAlias = dict[str, ParamValue]


# --------------------------------------------------------------------------- #
# Instrument taxonomy
# --------------------------------------------------------------------------- #


class AssetClass(StrEnum):
    """Instrument category. Drives multiplier, margin and settlement rules."""

    EQUITY = "equity"
    ETF = "etf"
    OPTION = "option"
    INDEX = "index"
    CASH = "cash"
    FUTURE = "future"  # Reserved — see "Future Improvements"; not simulated in v1.


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class OptionStyle(StrEnum):
    AMERICAN = "american"
    EUROPEAN = "european"


class SettlementType(StrEnum):
    """How an in-the-money option resolves at expiry."""

    PHYSICAL = "physical"  # Deliver/receive the underlying (equity options).
    CASH = "cash"  # Credit/debit intrinsic value (index options).


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


class Timeframe(StrEnum):
    """Bar aggregation period.

    ``TICK`` is accepted by the data layer but is not a schedulable engine
    step in v1; the engine clock requires a fixed-width timeframe.
    """

    TICK = "tick"
    S1 = "1s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MO1 = "1mo"

    @property
    def duration(self) -> timedelta:
        """Nominal wall-clock width of one bar.

        Raises ``ValueError`` for ``TICK`` and for calendar-variable periods
        (``W1``/``MO1`` return their nominal 7d / 30d approximation).
        """
        raise NotImplementedError

    @property
    def is_intraday(self) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Orders and execution
# --------------------------------------------------------------------------- #


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """``+1`` for BUY, ``-1`` for SELL — used to sign quantity deltas."""
        raise NotImplementedError

    @property
    def opposite(self) -> OrderSide:
        raise NotImplementedError


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    MARKET_ON_OPEN = "market_on_open"
    MARKET_ON_CLOSE = "market_on_close"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"  # Good-till-cancelled
    GTD = "gtd"  # Good-till-date (requires Order.expires_at)
    IOC = "ioc"  # Immediate-or-cancel
    FOK = "fok"  # Fill-or-kill


class OrderStatus(StrEnum):
    """Order lifecycle. Terminal states: FILLED, CANCELLED, REJECTED, EXPIRED."""

    PENDING_NEW = "pending_new"  # Created by strategy, not yet accepted by broker.
    ACCEPTED = "accepted"  # Live in the simulated book, awaiting a fillable bar.
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError


class RejectReason(StrEnum):
    """Why the simulated broker or risk layer refused an order.

    Whether a violation rejects or merely flags is controlled by
    ``AccountingConfig.on_capital_breach`` (Accounting requirement #2).
    """

    INSUFFICIENT_CAPITAL = "insufficient_capital"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    NO_MARKET_DATA = "no_market_data"
    ZERO_OR_NEGATIVE_QUANTITY = "zero_or_negative_quantity"
    BELOW_MIN_LOT = "below_min_lot"
    INSTRUMENT_EXPIRED = "instrument_expired"
    INSTRUMENT_NOT_TRADEABLE = "instrument_not_tradeable"
    SHORTING_DISALLOWED = "shorting_disallowed"
    NOT_SHORTABLE = "not_shortable"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    MAX_POSITION_EXCEEDED = "max_position_exceeded"
    MARKET_CLOSED = "market_closed"
    STALE_QUOTE = "stale_quote"
    UNSUPPORTED_ORDER_TYPE = "unsupported_order_type"
    LIQUIDITY_CAP = "liquidity_cap"  # Order exceeded max % of bar volume.


class ExecutionTiming(StrEnum):
    """When a signal raised on bar *t* is allowed to transact.

    ``NEXT_BAR_OPEN`` is the engine default and the only lookahead-free choice
    (Execution requirement #2). The others exist for research comparison and
    emit a warning into ``BacktestResult.warnings``.
    """

    NEXT_BAR_OPEN = "next_bar_open"
    NEXT_BAR_CLOSE = "next_bar_close"
    SAME_BAR_CLOSE = "same_bar_close"  # Lookahead-prone.


class PriceSelection(StrEnum):
    """Which side of the spread a fill is priced at.

    ``WORST`` is the conservative default: pay the ask when buying, hit the bid
    when selling (Execution requirement #1).
    """

    MID = "mid"
    WORST = "worst"
    BEST = "best"  # Optimistic; research only.
    LAST = "last"  # Trade/close price, ignoring the spread.


class FillLiquidity(StrEnum):
    TAKER = "taker"
    MAKER = "maker"


# --------------------------------------------------------------------------- #
# Strategy / sizing / accounting
# --------------------------------------------------------------------------- #


class StrategyMode(StrEnum):
    """Determines which strategy base class and data plan the engine wires up."""

    SINGLE_ASSET = "single_asset"
    SINGLE_ASSET_OPTIONS = "single_asset_options"
    PORTFOLIO = "portfolio"


class SizingMode(StrEnum):
    FIXED_QUANTITY = "fixed_quantity"
    FIXED_NOTIONAL = "fixed_notional"
    PERCENT_EQUITY = "percent_equity"
    RISK_PERCENT = "risk_percent"  # Size from stop distance and risk budget.
    TARGET_WEIGHT = "target_weight"  # Rebalance toward a portfolio weight.
    CUSTOM = "custom"  # Delegates to a user callable.


class TradeCloseReason(StrEnum):
    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    OPTION_EXPIRY_WORTHLESS = "option_expiry_worthless"
    OPTION_EXERCISE = "option_exercise"
    OPTION_ASSIGNMENT = "option_assignment"
    EOD_LIQUIDATION = "eod_liquidation"
    MARGIN_CALL = "margin_call"
    DELISTED = "delisted"
    END_OF_BACKTEST = "end_of_backtest"


class CorporateActionType(StrEnum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"


class MarginModel(StrEnum):
    CASH = "cash"  # No leverage; shorts and naked options disallowed.
    REG_T = "reg_t"  # 50% initial / 25% maintenance on equities.
    PORTFOLIO = "portfolio"  # Risk-based; reserved.


class RunState(IntEnum):
    """Engine lifecycle, surfaced for progress reporting and cancellation."""

    CREATED = auto()
    WARMING_UP = auto()
    RUNNING = auto()
    FINALISING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class _Sentinel(Enum):
    """Internal sentinel for "argument not supplied" where ``None`` is valid."""

    UNSET = auto()


UNSET = _Sentinel.UNSET
