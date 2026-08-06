"""Domain value objects — the vocabulary shared by every other layer.

Nothing in this package imports from ``data``, ``strategy``, ``execution``,
``portfolio``, ``engine`` or ``results``; the dependency arrow points only
inward, toward :mod:`sigmaloop.types` and :mod:`sigmaloop.errors`.
"""

from __future__ import annotations

from sigmaloop.domain.account import AccountState, CashFlow, CorporateAction, EquityPoint
from sigmaloop.domain.bar import (
    Bar,
    BarSeries,
    Greeks,
    MarketSnapshot,
    OptionChain,
    OptionQuote,
    PricedInstrument,
    Quote,
)
from sigmaloop.domain.instrument import (
    Equity,
    Instrument,
    InstrumentRegistry,
    OptionContract,
)
from sigmaloop.domain.order import (
    BracketSpec,
    Fill,
    Order,
    OrderIntent,
    Rejection,
    SizingRequest,
)
from sigmaloop.domain.position import Lot, OptionTrade, Position, Trade

__all__ = [
    "AccountState",
    "Bar",
    "BarSeries",
    "BracketSpec",
    "CashFlow",
    "CorporateAction",
    "Equity",
    "EquityPoint",
    "Fill",
    "Greeks",
    "Instrument",
    "InstrumentRegistry",
    "Lot",
    "MarketSnapshot",
    "OptionChain",
    "OptionContract",
    "OptionQuote",
    "OptionTrade",
    "Order",
    "OrderIntent",
    "Position",
    "PricedInstrument",
    "Quote",
    "Rejection",
    "SizingRequest",
    "Trade",
]
