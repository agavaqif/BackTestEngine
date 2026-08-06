"""Fill price resolution and synthetic spreads (Execution requirement 1).

Two separable concerns:

1. **Which quote side** a fill prints at — :class:`FillPriceModel`, driven by
   :class:`~sigmaloop.types.PriceSelection`. ``WORST`` (buy the ask, sell the
   bid) is the conservative default the requirements call for.
2. **Where the quote comes from** when the feed has none — :class:`SpreadModel`.
   CSV and Yahoo publish OHLCV only, so without a synthetic spread ``WORST``
   would silently collapse to ``MID`` and quietly overstate returns. Every
   synthesised quote is flagged (:attr:`Quote.is_synthetic`) and the run's
   summary reports how many fills relied on one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.bar import Bar, OptionQuote, Quote
from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import Basis, OrderSide, Price, PriceSelection

__all__ = [
    "SpreadModel",
    "FixedBpsSpreadModel",
    "TickSpreadModel",
    "VolatilitySpreadModel",
    "FillPriceModel",
    "QuoteFillPriceModel",
    "OhlcFillPriceModel",
]


class SpreadModel(ABC):
    """Synthesises a bid/ask when the data feed provides none."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def quote_for(self, bar: Bar, instrument: Instrument) -> Quote:
        """Build a synthetic :class:`Quote` centred on the bar's mark price."""
        raise NotImplementedError

    def half_spread(self, price: Price, instrument: Instrument) -> Price:
        """Per-side distance from mid, in price units."""
        raise NotImplementedError


class FixedBpsSpreadModel(SpreadModel):
    """Constant spread in basis points of price. Simple and predictable.

    Reasonable defaults: ~1-2bp for large-cap equities, far wider for options,
    where the spread is often the dominant cost and should be configured per
    asset class rather than guessed.
    """

    name: ClassVar[str] = "fixed_bps"

    def __init__(self, spread_bps: Basis = 2.0, option_spread_bps: Basis = 100.0) -> None:
        raise NotImplementedError

    def quote_for(self, bar: Bar, instrument: Instrument) -> Quote:
        raise NotImplementedError


class TickSpreadModel(SpreadModel):
    """Spread of N ticks. Appropriate for low-priced instruments where a bps
    spread would round to less than one tick."""

    name: ClassVar[str] = "ticks"

    def __init__(self, ticks: float = 1.0) -> None:
        raise NotImplementedError

    def quote_for(self, bar: Bar, instrument: Instrument) -> Quote:
        raise NotImplementedError


class VolatilitySpreadModel(SpreadModel):
    """Spread scaled by recent realised range — wider in stressed markets.

    Closer to reality than a fixed spread: liquidity evaporates exactly when
    strategies most want to trade, and a constant-cost assumption flatters
    high-turnover systems.
    """

    name: ClassVar[str] = "volatility"

    def __init__(self, base_bps: Basis = 2.0, range_multiplier: float = 0.1) -> None:
        raise NotImplementedError

    def quote_for(self, bar: Bar, instrument: Instrument) -> Quote:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PricingContext:
    """Everything the price model needs to resolve one fill."""

    instrument: Instrument
    side: OrderSide
    selection: PriceSelection
    bar: Bar | None = None
    option_quote: OptionQuote | None = None
    synthetic_quote: Quote | None = None


class FillPriceModel(ABC):
    """Resolves the reference price for a fill, before slippage.

    Kept separate from :class:`~sigmaloop.execution.slippage.SlippageModel` so
    the two costs stay attributable: the trade log records both the reference
    price and the slippage applied to it.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def resolve(self, context: PricingContext) -> Price:
        raise NotImplementedError

    @abstractmethod
    def supports(self, selection: PriceSelection) -> bool:
        raise NotImplementedError


class QuoteFillPriceModel(FillPriceModel):
    """Prices from bid/ask — real if the feed has quotes, synthetic otherwise.

    Fails loudly rather than silently when ``require_real_quotes`` is set and
    the feed has none, so an options run cannot accidentally be evaluated on
    invented spreads.
    """

    name: ClassVar[str] = "quote"

    def __init__(
        self,
        spread_model: SpreadModel | None = None,
        require_real_quotes: bool = False,
    ) -> None:
        raise NotImplementedError

    def resolve(self, context: PricingContext) -> Price:
        raise NotImplementedError

    def supports(self, selection: PriceSelection) -> bool:
        raise NotImplementedError


class OhlcFillPriceModel(FillPriceModel):
    """Prices from a chosen OHLC field, ignoring the spread entirely.

    Only valid with :attr:`~sigmaloop.types.PriceSelection.LAST`. Fast and
    familiar, but optimistic; the engine records a warning when it is used with
    a strategy that trades frequently.
    """

    name: ClassVar[str] = "ohlc"

    def __init__(self, field: str = "open") -> None:
        raise NotImplementedError

    def resolve(self, context: PricingContext) -> Price:
        raise NotImplementedError

    def supports(self, selection: PriceSelection) -> bool:
        raise NotImplementedError
