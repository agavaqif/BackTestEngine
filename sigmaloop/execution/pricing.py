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

Division of labour with the execution model
-------------------------------------------
The :class:`~sigmaloop.execution.models.ExecutionModel` decides *where in the
bar* an order transacts — the open, the close, or the limit/stop level it was
triggered through. This module decides *which side of the book* that print
lands on. The broker adds the two (:meth:`FillPriceModel.spread_adjustment`),
so a gapped limit fill still pays the spread and a market fill at the open is
not silently repriced to the bar's close.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

from sigmaloop.domain.bar import Bar, OptionQuote, Quote
from sigmaloop.domain.instrument import Instrument
from sigmaloop.errors import ExecutionError, ValidationError
from sigmaloop.types import AssetClass, Basis, OrderSide, Price, PriceSelection
from sigmaloop.utils.money import round_to_tick

__all__ = [
    "SpreadModel",
    "FixedBpsSpreadModel",
    "TickSpreadModel",
    "VolatilitySpreadModel",
    "FillPriceModel",
    "QuoteFillPriceModel",
    "OhlcFillPriceModel",
]

#: One basis point as a fraction. 1bp == 0.01% == 1e-4.
_BPS = 1e-4

#: OHLC fields :class:`OhlcFillPriceModel` will price from.
_OHLC_FIELDS = ("open", "high", "low", "close", "typical_price", "vwap")

#: Selections that transact at the mid, so no side adjustment applies.
_NEUTRAL_SELECTIONS = frozenset({PriceSelection.MID, PriceSelection.LAST})


def _require_non_negative(value: float, name: str, owner: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(
            f"{owner}.{name} must be a non-negative, finite number; a negative "
            "spread would pay the trader to cross it.",
            **{name: value},
        )
    return value


class SpreadModel(ABC):
    """Synthesises a bid/ask when the data feed provides none."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def quote_for(self, bar: Bar, instrument: Instrument, at: Price | None = None) -> Quote:
        """Build a synthetic :class:`Quote` centred on ``at``.

        ``at`` is the price the fill is about to transact at — the bar's open
        under next-bar-open timing, or the limit/stop level a resting order was
        triggered through. It defaults to the bar's mark for callers that have
        no such level, but the execution layer always supplies one: a bar that
        opened at 100 and closed at 140 has a very different spread at the two
        ends, and charging the close's spread on an opening fill misprices
        every gap day in the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def half_spread(self, price: Price, instrument: Instrument) -> Price:
        """Per-side distance from mid, in price units."""
        raise NotImplementedError

    def _centred_quote(self, mid: Price, half: Price, instrument: Instrument) -> Quote:
        """Widen ``mid`` by ``half`` on each side and snap to the tick grid.

        Nearest, so the modelled spread comes back at the width the model asked
        for; rounding each side outward instead would quietly double a 2bp
        equity spread into four ticks. What nearest *can* do is round both sides
        onto the same tick and hand back a zero spread — the silent collapse to
        MID this whole module exists to prevent — so a degenerate quote is
        widened to one tick, which is the narrowest market there is.

        The bid is floored at zero: a negative one is not a price, and a wide
        model on a penny stock would otherwise produce one.
        """
        tick = instrument.tick_size
        bid = round_to_tick(max(mid - half, 0.0), tick, "nearest")
        ask = round_to_tick(mid + half, tick, "nearest")
        if ask - bid < tick * 0.5:
            bid = max(round_to_tick(mid, tick, "down"), 0.0)
            ask = round_to_tick(bid + tick, tick, "nearest")
        return Quote(bid=bid, ask=ask, is_synthetic=True)


class FixedBpsSpreadModel(SpreadModel):
    """Constant spread in basis points of price. Simple and predictable.

    Reasonable defaults: ~1-2bp for large-cap equities, far wider for options,
    where the spread is often the dominant cost and should be configured per
    asset class rather than guessed.
    """

    name: ClassVar[str] = "fixed_bps"

    def __init__(self, spread_bps: Basis = 2.0, option_spread_bps: Basis = 100.0) -> None:
        self._spread_bps = _require_non_negative(spread_bps, "spread_bps", type(self).__name__)
        self._option_spread_bps = _require_non_negative(
            option_spread_bps, "option_spread_bps", type(self).__name__
        )

    def half_spread(self, price: Price, instrument: Instrument) -> Price:
        bps = (
            self._option_spread_bps
            if instrument.asset_class is AssetClass.OPTION
            else self._spread_bps
        )
        return abs(price) * bps * _BPS * 0.5

    def quote_for(self, bar: Bar, instrument: Instrument, at: Price | None = None) -> Quote:
        mid = bar.close if at is None else at
        return self._centred_quote(mid, self.half_spread(mid, instrument), instrument)


class TickSpreadModel(SpreadModel):
    """Spread of N ticks. Appropriate for low-priced instruments where a bps
    spread would round to less than one tick."""

    name: ClassVar[str] = "ticks"

    def __init__(self, ticks: float = 1.0) -> None:
        self._ticks = _require_non_negative(ticks, "ticks", type(self).__name__)

    def half_spread(self, price: Price, instrument: Instrument) -> Price:
        return self._ticks * instrument.tick_size * 0.5

    def quote_for(self, bar: Bar, instrument: Instrument, at: Price | None = None) -> Quote:
        mid = bar.close if at is None else at
        return self._centred_quote(mid, self.half_spread(mid, instrument), instrument)


class VolatilitySpreadModel(SpreadModel):
    """Spread scaled by recent realised range — wider in stressed markets.

    Closer to reality than a fixed spread: liquidity evaporates exactly when
    strategies most want to trade, and a constant-cost assumption flatters
    high-turnover systems.
    """

    name: ClassVar[str] = "volatility"

    def __init__(self, base_bps: Basis = 2.0, range_multiplier: float = 0.1) -> None:
        self._base_bps = _require_non_negative(base_bps, "base_bps", type(self).__name__)
        self._range_multiplier = _require_non_negative(
            range_multiplier, "range_multiplier", type(self).__name__
        )

    def half_spread(self, price: Price, instrument: Instrument) -> Price:
        """The floor component only — the range term needs a bar.

        Callers holding a bar should use :meth:`quote_for`, which adds it. This
        exists so the model still answers the interface's question when all the
        caller has is a price.
        """
        return abs(price) * self._base_bps * _BPS * 0.5

    def quote_for(self, bar: Bar, instrument: Instrument, at: Price | None = None) -> Quote:
        mid = bar.close if at is None else at
        half = self.half_spread(mid, instrument) + self._range_multiplier * bar.range * 0.5
        return self._centred_quote(mid, half, instrument)


@dataclass(frozen=True, slots=True)
class PricingContext:
    """Everything the price model needs to resolve one fill."""

    instrument: Instrument
    side: OrderSide
    selection: PriceSelection
    bar: Bar | None = None
    option_quote: OptionQuote | None = None
    synthetic_quote: Quote | None = None
    #: The level the execution model settled on for this fill — the bar's open,
    #: its close, or a triggered limit/stop. A synthesised book is centred here
    #: rather than on the bar's mark, so the spread charged is the spread that
    #: prevailed where the trade actually happened. ``0.0`` means "not known",
    #: and the spread model falls back to the bar.
    reference_price: Price = 0.0


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

    def quote_used(self, context: PricingContext) -> Quote | None:
        """The book this model would price against, real or synthesised.

        The broker calls it once per fill, so it can count synthetic-quote fills
        and hand the same quote back through
        :attr:`PricingContext.synthetic_quote` rather than have the spread model
        rebuild it on every internal ``resolve``.
        """
        return None

    def spread_adjustment(self, context: PricingContext) -> Price:
        """Signed per-unit cost of transacting on ``selection``'s side of the book.

        Positive when buying and negative when selling under ``WORST``, so the
        broker can add it to whatever price the execution model derived from the
        bar. Expressed as a difference from the mid rather than as an absolute
        price, because the execution model — not this one — owns *where in the
        bar* the print happened.
        """
        if context.selection in _NEUTRAL_SELECTIONS:
            return 0.0
        mid = self.resolve(replace(context, selection=PriceSelection.MID))
        return self.resolve(context) - mid


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
        self._spread_model = spread_model
        self._require_real_quotes = require_real_quotes

    @property
    def spread_model(self) -> SpreadModel | None:
        return self._spread_model

    def supports(self, selection: PriceSelection) -> bool:
        return True

    def quote_used(self, context: PricingContext) -> Quote | None:
        if context.option_quote is not None:
            return context.option_quote.quote
        bar = context.bar
        if bar is None:
            return None
        if bar.quote is not None:
            return bar.quote
        if context.synthetic_quote is not None:
            return context.synthetic_quote
        if self._spread_model is None:
            return None
        at = context.reference_price if context.reference_price > 0.0 else None
        return self._spread_model.quote_for(bar, context.instrument, at)

    def resolve(self, context: PricingContext) -> Price:
        quote = self.quote_used(context)
        if self._require_real_quotes and (quote is None or quote.is_synthetic):
            raise ExecutionError(
                "require_real_quotes is set but this fill would price off a "
                f"{'synthetic' if quote is not None else 'missing'} book. Load a "
                "quote-carrying feed (DataRequest.include_quotes) or clear the "
                "flag to accept a modelled spread.",
                instrument_id=context.instrument.instrument_id,
                selection=context.selection.value,
            )
        if quote is None:
            # No book at all and no spread model: every selection collapses to
            # the bar's own price. That is the case the synthetic spread exists
            # to prevent, and the engine warns about it once per run.
            return self._bar_price(context)
        if context.selection is PriceSelection.LAST:
            return self._bar_price(context)
        if context.selection is PriceSelection.MID:
            return quote.mid
        is_buy = context.side is OrderSide.BUY
        if context.selection is PriceSelection.WORST:
            return quote.ask if is_buy else quote.bid
        return quote.bid if is_buy else quote.ask

    def _bar_price(self, context: PricingContext) -> Price:
        """The instrument's own last print, whichever form the feed took."""
        if context.option_quote is not None:
            return context.option_quote.price_for(
                PriceSelection.LAST, context.side is OrderSide.BUY
            )
        if context.bar is not None:
            return context.bar.close
        raise ExecutionError(
            "Cannot price a fill with neither a bar nor an option quote; the "
            "instrument did not trade at this step and the order should have "
            "been left working instead.",
            instrument_id=context.instrument.instrument_id,
        )


class OhlcFillPriceModel(FillPriceModel):
    """Prices from a chosen OHLC field, ignoring the spread entirely.

    Only valid with :attr:`~sigmaloop.types.PriceSelection.LAST`. Fast and
    familiar, but optimistic; the engine records a warning when it is used with
    a strategy that trades frequently.

    Because it charges no spread, :meth:`spread_adjustment` is always zero and
    the fill prints wherever the execution model put it. :attr:`field` therefore
    governs only the case where the execution model leaves the price open —
    every model shipped in :mod:`sigmaloop.execution.models` names one.
    """

    name: ClassVar[str] = "ohlc"

    def __init__(self, field: str = "open") -> None:
        if field not in _OHLC_FIELDS:
            raise ValidationError(
                f"Unknown OHLC field {field!r}; expected one of {_OHLC_FIELDS}.",
                field=field,
            )
        self._field = field

    @property
    def field(self) -> str:
        return self._field

    def supports(self, selection: PriceSelection) -> bool:
        return selection is PriceSelection.LAST

    def resolve(self, context: PricingContext) -> Price:
        if not self.supports(context.selection):
            raise ExecutionError(
                f"{type(self).__name__} prices from the {self._field!r} field and "
                f"knows nothing about the spread, so it cannot honour "
                f"{context.selection.value!r}. Use PriceSelection.LAST, or switch "
                "to QuoteFillPriceModel to transact on a side of the book.",
                instrument_id=context.instrument.instrument_id,
                selection=context.selection.value,
            )
        bar = context.bar
        if bar is None:
            raise ExecutionError(
                "OhlcFillPriceModel needs a bar; this step has none for the "
                "instrument, so there is no OHLC field to price from.",
                instrument_id=context.instrument.instrument_id,
            )
        value = getattr(bar, self._field)
        if value is None:
            # vwap is optional on the feed. Falling back to the close is the
            # nearest honest print rather than a fabricated one.
            return bar.close
        return float(value)

    def spread_adjustment(self, context: PricingContext) -> Price:
        return 0.0
