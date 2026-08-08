"""Open positions, tax lots and closed round-trip trades."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import Fill
from sigmaloop.errors import ValidationError
from sigmaloop.types import (
    AssetClass,
    FillId,
    InstrumentId,
    Money,
    OptionRight,
    Percent,
    PositionSide,
    Price,
    Quantity,
    Symbol,
    TradeCloseReason,
    TradeId,
    UtcDatetime,
)
from sigmaloop.utils.timeutil import ensure_utc

__all__ = ["INITIAL_RISK_KEY", "Lot", "OptionTrade", "Position", "Trade"]

#: Quantities at or below this are float dust left by repeated add/reduce
#: arithmetic, not a holding. Comfortably under any tradeable lot, including the
#: fractional sizes the sizers emit when ``allow_fractional_shares`` is set.
_FLAT_TOLERANCE: Quantity = 1e-9

#: ``Trade.metadata`` key holding the cash a trade risked at entry (positive).
#: The portfolio writes it when the entry carried a stop, because only the
#: portfolio knows the stop level and the contract multiplier;
#: :attr:`Trade.r_multiple` reads it back.
INITIAL_RISK_KEY = "initial_risk"


@dataclass(slots=True)
class Lot:
    """One acquisition tranche within a position.

    Lots let the accounting layer compute realised P&L under an explicit
    matching policy (FIFO by default, configurable via
    ``AccountingConfig.lot_matching``) instead of only average cost, which
    matters for partial exits and for options roll analysis.
    """

    #: Signed like :attr:`Position.quantity` — negative for a short tranche — so
    #: a lot read on its own still says which way it was opened.
    quantity: Quantity
    price: Price
    opened_at: UtcDatetime
    fill_id: FillId
    commission: Money = 0.0


@dataclass(slots=True)
class Position:
    """Live holding in one instrument. Mutated in place by the portfolio.

    :attr:`quantity` is signed: positive long, negative short, zero means the
    position object is retained only until end-of-bar cleanup.
    """

    instrument: Instrument
    quantity: Quantity = 0.0
    avg_price: Price = 0.0
    realized_pnl: Money = 0.0
    commission_paid: Money = 0.0
    fees_paid: Money = 0.0
    borrow_cost_paid: Money = 0.0
    dividends_received: Money = 0.0
    mark_price: Price = 0.0
    last_update: UtcDatetime | None = None
    opened_at: UtcDatetime | None = None
    lots: deque[Lot] = field(default_factory=deque)
    #: Running extremes since entry, used for MAE/MFE on the closed trade.
    max_favorable_price: Price = 0.0
    max_adverse_price: Price = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def instrument_id(self) -> InstrumentId:
        return self.instrument.instrument_id

    @property
    def side(self) -> PositionSide:
        if self.quantity > _FLAT_TOLERANCE:
            return PositionSide.LONG
        if self.quantity < -_FLAT_TOLERANCE:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > _FLAT_TOLERANCE

    @property
    def is_short(self) -> bool:
        return self.quantity < -_FLAT_TOLERANCE

    @property
    def cost_basis(self) -> Money:
        """``|quantity| * avg_price * multiplier``."""
        return abs(self.quantity) * self.avg_price * self.instrument.multiplier

    @property
    def market_value(self) -> Money:
        """Signed mark-to-market value: ``quantity * mark_price * multiplier``."""
        return self.quantity * self.mark_price * self.instrument.multiplier

    @property
    def unrealized_pnl(self) -> Money:
        return (self.mark_price - self.avg_price) * self.quantity * self.instrument.multiplier

    @property
    def unrealized_pnl_pct(self) -> Percent:
        basis = self.cost_basis
        return 0.0 if basis == 0.0 else self.unrealized_pnl / basis

    @property
    def total_pnl(self) -> Money:
        """Realised + unrealised, net of commissions, fees and borrow.

        Dividends are added rather than netted: they are carry the position
        genuinely earned, and dropping them would make a held-through-ex-date
        long look worse than it was.
        """
        return (
            self.realized_pnl
            + self.unrealized_pnl
            + self.dividends_received
            - self.commission_paid
            - self.fees_paid
            - self.borrow_cost_paid
        )

    @property
    def exposure(self) -> Money:
        """Absolute notional at risk — always positive, shorts included."""
        return abs(self.market_value)

    def apply_fill(self, fill: Fill) -> Money:
        """Fold a fill in and return the realised P&L it produced.

        Handles the four cases: open, increase, reduce (realises), and
        flip-through-zero (realises the whole old side, opens the new one).
        """
        if fill.instrument_id != self.instrument_id:
            raise ValidationError(
                "Fill is for a different instrument than this position; booking it "
                "here would realise one symbol's P&L against another's cost basis, "
                "and the ledger would still balance while both rows were wrong.",
                instrument_id=self.instrument_id,
                fill_instrument_id=fill.instrument_id,
                fill_id=fill.fill_id,
            )
        if fill.quantity <= 0.0:
            raise ValidationError(
                "Position.apply_fill needs a positive fill quantity; direction "
                "comes from fill.side, not from the sign of the size.",
                instrument_id=self.instrument_id,
                fill_id=fill.fill_id,
                quantity=fill.quantity,
            )

        self.commission_paid += fill.commission
        self.fees_paid += fill.fees
        self.last_update = fill.timestamp
        # A fill is the freshest print there is, so it doubles as a mark. Without
        # this, market_value and exposure read zero between the opening fill and
        # the next mark_to_market, and every risk check in between sees no risk.
        self.mark_price = fill.price

        signed = fill.side.sign * fill.quantity
        current = self.quantity
        realized = 0.0

        if abs(current) <= _FLAT_TOLERANCE:
            # Flat (possibly with dust from a previous exit): start clean rather
            # than averaging the new entry into a residue.
            self.quantity = 0.0
            self._reset_entry()
            self._open_or_increase(signed, fill)
        elif (current > 0.0) == (signed > 0.0):
            self._open_or_increase(signed, fill)
        else:
            closing = min(fill.quantity, abs(current))
            realized = self._realise(closing, fill.price)
            self.realized_pnl += realized
            self.quantity = current + signed
            overshoot = fill.quantity - closing
            if overshoot > _FLAT_TOLERANCE:
                # Flip: the old side is fully realised above; the remainder is a
                # new entry at this fill's price, with its own lot and extremes.
                self.quantity = 0.0
                self._reset_entry()
                self._open_or_increase(math.copysign(overshoot, signed), fill)
            elif not self.is_open:
                self.quantity = 0.0
                self._reset_entry()
            else:
                # A partial exit consumed the oldest lots, so the cost of what is
                # left is no longer the average of what was bought.
                self.avg_price = self._remaining_avg_price()

        # A fill is a print the position genuinely traded through, so it counts
        # towards the excursions exactly as a mark does. Without this, scaling
        # out into a spike leaves MFE reading the last quiet mark and the closed
        # trade understates how far in front it ever got.
        if self.is_open:
            self._track_excursion(fill.price)
        return realized

    def mark(self, price: Price, at: UtcDatetime) -> None:
        """Update the mark price and the MAE/MFE extremes."""
        stamp = ensure_utc(at)
        self.mark_price = price
        self.last_update = stamp
        if not self.is_open:
            return
        if self.opened_at is None:
            # Seeded directly rather than built from fills (restored state): this
            # mark is the earliest entry reference there is.
            self.opened_at = stamp
            self.max_favorable_price = price
            self.max_adverse_price = price
            return
        self._track_excursion(price)

    def accrue_borrow(self, at: UtcDatetime, bar_fraction_of_year: float) -> Money:
        """Charge short-borrow cost for one bar; returns the amount charged."""
        if not self.is_short or bar_fraction_of_year <= 0.0:
            return 0.0
        rate = self.instrument.borrow_rate_annual
        if rate <= 0.0:
            return 0.0
        # Charged on current notional, not entry notional: a short that has moved
        # against you costs more to keep, which is exactly the squeeze dynamic.
        charge = self.exposure * rate * bar_fraction_of_year
        self.borrow_cost_paid += charge
        self.last_update = ensure_utc(at)
        return charge

    # ---- internals -------------------------------------------------------- #

    def _track_excursion(self, price: Price) -> None:
        """Fold ``price`` into the running MAE/MFE extremes.

        Keyed off :attr:`opened_at` rather than off the extremes being zero: an
        option marked worthless is legitimately at 0.0, and a sentinel that
        cannot tell that apart from "never set" would discard the very extreme
        the trade turned on.
        """
        if self.is_short:
            # The extremes invert with the side: for a short, down is favourable.
            self.max_favorable_price = min(self.max_favorable_price, price)
            self.max_adverse_price = max(self.max_adverse_price, price)
        else:
            self.max_favorable_price = max(self.max_favorable_price, price)
            self.max_adverse_price = min(self.max_adverse_price, price)

    def _open_or_increase(self, delta: Quantity, fill: Fill) -> None:
        """Add ``delta`` units in the position's own direction as a fresh lot."""
        prior = abs(self.quantity)
        total = prior + abs(delta)
        self.avg_price = (self.avg_price * prior + fill.price * abs(delta)) / total
        self.quantity += delta
        self.lots.append(
            Lot(
                quantity=delta,
                price=fill.price,
                opened_at=fill.timestamp,
                fill_id=fill.fill_id,
                commission=fill.commission,
            )
        )
        if self.opened_at is None:
            self.opened_at = fill.timestamp
            self.max_favorable_price = fill.price
            self.max_adverse_price = fill.price

    def _realise(self, quantity: Quantity, exit_price: Price) -> Money:
        """Consume ``quantity`` units from the oldest lots; return their P&L.

        FIFO. :attr:`lots` is ordered oldest-first, so the LIFO policy offered by
        ``AccountingConfig.lot_matching`` consumes the same deque from the other
        end; choosing between them belongs to the accounting layer that owns the
        config, not to the position.
        """
        multiplier = self.instrument.multiplier
        outstanding = quantity
        realized = 0.0
        while outstanding > _FLAT_TOLERANCE and self.lots:
            lot = self.lots[0]
            take = min(outstanding, abs(lot.quantity))
            direction = 1.0 if lot.quantity > 0.0 else -1.0
            realized += (exit_price - lot.price) * take * direction * multiplier
            outstanding -= take
            if abs(lot.quantity) - take <= _FLAT_TOLERANCE:
                self.lots.popleft()
            else:
                lot.quantity -= math.copysign(take, lot.quantity)
        if outstanding > _FLAT_TOLERANCE:
            # No lot history left — a position seeded directly rather than built
            # from fills. Fall back to average cost so the P&L is still real.
            direction = 1.0 if self.quantity > 0.0 else -1.0
            realized += (exit_price - self.avg_price) * outstanding * direction * multiplier
        return realized

    def _remaining_avg_price(self) -> Price:
        total = sum(abs(lot.quantity) for lot in self.lots)
        if total <= _FLAT_TOLERANCE:
            return self.avg_price
        return sum(abs(lot.quantity) * lot.price for lot in self.lots) / total

    def _reset_entry(self) -> None:
        """Clear the entry-describing state, keeping the cumulative counters.

        Run when the position goes flat or flips. The portfolio builds the closed
        :class:`Trade` from the entry price, ``opened_at`` and the excursion
        extremes *before* it applies the closing fill, so nothing is lost here —
        whereas leaving them in place would silently carry one round trip's entry
        into the next re-entry on the same instrument.
        """
        self.avg_price = 0.0
        self.opened_at = None
        self.lots.clear()
        self.max_favorable_price = 0.0
        self.max_adverse_price = 0.0


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip — the row type of the trade log.

    Produced by the portfolio when a position's exposure returns to (or crosses)
    zero. Entry figures are the lot-weighted averages of the opening fills.
    """

    trade_id: TradeId
    instrument_id: InstrumentId
    symbol: Symbol
    asset_class: AssetClass
    direction: PositionSide
    quantity: Quantity
    entry_time: UtcDatetime
    entry_price: Price
    exit_time: UtcDatetime
    exit_price: Price
    gross_pnl: Money
    commission: Money
    fees: Money
    net_pnl: Money
    return_pct: Percent
    close_reason: TradeCloseReason
    #: Max adverse / favourable excursion in cash while the trade was open.
    mae: Money = 0.0
    mfe: Money = 0.0
    bars_held: int = 0
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise both ends of the round trip to tz-aware UTC.

        :attr:`holding_period` subtracts one from the other, and mixing a naive
        stamp with an aware one raises deep inside a metric rather than here.
        """
        object.__setattr__(self, "entry_time", ensure_utc(self.entry_time))
        object.__setattr__(self, "exit_time", ensure_utc(self.exit_time))

    @property
    def holding_period(self) -> timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        """Net, not gross: a trade that made money and gave it all to the broker
        was not a win."""
        return self.net_pnl > 0.0

    @property
    def r_multiple(self) -> float | None:
        """Net P&L divided by initial risk, when a stop distance was recorded.

        Reads :data:`INITIAL_RISK_KEY` from :attr:`metadata`; ``None`` when the
        entry carried no stop, because "1R" is meaningless without one and a
        fabricated denominator would quietly corrupt the expectancy metric.
        """
        recorded = self.metadata.get(INITIAL_RISK_KEY)
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
            return None
        risk = abs(float(recorded))
        return None if risk == 0.0 else self.net_pnl / risk


@dataclass(frozen=True, slots=True)
class OptionTrade(Trade):
    """Round trip on an option contract — the options trade log row.

    Adds contract identity and the entry/exit market context an options
    post-mortem needs (was it delta, theta or the underlying that paid?).
    """

    # RUF009 guards against expensive or shared defaults; Symbol is a NewType,
    # so this is the identity function on a str literal and costs nothing.
    underlying_symbol: Symbol = Symbol("")  # noqa: RUF009
    right: OptionRight = OptionRight.CALL
    strike: Price = 0.0
    expiry: date | None = None
    multiplier: float = 100.0
    dte_at_entry: int = 0
    dte_at_exit: int = 0
    delta_at_entry: float | None = None
    iv_at_entry: float | None = None
    iv_at_exit: float | None = None
    underlying_price_at_entry: Price | None = None
    underlying_price_at_exit: Price | None = None
    was_assigned: bool = False
    was_exercised: bool = False
    expired_worthless: bool = False

    @property
    def premium_collected(self) -> Money:
        """Positive for short-premium trades, negative for long.

        The entry cash flow, signed from the account's point of view: selling a
        contract credits the premium, buying one debits it.
        """
        premium = self.entry_price * abs(self.quantity) * self.multiplier
        return premium if self.direction is PositionSide.SHORT else -premium

    @property
    def underlying_return_pct(self) -> Percent | None:
        """What the underlying did over the life of the trade.

        ``None`` when the feed did not carry the underlying's price, which is the
        honest answer: a zero would read as "the underlying went nowhere" and
        credit the whole result to theta.
        """
        entry = self.underlying_price_at_entry
        exit_price = self.underlying_price_at_exit
        if entry is None or exit_price is None or entry == 0.0:
            return None
        return (exit_price - entry) / entry
