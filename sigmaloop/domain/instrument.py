"""Tradeable instrument definitions.

Instruments are immutable, hashable value objects created once by the data
layer and shared by reference everywhere else (``slots=True`` keeps them small;
a portfolio-mode run may hold hundreds of thousands of option contracts).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import ClassVar

from sigmaloop.errors import InstrumentNotFoundError, ValidationError
from sigmaloop.types import (
    AssetClass,
    Currency,
    InstrumentId,
    OptionRight,
    OptionStyle,
    Price,
    Quantity,
    SettlementType,
    Symbol,
    UtcDatetime,
)

__all__ = [
    "Equity",
    "Instrument",
    "InstrumentRegistry",
    "OptionContract",
]

#: Decimal places kept when snapping to a tick / lot. Ten is far beyond any
#: real tick size and well inside float64's ~15 significant digits.
_PRICE_DECIMALS = 10
_QUANTITY_DECIMALS = 10
#: Absorbs the representation error in e.g. 0.3 / 0.1 == 2.9999999999999996,
#: which would otherwise floor a clean 3 lots down to 2.
_LOT_EPSILON = 1e-9

#: Fixed-width tail of an OCC option symbol: ``YYMMDD`` + right + 8-digit strike.
#: Everything before it is the root, however it has been padded.
_OCC_TAIL_WIDTH = 15

_OCC_RIGHTS: dict[str, OptionRight] = {"C": OptionRight.CALL, "P": OptionRight.PUT}


@dataclass(frozen=True, slots=True, kw_only=True)
class Instrument(ABC):
    """Base for anything the engine can hold or trade.

    Attributes
    ----------
    instrument_id:
        Canonical unique key; see :data:`sigmaloop.types.InstrumentId`.
    symbol:
        Display ticker.
    asset_class:
        Drives margin, settlement and multiplier semantics.
    multiplier:
        Contract size. ``1`` for shares, ``100`` for standard equity options.
        Notional = ``price * quantity * multiplier``.
    tick_size:
        Minimum price increment; fill prices are rounded to it.
    lot_size:
        Minimum tradeable quantity increment. ``1.0`` for whole shares,
        ``<1`` when ``ExecutionConfig.allow_fractional`` is set.
    is_shortable:
        If False, a sell that would open/increase a short is rejected with
        :attr:`~sigmaloop.types.RejectReason.NOT_SHORTABLE`.
    borrow_rate_annual:
        Annualised short-borrow cost, charged per bar on short exposure.
    """

    instrument_id: InstrumentId
    symbol: Symbol
    asset_class: AssetClass
    currency: Currency = Currency.USD
    exchange: str | None = None
    multiplier: float = 1.0
    tick_size: Price = 0.01
    lot_size: Quantity = 1.0
    is_tradeable: bool = True
    is_shortable: bool = True
    borrow_rate_annual: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict, compare=False)

    #: Prefix used when building ``instrument_id`` for this class.
    ID_PREFIX: ClassVar[str] = "INS"

    def __post_init__(self) -> None:
        """Validate invariants (positive multiplier/tick, non-empty symbol)."""
        if not self.symbol or not self.symbol.strip():
            raise ValidationError("Instrument.symbol must be a non-empty ticker.")
        if not self.instrument_id:
            raise ValidationError("Instrument.instrument_id must be non-empty.", symbol=self.symbol)
        if self.multiplier <= 0:
            raise ValidationError(
                "Instrument.multiplier must be positive.",
                symbol=self.symbol,
                multiplier=self.multiplier,
            )
        if self.tick_size <= 0:
            raise ValidationError(
                "Instrument.tick_size must be positive.",
                symbol=self.symbol,
                tick_size=self.tick_size,
            )
        if self.lot_size <= 0:
            raise ValidationError(
                "Instrument.lot_size must be positive.",
                symbol=self.symbol,
                lot_size=self.lot_size,
            )

    @abstractmethod
    def notional(self, price: Price, quantity: Quantity) -> float:
        """Cash value of ``quantity`` units at ``price``, multiplier-adjusted."""
        raise NotImplementedError

    @abstractmethod
    def is_expired(self, as_of: UtcDatetime) -> bool:
        """True if the instrument no longer trades at ``as_of``."""
        raise NotImplementedError

    def round_price(self, price: Price) -> Price:
        """Snap ``price`` to the nearest valid tick, halves away from zero.

        Not :func:`round`, which is banker's rounding: on a 0.01 grid that sends
        0.125 down to 0.12 but 0.375 up to 0.38, so a price series landing on
        half-ticks alternates direction for no reason a reader could predict.
        See :func:`sigmaloop.utils.money.round_money` for the same argument.
        """
        ticks = math.floor(abs(price) / self.tick_size + 0.5)
        # Re-round the product: tick sizes such as 0.01 are not exactly
        # representable, so ticks * tick_size drifts into the 1e-17 range.
        magnitude = round(ticks * self.tick_size, _PRICE_DECIMALS)
        return -magnitude if price < 0 else magnitude

    def round_quantity(self, quantity: Quantity) -> Quantity:
        """Floor ``quantity`` (toward zero) to a valid lot multiple."""
        lots = math.floor(abs(quantity) / self.lot_size + _LOT_EPSILON)
        magnitude = round(lots * self.lot_size, _QUANTITY_DECIMALS)
        # `magnitude or 0.0` rather than a bare negation: rounding a small short
        # down to nothing would otherwise yield -0.0, which prints as "-0" in
        # every report and compares equal to 0.0 so no test would catch it.
        return -magnitude if quantity < 0 and magnitude else magnitude


@dataclass(frozen=True, slots=True, kw_only=True)
class Equity(Instrument):
    """A share, ETF or index. ``multiplier`` is always 1."""

    asset_class: AssetClass = AssetClass.EQUITY
    sector: str | None = None
    industry: str | None = None
    listed_on: date | None = None
    delisted_on: date | None = None

    ID_PREFIX: ClassVar[str] = "EQ"

    @classmethod
    def make_id(cls, symbol: Symbol) -> InstrumentId:
        """Build ``"EQ:<SYMBOL>"``."""
        return InstrumentId(f"{cls.ID_PREFIX}:{symbol.strip().upper()}")

    def notional(self, price: Price, quantity: Quantity) -> float:
        return price * quantity * self.multiplier

    def is_expired(self, as_of: UtcDatetime) -> bool:
        return self.delisted_on is not None and as_of.date() >= self.delisted_on


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionContract(Instrument):
    """A single listed option series.

    The engine never synthesises option prices; contracts must be backed by an
    :class:`~sigmaloop.data.provider.OptionsDataProvider`. Greeks live on the
    market-data side (:class:`~sigmaloop.domain.bar.OptionQuote`), not here,
    because they are time-varying.
    """

    asset_class: AssetClass = AssetClass.OPTION
    underlying_id: InstrumentId
    underlying_symbol: Symbol
    right: OptionRight
    strike: Price
    expiry: date
    style: OptionStyle = OptionStyle.AMERICAN
    settlement: SettlementType = SettlementType.PHYSICAL
    multiplier: float = 100.0
    occ_symbol: str | None = None

    ID_PREFIX: ClassVar[str] = "OPT"

    def __post_init__(self) -> None:
        """Validate the base invariants, then the strike.

        A non-positive strike is not merely odd data: :meth:`moneyness` divides
        by it for puts, and an exercise at a zero strike would hand the account
        free shares. Both failures are silent, so the contract refuses to exist.

        The base is named explicitly rather than via ``super()``: ``slots=True``
        rebuilds the dataclass as a new class object, leaving the zero-argument
        form's ``__class__`` cell pointing at the pre-slots class it was
        compiled against, which raises at construction time.
        """
        Instrument.__post_init__(self)
        if not (math.isfinite(self.strike) and self.strike > 0.0):
            raise ValidationError(
                "OptionContract.strike must be a positive, finite price.",
                symbol=self.symbol,
                strike=self.strike,
            )

    @classmethod
    def make_id(
        cls, underlying: Symbol, expiry: date, right: OptionRight, strike: Price
    ) -> InstrumentId:
        """Build ``"OPT:SPY:20250117:C:00500000"`` (strike scaled by 1000)."""
        strike_thousandths = round(strike * 1000)
        return InstrumentId(
            f"{cls.ID_PREFIX}:{underlying.strip().upper()}:{expiry:%Y%m%d}:"
            f"{'C' if right is OptionRight.CALL else 'P'}:{strike_thousandths:08d}"
        )

    @classmethod
    def from_occ(cls, occ_symbol: str, **overrides: object) -> OptionContract:
        """Parse an OCC symbol — ``"SPY   250117C00500000"`` — into a contract.

        The inverse of :meth:`make_id`, and the shape every US options feed and
        broker file names contracts in. The layout is fixed-width: 6 characters
        of root, ``YYMMDD``, ``C`` or ``P``, then the strike in thousandths over
        8 digits.

        The root is read from what is left after the fixed 15-character tail
        rather than from the first 6 columns, so a symbol whose padding has been
        stripped in transit still parses. Two-digit years map to 2000-2099: the
        format has no century and listed options do not predate it.

        ``overrides`` reach the constructor untouched, which is how a caller
        supplies the facts the symbol cannot carry — ``style``, ``settlement``,
        a non-standard ``multiplier`` after an adjustment.
        """
        cleaned = occ_symbol.strip()
        tail = cleaned[-_OCC_TAIL_WIDTH:]
        root = cleaned[:-_OCC_TAIL_WIDTH].strip().upper()
        if len(cleaned) <= _OCC_TAIL_WIDTH or not root:
            raise ValidationError(
                f"{occ_symbol!r} is not an OCC option symbol; expected a root "
                "followed by YYMMDD, C or P, and an 8-digit strike in thousandths.",
                occ_symbol=occ_symbol,
            )
        right_code = tail[6]
        if right_code not in _OCC_RIGHTS:
            raise ValidationError(
                f"OCC symbol {occ_symbol!r} has right {right_code!r}; expected 'C' or 'P'.",
                occ_symbol=occ_symbol,
            )
        try:
            expiry = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
            strike_thousandths = int(tail[7:])
        except ValueError as exc:
            raise ValidationError(
                f"OCC symbol {occ_symbol!r} carries an unparseable expiry or strike.",
                occ_symbol=occ_symbol,
            ) from exc

        underlying = Symbol(root)
        right = _OCC_RIGHTS[right_code]
        strike = strike_thousandths / 1000.0
        fields: dict[str, object] = {
            "instrument_id": cls.make_id(underlying, expiry, right, strike),
            "symbol": Symbol(cleaned),
            "underlying_id": Equity.make_id(underlying),
            "underlying_symbol": underlying,
            "right": right,
            "strike": strike,
            "expiry": expiry,
            "occ_symbol": cleaned,
        }
        fields.update(overrides)
        return cls(**fields)  # type: ignore[arg-type]

    def days_to_expiry(self, as_of: UtcDatetime) -> int:
        """Calendar days from ``as_of`` to :attr:`expiry` (0 == 0DTE).

        Negative once the contract is past expiry, rather than clamped at zero:
        a caller filtering on ``dte <= 0`` must be able to tell "expires today"
        from "expired last week".
        """
        return (self.expiry - as_of.date()).days

    def moneyness(self, underlying_price: Price) -> float:
        """``underlying / strike`` for calls, ``strike / underlying`` for puts.

        ``1.0`` is at-the-money and ``> 1.0`` is in-the-money for either right,
        which is what lets a selector express "10% ITM" without branching.
        """
        if self.right is OptionRight.CALL:
            return underlying_price / self.strike
        if underlying_price <= 0.0:
            # A worthless underlying makes every put infinitely in the money.
            # Reporting that keeps the degenerate case visible; a zero would
            # read as at-the-money and quietly select the wrong contract.
            return math.inf
        return self.strike / underlying_price

    def intrinsic_value(self, underlying_price: Price) -> Price:
        """Per-unit intrinsic value; 0 when out of the money."""
        if self.right is OptionRight.CALL:
            return max(underlying_price - self.strike, 0.0)
        return max(self.strike - underlying_price, 0.0)

    def is_itm(self, underlying_price: Price) -> bool:
        return self.intrinsic_value(underlying_price) > 0.0

    def notional(self, price: Price, quantity: Quantity) -> float:
        return price * quantity * self.multiplier

    def is_expired(self, as_of: UtcDatetime) -> bool:
        """False on the expiry date itself — the contract trades until its close."""
        return as_of.date() > self.expiry


class InstrumentRegistry:
    """Process-local interning table: ``InstrumentId -> Instrument``.

    One instance per run, owned by :class:`~sigmaloop.engine.context.RunContext`.
    Guarantees a single shared object per instrument so identity comparisons and
    dict lookups stay cheap, and so the portfolio can resolve multipliers without
    calling back into a provider.
    """

    __slots__ = ("_by_id", "_by_symbol", "_options_by_underlying")

    def __init__(self) -> None:
        self._by_id: dict[InstrumentId, Instrument] = {}
        self._by_symbol: dict[Symbol, list[Instrument]] = {}
        self._options_by_underlying: dict[InstrumentId, list[OptionContract]] = {}

    def register(self, instrument: Instrument) -> Instrument:
        """Insert, or return the already-interned equal instance.

        Two different instruments claiming one id is refused rather than
        overwritten: every position, order and fill in the run keys off that id,
        and silently rebinding it would reprice holdings the strategy already
        opened against the instrument it thought it had.
        """
        existing = self._by_id.get(instrument.instrument_id)
        if existing is not None:
            if existing == instrument:
                return existing
            raise ValidationError(
                "Two different instruments claim the same instrument_id.",
                instrument_id=instrument.instrument_id,
                existing=existing.symbol,
                incoming=instrument.symbol,
            )
        self._by_id[instrument.instrument_id] = instrument
        self._by_symbol.setdefault(instrument.symbol, []).append(instrument)
        if isinstance(instrument, OptionContract):
            self._options_by_underlying.setdefault(instrument.underlying_id, []).append(instrument)
        return instrument

    def get(self, instrument_id: InstrumentId) -> Instrument:
        """Look up by id; raises ``InstrumentNotFoundError`` if absent."""
        found = self._by_id.get(instrument_id)
        if found is None:
            raise InstrumentNotFoundError(
                f"No instrument registered under {instrument_id!r}. Instruments are "
                "interned by the data layer as they are loaded; an id that reaches "
                "the engine unregistered is a strategy referencing a symbol the run "
                "never subscribed to.",
                instrument_id=instrument_id,
                registered=len(self._by_id),
            )
        return found

    def try_get(self, instrument_id: InstrumentId) -> Instrument | None:
        return self._by_id.get(instrument_id)

    def by_symbol(self, symbol: Symbol) -> tuple[Instrument, ...]:
        """All instruments sharing a ticker (the equity plus its options)."""
        return tuple(self._by_symbol.get(symbol, ()))

    def options_on(self, underlying_id: InstrumentId) -> tuple[OptionContract, ...]:
        return tuple(self._options_by_underlying.get(underlying_id, ()))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self._by_id
