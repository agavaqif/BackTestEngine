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

from sigmaloop.errors import ValidationError
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
        """Snap ``price`` to the nearest valid tick."""
        ticks = round(price / self.tick_size)
        # Re-round the product: tick sizes such as 0.01 are not exactly
        # representable, so ticks * tick_size drifts into the 1e-17 range.
        return round(ticks * self.tick_size, _PRICE_DECIMALS)

    def round_quantity(self, quantity: Quantity) -> Quantity:
        """Floor ``quantity`` (toward zero) to a valid lot multiple."""
        lots = math.floor(abs(quantity) / self.lot_size + _LOT_EPSILON)
        magnitude = round(lots * self.lot_size, _QUANTITY_DECIMALS)
        return -magnitude if quantity < 0 else magnitude


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

    @classmethod
    def make_id(
        cls, underlying: Symbol, expiry: date, right: OptionRight, strike: Price
    ) -> InstrumentId:
        """Build ``"OPT:SPY:20250117:C:00500000"`` (strike scaled by 1000)."""
        raise NotImplementedError

    @classmethod
    def from_occ(cls, occ_symbol: str, **overrides: object) -> OptionContract:
        """Parse a 21-character OCC symbol into a contract."""
        raise NotImplementedError

    def days_to_expiry(self, as_of: UtcDatetime) -> int:
        """Calendar days from ``as_of`` to :attr:`expiry` (0 == 0DTE)."""
        raise NotImplementedError

    def moneyness(self, underlying_price: Price) -> float:
        """``underlying / strike`` for calls, ``strike / underlying`` for puts."""
        raise NotImplementedError

    def intrinsic_value(self, underlying_price: Price) -> Price:
        """Per-unit intrinsic value; 0 when out of the money."""
        raise NotImplementedError

    def is_itm(self, underlying_price: Price) -> bool:
        raise NotImplementedError

    def notional(self, price: Price, quantity: Quantity) -> float:
        raise NotImplementedError

    def is_expired(self, as_of: UtcDatetime) -> bool:
        raise NotImplementedError


class InstrumentRegistry:
    """Process-local interning table: ``InstrumentId -> Instrument``.

    One instance per run, owned by :class:`~sigmaloop.engine.context.RunContext`.
    Guarantees a single shared object per instrument so identity comparisons and
    dict lookups stay cheap, and so the portfolio can resolve multipliers without
    calling back into a provider.
    """

    __slots__ = ("_by_id", "_by_symbol", "_options_by_underlying")

    def __init__(self) -> None:
        raise NotImplementedError

    def register(self, instrument: Instrument) -> Instrument:
        """Insert, or return the already-interned equal instance."""
        raise NotImplementedError

    def get(self, instrument_id: InstrumentId) -> Instrument:
        """Look up by id; raises ``InstrumentNotFoundError`` if absent."""
        raise NotImplementedError

    def try_get(self, instrument_id: InstrumentId) -> Instrument | None:
        raise NotImplementedError

    def by_symbol(self, symbol: Symbol) -> tuple[Instrument, ...]:
        """All instruments sharing a ticker (the equity plus its options)."""
        raise NotImplementedError

    def options_on(self, underlying_id: InstrumentId) -> tuple[OptionContract, ...]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, instrument_id: object) -> bool:
        raise NotImplementedError
