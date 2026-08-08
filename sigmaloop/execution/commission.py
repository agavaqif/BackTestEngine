"""Commission and fee models (Execution requirement 3).

Commissions are configurable and composable: a realistic US equity/options
account pays a broker commission plus regulatory fees (SEC, TAF, OCC, ORF),
each with its own base and its own side-dependence.
:class:`CompositeCommissionModel` sums them so no single model has to encode
an entire fee schedule.

Nothing here rounds to the cent. Regulatory fees are quoted in fractions of a
basis point and a per-fill rounding would zero them out; the report rounds once,
at the boundary, via :func:`~sigmaloop.utils.money.round_money`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.instrument import Instrument
from sigmaloop.errors import ValidationError
from sigmaloop.types import AssetClass, Money, OrderSide, Percent, Price, Quantity
from sigmaloop.utils.money import notional as notional_value

__all__ = [
    "CommissionContext",
    "CommissionModel",
    "ZeroCommissionModel",
    "PerShareCommissionModel",
    "PerTradeCommissionModel",
    "PercentValueCommissionModel",
    "PerContractCommissionModel",
    "RegulatoryFeeModel",
    "TieredCommissionModel",
    "CompositeCommissionModel",
]


#: Quantities at or below this are float dust, not a fill.
_QUANTITY_TOLERANCE: Quantity = 1e-9


def _require_non_negative(value: float, name: str, owner: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError(
            f"{owner}.{name} must be a non-negative, finite amount; a negative "
            "commission would pay the account to trade.",
            **{name: value},
        )
    return value


def _incremental(
    cumulative: Callable[[Quantity], Money], filled_before: Quantity, quantity: Quantity
) -> Money:
    """This fill's share of a schedule defined on the ORDER's running total.

    Per-order floors, caps and flat fees belong to the order, not to whichever
    print happened to complete it. Billing each fill in isolation charges a $1
    minimum four times on an order a liquidity cap split four ways, and lets a
    capped fee be collected once per slice. Charging the cumulative schedule and
    subtracting what earlier fills already paid makes a split order cost exactly
    what the same order would have cost filling in one print.
    """
    return max(cumulative(filled_before + quantity) - cumulative(filled_before), 0.0)


@dataclass(frozen=True, slots=True)
class CommissionContext:
    """Inputs for one commission calculation."""

    instrument: Instrument
    side: OrderSide
    quantity: Quantity
    price: Price
    #: Cumulative shares/contracts traded this month, for tiered schedules.
    period_volume: Quantity = 0.0
    is_closing: bool = False
    #: How much of *this order* had already filled before this fill. Per-order
    #: floors, caps and flat fees are charged against the running total, so a
    #: partially filled order is not billed as several whole ones.
    filled_before: Quantity = 0.0

    @property
    def notional(self) -> Money:
        """Traded value, contract multiplier included."""
        return notional_value(self.price, abs(self.quantity), self.instrument.multiplier)

    @property
    def unit_notional(self) -> Money:
        """Traded value of a single share or contract.

        Per-order caps are a fraction of the *order's* value, which spans fills
        that printed at their own prices. This fill's price stands in for all of
        them: the broker hands over one print at a time, and the error is bounded
        by how far the order's prints spread — cents on a schedule already
        expressed in fractions of a percent.
        """
        return notional_value(self.price, 1.0, self.instrument.multiplier)

    @property
    def is_first_fill(self) -> bool:
        """True when no part of this order has printed yet."""
        return self.filled_before <= _QUANTITY_TOLERANCE


class CommissionModel(ABC):
    """Computes the cash cost of executing one fill."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def commission(self, context: CommissionContext) -> Money:
        """Broker commission, always >= 0."""
        raise NotImplementedError

    def fees(self, context: CommissionContext) -> Money:
        """Regulatory/exchange fees, separate from commission. Default 0."""
        return 0.0

    def total(self, context: CommissionContext) -> Money:
        return self.commission(context) + self.fees(context)


class ZeroCommissionModel(CommissionModel):
    """No costs. Useful as a baseline to isolate the impact of frictions."""

    name: ClassVar[str] = "zero"

    def commission(self, context: CommissionContext) -> Money:
        return 0.0


class PerShareCommissionModel(CommissionModel):
    """``rate * shares``, with optional per-order floor and cap."""

    name: ClassVar[str] = "per_share"

    def __init__(
        self,
        rate: Money = 0.005,
        minimum: Money = 1.0,
        maximum_pct_of_value: Percent | None = 0.01,
    ) -> None:
        self._rate = _require_non_negative(rate, "rate", type(self).__name__)
        self._minimum = _require_non_negative(minimum, "minimum", type(self).__name__)
        if maximum_pct_of_value is not None:
            _require_non_negative(maximum_pct_of_value, "maximum_pct_of_value", type(self).__name__)
        self._maximum_pct_of_value = maximum_pct_of_value

    def commission(self, context: CommissionContext) -> Money:
        """Floor and cap are per ORDER, so they run against the running total."""
        return _incremental(
            lambda filled: self._cumulative(filled, context.unit_notional),
            abs(context.filled_before),
            abs(context.quantity),
        )

    def _cumulative(self, quantity: Quantity, unit_notional: Money) -> Money:
        """Total owed once ``quantity`` units of the order have printed.

        The floor applies before the cap, as brokers apply them. Order matters on
        small orders: 10 shares of a $2 stock owes the $1 minimum on the rate, but
        the 1%-of-value cap takes it back to $0.20. Applying the cap first and
        then the floor would bill the minimum on every odd lot and overstate the
        cost of trading cheap names.
        """
        if quantity <= _QUANTITY_TOLERANCE:
            return 0.0
        charge = max(self._rate * quantity, self._minimum)
        if self._maximum_pct_of_value is not None:
            charge = min(charge, self._maximum_pct_of_value * unit_notional * quantity)
        return charge


class PerTradeCommissionModel(CommissionModel):
    """Flat fee per order, regardless of size."""

    name: ClassVar[str] = "per_trade"

    def __init__(self, fee: Money = 1.0) -> None:
        self._fee = _require_non_negative(fee, "fee", type(self).__name__)

    def commission(self, context: CommissionContext) -> Money:
        """Charged once, on the first print. An order a liquidity cap splits
        across four bars is still one order and owes one fee."""
        return self._fee if context.is_first_fill else 0.0


class PercentValueCommissionModel(CommissionModel):
    """A percentage of traded notional."""

    name: ClassVar[str] = "percent_value"

    def __init__(self, rate: Percent = 0.001, minimum: Money = 0.0) -> None:
        self._rate = _require_non_negative(rate, "rate", type(self).__name__)
        self._minimum = _require_non_negative(minimum, "minimum", type(self).__name__)

    def commission(self, context: CommissionContext) -> Money:
        return _incremental(
            lambda filled: (
                max(self._rate * context.unit_notional * filled, self._minimum)
                if filled > _QUANTITY_TOLERANCE
                else 0.0
            ),
            abs(context.filled_before),
            abs(context.quantity),
        )


class PerContractCommissionModel(CommissionModel):
    """Options: ``per_contract * contracts`` plus an optional per-order base."""

    name: ClassVar[str] = "per_contract"

    def __init__(
        self,
        per_contract: Money = 0.65,
        per_order: Money = 0.0,
        #: Many brokers waive commission to close a contract worth < $0.05.
        waive_close_below_price: Price | None = None,
    ) -> None:
        self._per_contract = _require_non_negative(
            per_contract, "per_contract", type(self).__name__
        )
        self._per_order = _require_non_negative(per_order, "per_order", type(self).__name__)
        if waive_close_below_price is not None:
            _require_non_negative(
                waive_close_below_price, "waive_close_below_price", type(self).__name__
            )
        self._waive_close_below_price = waive_close_below_price

    def commission(self, context: CommissionContext) -> Money:
        waive = self._waive_close_below_price
        if waive is not None and context.is_closing and context.price < waive:
            return 0.0
        base = self._per_order if context.is_first_fill else 0.0
        return self._per_contract * abs(context.quantity) + base


class RegulatoryFeeModel(CommissionModel):
    """US sell-side regulatory fees: SEC Section 31 and FINRA TAF.

    Charged on sells only, which is why fees are modelled separately from
    commission rather than folded into a single per-share rate.
    """

    name: ClassVar[str] = "regulatory"

    def __init__(
        self,
        sec_fee_rate: Percent = 0.0000278,
        finra_taf_per_share: Money = 0.000166,
        finra_taf_cap: Money = 8.30,
        options_orf_per_contract: Money = 0.0388,
    ) -> None:
        owner = type(self).__name__
        self._sec_fee_rate = _require_non_negative(sec_fee_rate, "sec_fee_rate", owner)
        self._finra_taf_per_share = _require_non_negative(
            finra_taf_per_share, "finra_taf_per_share", owner
        )
        self._finra_taf_cap = _require_non_negative(finra_taf_cap, "finra_taf_cap", owner)
        self._options_orf_per_contract = _require_non_negative(
            options_orf_per_contract, "options_orf_per_contract", owner
        )

    def commission(self, context: CommissionContext) -> Money:
        """Always 0 — this model contributes only :meth:`fees`."""
        return 0.0

    def fees(self, context: CommissionContext) -> Money:
        """SEC and TAF on sells; the options regulatory fee on both sides.

        ORF is not side-dependent — the exchanges charge it to open and to
        close — so an options strategy that only ever buys still pays it, and a
        sell-only rule would understate every long-premium round trip by half.
        """
        quantity = abs(context.quantity)
        if context.instrument.asset_class is AssetClass.OPTION:
            fee = self._options_orf_per_contract * quantity
            if context.side is OrderSide.SELL:
                fee += self._sec_fee_rate * context.notional
            return fee
        if context.side is not OrderSide.SELL:
            return 0.0
        # The TAF cap is per order, so a split sell pays it once, not once a slice.
        taf = _incremental(
            lambda filled: min(self._finra_taf_per_share * filled, self._finra_taf_cap),
            abs(context.filled_before),
            quantity,
        )
        return self._sec_fee_rate * context.notional + taf


class TieredCommissionModel(CommissionModel):
    """Volume-banded rates, e.g. cheaper above 300k shares/month."""

    name: ClassVar[str] = "tiered"

    def __init__(self, tiers: Sequence[tuple[Quantity, Money]], minimum: Money = 0.0) -> None:
        """``tiers`` are ``(monthly volume at or above which, per-unit rate)``.

        Sorted on construction so a config may list them in any order, and the
        lowest band applies below every threshold — a schedule that starts at
        300k shares still has to price the first trade of the month.
        """
        if not tiers:
            raise ValidationError(
                "TieredCommissionModel needs at least one (threshold, rate) tier."
            )
        owner = type(self).__name__
        for threshold, rate in tiers:
            _require_non_negative(threshold, "tier threshold", owner)
            _require_non_negative(rate, "tier rate", owner)
        self._tiers = tuple(sorted(tiers, key=lambda tier: tier[0]))
        self._minimum = _require_non_negative(minimum, "minimum", owner)

    def commission(self, context: CommissionContext) -> Money:
        rate = self._tiers[0][1]
        for threshold, tier_rate in self._tiers:
            if context.period_volume < threshold:
                break
            rate = tier_rate
        return _incremental(
            lambda filled: (
                max(rate * filled, self._minimum) if filled > _QUANTITY_TOLERANCE else 0.0
            ),
            abs(context.filled_before),
            abs(context.quantity),
        )


class CompositeCommissionModel(CommissionModel):
    """Sums child models — the realistic default (broker + regulatory)."""

    name: ClassVar[str] = "composite"

    def __init__(self, models: Sequence[CommissionModel]) -> None:
        if not models:
            raise ValidationError(
                "CompositeCommissionModel needs at least one child model; use "
                "ZeroCommissionModel to run without costs."
            )
        self._models = tuple(models)

    @property
    def models(self) -> tuple[CommissionModel, ...]:
        return self._models

    def commission(self, context: CommissionContext) -> Money:
        return sum(model.commission(context) for model in self._models)

    def fees(self, context: CommissionContext) -> Money:
        return sum(model.fees(context) for model in self._models)
