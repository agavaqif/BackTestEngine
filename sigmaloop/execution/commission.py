"""Commission and fee models (Execution requirement 3).

Commissions are configurable and composable: a realistic US equity/options
account pays a broker commission plus regulatory fees (SEC, TAF, OCC, ORF),
each with its own base and its own side-dependence.
:class:`CompositeCommissionModel` sums them so no single model has to encode
an entire fee schedule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import Money, OrderSide, Percent, Price, Quantity

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


class CommissionModel(ABC):
    """Computes the cash cost of executing one fill."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def commission(self, context: CommissionContext) -> Money:
        """Broker commission, always >= 0."""
        raise NotImplementedError

    def fees(self, context: CommissionContext) -> Money:
        """Regulatory/exchange fees, separate from commission. Default 0."""
        raise NotImplementedError

    def total(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class ZeroCommissionModel(CommissionModel):
    """No costs. Useful as a baseline to isolate the impact of frictions."""

    name: ClassVar[str] = "zero"

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class PerShareCommissionModel(CommissionModel):
    """``rate * shares``, with optional per-order floor and cap."""

    name: ClassVar[str] = "per_share"

    def __init__(
        self,
        rate: Money = 0.005,
        minimum: Money = 1.0,
        maximum_pct_of_value: Percent | None = 0.01,
    ) -> None:
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class PerTradeCommissionModel(CommissionModel):
    """Flat fee per order, regardless of size."""

    name: ClassVar[str] = "per_trade"

    def __init__(self, fee: Money = 1.0) -> None:
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class PercentValueCommissionModel(CommissionModel):
    """A percentage of traded notional."""

    name: ClassVar[str] = "percent_value"

    def __init__(self, rate: Percent = 0.001, minimum: Money = 0.0) -> None:
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


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
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


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
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        """Always 0 — this model contributes only :meth:`fees`."""
        raise NotImplementedError

    def fees(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class TieredCommissionModel(CommissionModel):
    """Volume-banded rates, e.g. cheaper above 300k shares/month."""

    name: ClassVar[str] = "tiered"

    def __init__(self, tiers: Sequence[tuple[Quantity, Money]], minimum: Money = 0.0) -> None:
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError


class CompositeCommissionModel(CommissionModel):
    """Sums child models — the realistic default (broker + regulatory)."""

    name: ClassVar[str] = "composite"

    def __init__(self, models: Sequence[CommissionModel]) -> None:
        raise NotImplementedError

    def commission(self, context: CommissionContext) -> Money:
        raise NotImplementedError

    def fees(self, context: CommissionContext) -> Money:
        raise NotImplementedError
