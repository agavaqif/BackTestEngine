"""Execution layer: pricing, slippage, commissions, matching and expiry."""

from __future__ import annotations

from sigmaloop.execution.broker import Broker, BrokerResult, SimulatedBroker
from sigmaloop.execution.commission import (
    CommissionContext,
    CommissionModel,
    CompositeCommissionModel,
    PercentValueCommissionModel,
    PerContractCommissionModel,
    PerShareCommissionModel,
    PerTradeCommissionModel,
    RegulatoryFeeModel,
    TieredCommissionModel,
    ZeroCommissionModel,
)
from sigmaloop.execution.expiry import (
    ExpiryEngine,
    ExpiryOutcome,
    ExpiryPolicy,
    StandardExpiryEngine,
)
from sigmaloop.execution.models import (
    ExecutionContext,
    ExecutionModel,
    FillDecision,
    NextBarCloseExecutionModel,
    NextBarOpenExecutionModel,
    SameBarCloseExecutionModel,
)
from sigmaloop.execution.pricing import (
    FillPriceModel,
    FixedBpsSpreadModel,
    OhlcFillPriceModel,
    PricingContext,
    QuoteFillPriceModel,
    SpreadModel,
    TickSpreadModel,
    VolatilitySpreadModel,
)
from sigmaloop.execution.slippage import (
    FixedBpsSlippageModel,
    NoSlippageModel,
    SlippageContext,
    SlippageModel,
    SpreadFractionSlippageModel,
    TickSlippageModel,
    VolumeShareSlippageModel,
)

__all__ = [
    "Broker",
    "BrokerResult",
    "CommissionContext",
    "CommissionModel",
    "CompositeCommissionModel",
    "ExecutionContext",
    "ExecutionModel",
    "ExpiryEngine",
    "ExpiryOutcome",
    "ExpiryPolicy",
    "FillDecision",
    "FillPriceModel",
    "FixedBpsSlippageModel",
    "FixedBpsSpreadModel",
    "NextBarCloseExecutionModel",
    "NextBarOpenExecutionModel",
    "NoSlippageModel",
    "OhlcFillPriceModel",
    "PerContractCommissionModel",
    "PerShareCommissionModel",
    "PerTradeCommissionModel",
    "PercentValueCommissionModel",
    "PricingContext",
    "QuoteFillPriceModel",
    "RegulatoryFeeModel",
    "SameBarCloseExecutionModel",
    "SimulatedBroker",
    "SlippageContext",
    "SlippageModel",
    "SpreadFractionSlippageModel",
    "SpreadModel",
    "StandardExpiryEngine",
    "TickSlippageModel",
    "TickSpreadModel",
    "TieredCommissionModel",
    "VolatilitySpreadModel",
    "VolumeShareSlippageModel",
    "ZeroCommissionModel",
]
