"""Strategy authoring layer — the primary user-facing API."""

from __future__ import annotations

from sigmaloop.strategy.api import (
    OptionLeg,
    OptionStructure,
    OrderBuilder,
    StrategyApi,
)
from sigmaloop.strategy.base import (
    OptionsStrategy,
    PortfolioStrategy,
    SingleAssetStrategy,
    Strategy,
)
from sigmaloop.strategy.context import StrategyContext
from sigmaloop.strategy.params import Parameter, ParameterSet, ParameterSpec

__all__ = [
    "OptionLeg",
    "OptionStructure",
    "OptionsStrategy",
    "OrderBuilder",
    "Parameter",
    "ParameterSet",
    "ParameterSpec",
    "PortfolioStrategy",
    "SingleAssetStrategy",
    "Strategy",
    "StrategyApi",
    "StrategyContext",
]
