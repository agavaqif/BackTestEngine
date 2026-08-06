"""Plugin registries and built-in registration."""

from __future__ import annotations

from sigmaloop.plugins.bootstrap import bootstrap, is_bootstrapped
from sigmaloop.plugins.registry import (
    CALENDARS,
    COMMISSION_MODELS,
    DATA_PROVIDERS,
    EXECUTION_MODELS,
    INDICATORS,
    METRIC_CALCULATORS,
    POSITION_SIZERS,
    REPORTERS,
    RISK_CHECKS,
    SLIPPAGE_MODELS,
    SPREAD_MODELS,
    PluginRegistry,
    all_registries,
    register,
)

__all__ = [
    "CALENDARS",
    "COMMISSION_MODELS",
    "DATA_PROVIDERS",
    "EXECUTION_MODELS",
    "INDICATORS",
    "METRIC_CALCULATORS",
    "POSITION_SIZERS",
    "PluginRegistry",
    "REPORTERS",
    "RISK_CHECKS",
    "SLIPPAGE_MODELS",
    "SPREAD_MODELS",
    "all_registries",
    "bootstrap",
    "is_bootstrapped",
    "register",
]
