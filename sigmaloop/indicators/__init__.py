"""Indicator framework and built-in library."""

from __future__ import annotations

from sigmaloop.indicators.base import (
    CompositeIndicator,
    Indicator,
    IndicatorSet,
    IndicatorSpec,
    RollingIndicator,
)
from sigmaloop.indicators.library import (
    AverageTrueRange,
    BollingerBands,
    ExponentialMovingAverage,
    Macd,
    RateOfChange,
    RelativeStrengthIndex,
    RollingHigh,
    RollingLow,
    RollingStdDev,
    SimpleMovingAverage,
)

__all__ = [
    "AverageTrueRange",
    "BollingerBands",
    "CompositeIndicator",
    "ExponentialMovingAverage",
    "Indicator",
    "IndicatorSet",
    "IndicatorSpec",
    "Macd",
    "RateOfChange",
    "RelativeStrengthIndex",
    "RollingHigh",
    "RollingIndicator",
    "RollingLow",
    "RollingStdDev",
    "SimpleMovingAverage",
]
