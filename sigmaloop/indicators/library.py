"""Built-in indicators.

A deliberately small set: the primitives most strategies need, each also
serving as a worked example of the :class:`~sigmaloop.indicators.base.Indicator`
contract for users writing their own.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, ClassVar

from sigmaloop.domain.bar import Bar, BarSeries
from sigmaloop.indicators.base import Indicator, RollingIndicator

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "RollingStdDev",
    "RelativeStrengthIndex",
    "AverageTrueRange",
    "BollingerBands",
    "Macd",
    "RollingHigh",
    "RollingLow",
    "RateOfChange",
]


class SimpleMovingAverage(RollingIndicator):
    """Arithmetic mean of the last ``period`` values. O(1) via a running sum."""

    name: ClassVar[str] = "sma"

    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        raise NotImplementedError

    def compute(self, series: BarSeries) -> npt.NDArray[np.float64]:
        """Vectorised via cumulative-sum differencing."""
        raise NotImplementedError


class ExponentialMovingAverage(Indicator[float]):
    """Recursive EMA, ``alpha = 2 / (period + 1)``.

    Seeded with an SMA over the first ``period`` bars so results do not depend
    on where the history happens to start.
    """

    name: ClassVar[str] = "ema"

    def __init__(self, period: int, field: str = "close", **params: object) -> None:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> float | None:
        raise NotImplementedError

    @property
    def value(self) -> float | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RollingStdDev(RollingIndicator):
    """Sample standard deviation over the window, via Welford's algorithm."""

    name: ClassVar[str] = "stddev"

    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        raise NotImplementedError


class RelativeStrengthIndex(Indicator[float]):
    """Wilder's RSI. Smoothed average gain / average loss, scaled 0-100."""

    name: ClassVar[str] = "rsi"

    def __init__(self, period: int = 14, **params: object) -> None:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> float | None:
        raise NotImplementedError

    @property
    def value(self) -> float | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class AverageTrueRange(Indicator[float]):
    """Wilder's ATR. The engine's default volatility unit for stop sizing."""

    name: ClassVar[str] = "atr"

    def __init__(self, period: int = 14, **params: object) -> None:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> float | None:
        raise NotImplementedError

    @property
    def value(self) -> float | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class BollingerBands(Indicator[tuple[float, float, float]]):
    """``(lower, middle, upper)`` at ``num_std`` deviations around an SMA."""

    name: ClassVar[str] = "bbands"

    def __init__(self, period: int = 20, num_std: float = 2.0, **params: object) -> None:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> tuple[float, float, float] | None:
        raise NotImplementedError

    @property
    def value(self) -> tuple[float, float, float] | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Macd(Indicator[tuple[float, float, float]]):
    """``(macd, signal, histogram)``. Example of a composite indicator."""

    name: ClassVar[str] = "macd"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **params: object,
    ) -> None:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> tuple[float, float, float] | None:
        raise NotImplementedError

    @property
    def value(self) -> tuple[float, float, float] | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RollingHigh(RollingIndicator):
    """Highest high over the window — the breakout primitive."""

    name: ClassVar[str] = "rolling_high"

    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        """Monotonic deque keeps this O(1) amortised rather than O(period)."""
        raise NotImplementedError


class RollingLow(RollingIndicator):
    """Lowest low over the window."""

    name: ClassVar[str] = "rolling_low"

    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        raise NotImplementedError


class RateOfChange(RollingIndicator):
    """Percent change over ``period`` bars — the "top 10% gainers" score."""

    name: ClassVar[str] = "roc"

    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        raise NotImplementedError
