"""Performance metrics.

Two families, computed from two different inputs:

* **Return-based** (Sharpe, CAGR, drawdown, volatility) — from the equity
  curve. These describe the *account*.
* **Trade-based** (win rate, expectancy, profit factor, payoff) — from the
  trade log. These describe the *strategy*.

They can disagree: a strategy with a 70% win rate can still have a falling
equity curve. Reporting both, side by side, is the point.

Annualisation never assumes 252. The factor comes from
:meth:`~sigmaloop.engine.clock.SimulationClock.bars_per_year`, derived from the
run's timeframe and trading calendar, so an hourly backtest annualises correctly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, ClassVar

from sigmaloop.domain.position import Trade
from sigmaloop.types import Money, Percent, UtcDatetime

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from sigmaloop.results.curves import EquityCurve

__all__ = [
    "MetricContext",
    "PerformanceMetrics",
    "MetricCalculator",
    "MetricsEngine",
    "ReturnMetricsCalculator",
    "DrawdownMetricsCalculator",
    "TradeMetricsCalculator",
    "RiskMetricsCalculator",
    "BenchmarkMetricsCalculator",
]


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Everything a calculator may read."""

    equity_curve: EquityCurve
    trades: Sequence[Trade]
    initial_cash: Money
    start: UtcDatetime
    end: UtcDatetime
    bars_per_year: float
    risk_free_rate: float = 0.0
    benchmark_returns: npt.NDArray[np.float64] | None = None
    #: Fraction of bars with any open position — the exposure denominator.
    bars_in_market: int = 0
    total_bars: int = 0


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """The standard result set (Metrics requirement).

    Every field is optional-by-default so a run too short to support a metric
    reports ``None`` rather than a misleading zero — an infinite Sharpe from
    three bars is worse than no Sharpe.
    """

    # --- headline -------------------------------------------------------- #
    net_profit: Money = 0.0
    total_return_pct: Percent = 0.0
    cagr: Percent | None = None
    final_equity: Money = 0.0

    # --- risk-adjusted ---------------------------------------------------- #
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    #: Annualised standard deviation of bar returns.
    volatility: float | None = None
    downside_deviation: float | None = None

    # --- drawdown ---------------------------------------------------------- #
    max_drawdown_pct: Percent = 0.0
    max_drawdown_value: Money = 0.0
    max_drawdown_duration: timedelta | None = None
    max_drawdown_start: UtcDatetime | None = None
    max_drawdown_end: UtcDatetime | None = None
    recovery_factor: float | None = None
    ulcer_index: float | None = None

    # --- trade statistics ---------------------------------------------------- #
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Percent = 0.0
    #: Expected P&L per trade: ``win_rate*avg_win + loss_rate*avg_loss``.
    expectancy: Money = 0.0
    expectancy_r: float | None = None
    profit_factor: float | None = None
    payoff_ratio: float | None = None
    avg_win: Money = 0.0
    avg_loss: Money = 0.0
    largest_win: Money = 0.0
    largest_loss: Money = 0.0
    avg_holding_period: timedelta | None = None
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # --- activity and cost ---------------------------------------------------- #
    total_commission: Money = 0.0
    total_fees: Money = 0.0
    total_slippage: Money = 0.0
    turnover: float = 0.0
    exposure_pct: Percent = 0.0
    #: Costs as a fraction of gross profit — how much the broker took.
    cost_to_profit_ratio: float | None = None

    # --- tail risk ------------------------------------------------------------ #
    value_at_risk_95: Percent | None = None
    conditional_var_95: Percent | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    best_bar_return: Percent | None = None
    worst_bar_return: Percent | None = None

    # --- benchmark-relative ---------------------------------------------------- #
    benchmark_return_pct: Percent | None = None
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None
    correlation: float | None = None

    #: Anything registered by a custom calculator.
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def headline(self) -> dict[str, object]:
        """The handful of numbers the text summary leads with."""
        raise NotImplementedError


class MetricCalculator(ABC):
    """A pluggable group of metrics.

    Grouped rather than one-metric-per-class because most metrics share
    intermediate work (a return series, a drawdown series); recomputing that per
    metric would be wasteful and error-prone.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def compute(self, context: MetricContext) -> dict[str, object]:
        """Return field-name -> value, merged into :class:`PerformanceMetrics`."""
        raise NotImplementedError

    def requires_trades(self) -> bool:
        raise NotImplementedError

    def minimum_bars(self) -> int:
        """Below this bar count the calculator is skipped and its fields stay
        ``None``, rather than producing a statistically meaningless number."""
        raise NotImplementedError


class ReturnMetricsCalculator(MetricCalculator):
    """Net profit, total return, CAGR, volatility, Sharpe, Sortino.

    Sharpe uses *arithmetic* mean of excess bar returns, annualised by
    ``sqrt(bars_per_year)``. Sortino replaces the denominator with downside
    deviation about the risk-free rate.
    """

    name: ClassVar[str] = "returns"

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        raise NotImplementedError

    def compute(self, context: MetricContext) -> dict[str, object]:
        raise NotImplementedError


class DrawdownMetricsCalculator(MetricCalculator):
    """Max drawdown (depth, dates, duration), recovery factor, Calmar, ulcer.

    Computed vectorised from the equity column via a running maximum — O(n)
    with no Python loop, which matters because this runs on every sweep point.
    """

    name: ClassVar[str] = "drawdown"

    def compute(self, context: MetricContext) -> dict[str, object]:
        raise NotImplementedError


class TradeMetricsCalculator(MetricCalculator):
    """Win rate, expectancy, profit factor, payoff, streaks, holding period."""

    name: ClassVar[str] = "trades"

    def compute(self, context: MetricContext) -> dict[str, object]:
        raise NotImplementedError

    def requires_trades(self) -> bool:
        raise NotImplementedError


class RiskMetricsCalculator(MetricCalculator):
    """VaR, CVaR, skewness, kurtosis, best/worst bar."""

    name: ClassVar[str] = "risk"

    def __init__(self, confidence: float = 0.95) -> None:
        raise NotImplementedError

    def compute(self, context: MetricContext) -> dict[str, object]:
        raise NotImplementedError


class BenchmarkMetricsCalculator(MetricCalculator):
    """Alpha, beta, information ratio and correlation versus a benchmark.

    Skipped entirely when ``ReportingConfig.benchmark`` is unset.
    """

    name: ClassVar[str] = "benchmark"

    def compute(self, context: MetricContext) -> dict[str, object]:
        raise NotImplementedError


class MetricsEngine:
    """Runs the registered calculators and assembles :class:`PerformanceMetrics`.

    Custom calculators register here, which is how a user adds a domain-specific
    metric without forking the engine.
    """

    __slots__ = ("_calculators",)

    def __init__(self, calculators: Sequence[MetricCalculator] | None = None) -> None:
        raise NotImplementedError

    def register(self, calculator: MetricCalculator) -> None:
        raise NotImplementedError

    def compute(self, context: MetricContext) -> PerformanceMetrics:
        raise NotImplementedError

    @classmethod
    def default(cls, risk_free_rate: float = 0.0) -> MetricsEngine:
        """The standard set required by the spec, plus the usual companions."""
        raise NotImplementedError
