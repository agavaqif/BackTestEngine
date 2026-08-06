"""``BacktestResult`` — the complete, self-describing output of one run.

Self-describing matters: the result carries the config and parameter set that
produced it, plus the diagnostics that qualify it (synthetic-quote fills, stale
marks, lookahead warnings). A result handed to someone else should not need a
verbal footnote to be interpretable.

A result is produced even for a failed run, carrying ``RunState.FAILED``, the
error, and every bar completed before the failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sigmaloop.domain.account import CashFlow
from sigmaloop.domain.order import Fill, Order
from sigmaloop.engine.config import BacktestConfig
from sigmaloop.metrics.performance import PerformanceMetrics
from sigmaloop.results.curves import DrawdownCurve, EquityCurve
from sigmaloop.results.trade_log import OptionTradeLog, TradeLog
from sigmaloop.strategy.params import ParameterSet
from sigmaloop.types import Money, RunId, RunState, UtcDatetime

__all__ = ["RunSummaryStats", "BacktestResult"]


@dataclass(frozen=True, slots=True)
class RunSummaryStats:
    """Run mechanics — how the simulation itself behaved.

    Distinct from performance metrics: these describe the *simulation's*
    fidelity, and they are what tell a reader whether the performance numbers
    can be trusted.
    """

    bars_processed: int = 0
    warmup_bars: int = 0
    instruments_traded: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_expired: int = 0
    partial_fills: int = 0
    #: Fills priced off a synthesised (not observed) spread.
    synthetic_quote_fills: int = 0
    #: Positions marked at a stale price because the bar was missing.
    stale_marks: int = 0
    capital_breaches: int = 0
    option_expiries: int = 0
    option_assignments: int = 0
    wall_clock_seconds: float = 0.0
    bars_per_second: float = 0.0
    peak_memory_mb: float | None = None


@dataclass(slots=True)
class BacktestResult:
    """Everything a run produced."""

    run_id: RunId
    state: RunState
    config: BacktestConfig
    strategy_name: str
    params: ParameterSet
    start: UtcDatetime
    end: UtcDatetime

    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    equity_curve: EquityCurve | None = None
    drawdown_curve: DrawdownCurve | None = None
    trades: TradeLog = field(default_factory=TradeLog)
    option_trades: OptionTradeLog = field(default_factory=OptionTradeLog)

    #: Full order and fill history; retained only when
    #: ``ReportingConfig.save_orders`` is set, since it is the largest artefact.
    orders: tuple[Order, ...] = ()
    fills: tuple[Fill, ...] = ()
    cash_flows: tuple[CashFlow, ...] = ()

    stats: RunSummaryStats = field(default_factory=RunSummaryStats)
    #: Caveats that qualify the numbers — lookahead-prone execution model,
    #: synthetic spreads, gappy data. Always shown in the summary.
    warnings: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    #: Custom per-bar series recorded via ``ctx.record``.
    recorded: dict[str, tuple[float, ...]] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    # ---- convenience ---------------------------------------------------------- #

    @property
    def succeeded(self) -> bool:
        raise NotImplementedError

    @property
    def initial_equity(self) -> Money:
        raise NotImplementedError

    @property
    def final_equity(self) -> Money:
        raise NotImplementedError

    @property
    def net_profit(self) -> Money:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        """Config + params hash — the identity two results are compared on."""
        raise NotImplementedError

    # ---- output ----------------------------------------------------------------- #

    def summary(self) -> str:
        """Human-readable text summary (Outputs requirement 4)."""
        raise NotImplementedError

    def save(self, directory: Path) -> Sequence[Path]:
        """Write every configured artefact; returns the files created."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form. Curves are downsampled unless ``full=True``."""
        raise NotImplementedError

    def compare(self, other: BacktestResult) -> dict[str, tuple[object, object]]:
        """Field-by-field diff of metrics and parameters against another run."""
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
