"""Run configuration.

One immutable, serialisable :class:`BacktestConfig` fully determines a run.
Together with the strategy class and its :class:`ParameterSet`, it is hashed
into the run id, so two results are directly comparable if and only if their
fingerprints match.

Every sub-config validates itself, and :meth:`BacktestConfig.validate` performs
the cross-cutting checks (options mode needs an options-capable provider; a
percent-equity default sizer needs a value in ``(0, 1]``; ``WORST`` pricing with
a quote-less provider needs a spread model). Failing here costs milliseconds;
failing mid-run costs the whole run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sigmaloop.data.universe import UniverseSpec
from sigmaloop.types import (
    Currency,
    ExecutionTiming,
    MarginModel,
    Money,
    ParamDict,
    PriceSelection,
    RunId,
    SizingMode,
    StrategyMode,
    Timeframe,
    UtcDatetime,
)

__all__ = [
    "DataConfig",
    "ExecutionConfig",
    "AccountingConfig",
    "OptionsConfig",
    "ReportingConfig",
    "ParallelConfig",
    "BacktestConfig",
]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where data comes from and how it is preprocessed."""

    #: Registered provider names, tried in order (first match wins per asset class).
    providers: tuple[str, ...] = ("csv",)
    provider_options: dict[str, dict[str, object]] = field(default_factory=dict)
    timeframe: Timeframe = Timeframe.D1
    #: Prepend history so indicators are warm at ``start``. ``None`` == derive
    #: from the strategy's declared indicators (the recommended setting).
    warmup_bars: int | None = None
    adjust_for_splits: bool = True
    adjust_for_dividends: bool = False
    #: Credit cash dividends as they are paid, instead of back-adjusting prices.
    cash_dividends: bool = True
    cache_directory: Path | None = None
    cache_max_bytes: int = 2 << 30
    prefetch_buffer: int = 512
    calendar: str = "nyse"
    #: Abort on a data-integrity breach rather than logging and skipping the bar.
    strict_validation: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Fill timing, pricing, slippage and commissions."""

    timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    price_selection: PriceSelection = PriceSelection.WORST
    execution_model: str = "next_bar_open"
    #: Synthesises a spread when the feed has none; required for WORST/BEST.
    spread_model: str | None = "fixed_bps"
    spread_model_options: dict[str, object] = field(default_factory=dict)
    slippage_model: str = "fixed_bps"
    slippage_options: dict[str, object] = field(default_factory=dict)
    #: Commission models are summed, so broker + regulatory compose naturally.
    commission_models: tuple[str, ...] = ("per_share", "regulatory")
    commission_options: dict[str, dict[str, object]] = field(default_factory=dict)
    #: Cap on the fraction of a bar's volume one order may consume.
    max_volume_participation: float | None = 0.025
    #: What to do with the unfilled remainder: carry to the next bar or cancel.
    carry_unfilled_remainder: bool = False
    allow_partial_fills: bool = True


@dataclass(frozen=True, slots=True)
class AccountingConfig:
    """Capital, sizing defaults, margin and breach policy."""

    initial_cash: Money = 100_000.0
    currency: Currency = Currency.USD
    #: Used whenever an order omits explicit sizing.
    default_sizing_mode: SizingMode = SizingMode.PERCENT_EQUITY
    default_sizing_value: float = 0.10
    allow_short: bool = True
    allow_fractional_shares: bool = False
    margin_model: MarginModel = MarginModel.CASH
    max_leverage: float = 1.0
    max_position_weight: float | None = None
    max_open_positions: int | None = None
    #: ``"reject"`` refuses the order; ``"flag"`` records the breach and lets it
    #: through, so a run can quantify how often the strategy overreaches.
    on_capital_breach: str = "reject"
    raise_on_reject: bool = False
    #: FIFO or LIFO lot matching for realised P&L.
    lot_matching: str = "fifo"
    #: Flatten everything on the final bar so no trade is left open.
    liquidate_at_end: bool = True
    #: Annualised rate credited on idle cash.
    cash_interest_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class OptionsConfig:
    """Options-mode data narrowing and expiry policy."""

    enabled: bool = False
    min_dte: int | None = None
    max_dte: int | None = None
    strike_window_pct: float | None = 0.25
    require_greeks: bool = True
    #: Skip contracts with no two-sided market — they cannot be traded.
    require_two_sided_quotes: bool = True
    max_spread_pct: float | None = 0.50
    min_open_interest: float | None = None
    close_before_expiry_bars: int | None = None
    expiry_assignment_probability: float = 1.0
    early_assignment_probability: float = 0.0
    allow_physical_settlement: bool = True


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    """What the run emits when it finishes."""

    output_directory: Path | None = None
    reporters: tuple[str, ...] = ("text",)
    #: Benchmark symbol for alpha/beta and relative return.
    benchmark: str | None = None
    #: Annual risk-free rate used by Sharpe and Sortino.
    risk_free_rate: float = 0.0
    save_trade_log: bool = True
    save_equity_curve: bool = True
    save_orders: bool = False
    #: Retain every per-bar snapshot. Expensive; off by default.
    capture_bar_snapshots: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Parallelism across symbols and parameter sets (NFR 2).

    Parallelism is at the RUN level, not inside a single run's bar loop: bar
    processing is inherently sequential (each bar depends on the prior ledger
    state), but independent runs — one per parameter set, or per symbol in
    single-asset mode — are embarrassingly parallel.

    Processes rather than threads, because the loop is CPU-bound Python. The
    per-worker cost is data loading, which the shared on-disk cache amortises.
    """

    enabled: bool = False
    #: ``None`` == ``os.cpu_count() - 1``.
    max_workers: int | None = None
    #: ``"process"`` for real parallelism; ``"thread"`` only for I/O-bound
    #: provider fetches; ``"serial"`` for debugging.
    executor: str = "process"
    chunk_size: int = 1
    #: Fail the whole sweep on the first worker error, or collect and continue.
    fail_fast: bool = False


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Complete, reproducible description of one run."""

    strategy_mode: StrategyMode
    start: UtcDatetime
    end: UtcDatetime
    symbols: tuple[str, ...] = ()
    universe: UniverseSpec | None = None
    strategy_params: ParamDict = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    accounting: AccountingConfig = field(default_factory=AccountingConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    #: Seeds every stochastic component (assignment draws, tie-breaks).
    seed: int = 0
    run_id: RunId | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        raise NotImplementedError

    def validate(self) -> Sequence[str]:
        """Return human-readable problems; empty means the config is runnable.

        Cross-cutting checks live here rather than in ``__post_init__`` because
        several of them need the plugin registry (does this provider exist? can
        it serve options?), which is not available at construction time.
        """
        raise NotImplementedError

    def fingerprint(self) -> str:
        """Stable hash over every field — the identity of this configuration."""
        raise NotImplementedError

    def derive(self, **overrides: object) -> BacktestConfig:
        """Copy with overrides. The sweep primitive."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BacktestConfig:
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: Path) -> BacktestConfig:
        """Load from YAML or JSON."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError
