"""Reporters — rendering a :class:`BacktestResult` for humans and machines.

Reporters are plugins (``sigmaloop.reporters``) so a team can add its own house
format without touching the engine. All of them read the result and write
output; none of them may mutate it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from sigmaloop.engine.runner import BatchResult
from sigmaloop.results.result import BacktestResult

__all__ = [
    "ReportOptions",
    "Reporter",
    "TextReporter",
    "CsvReporter",
    "JsonReporter",
    "HtmlReporter",
    "CompositeReporter",
]


@dataclass(frozen=True, slots=True)
class ReportOptions:
    """Presentation settings shared by all reporters."""

    output_directory: Path | None = None
    filename_prefix: str = ""
    currency_symbol: str = "$"
    decimal_places: int = 2
    #: Show the top N trades by absolute P&L in the summary.
    top_trades: int = 10
    include_monthly_returns: bool = True
    include_drawdown_table: bool = True
    include_parameters: bool = True
    include_warnings: bool = True
    max_width: int = 100


class Reporter(ABC):
    """Renders a result to text or to files."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        """Produce the report as a string."""
        raise NotImplementedError

    def write(
        self, result: BacktestResult, directory: Path, options: ReportOptions | None = None
    ) -> Sequence[Path]:
        """Write to disk; returns the files created."""
        raise NotImplementedError

    def render_batch(self, batch: BatchResult, options: ReportOptions | None = None) -> str:
        """Render a sweep as a comparison table. Default: not supported."""
        raise NotImplementedError


class TextReporter(Reporter):
    """Plain-text summary (Outputs requirement 4).

    Layout, in order:

    1. Header — strategy, mode, symbols, date range, parameters.
    2. **Warnings**, if any. First, not last: a caveat below the numbers is a
       caveat nobody reads.
    3. Headline — net profit, total return, CAGR, max drawdown, Sharpe.
    4. Trade statistics — count, win rate, expectancy, profit factor.
    5. Cost breakdown — commission, fees, slippage.
    6. Drawdown table and monthly return grid, if enabled.
    7. Top trades.
    8. Run mechanics — bars, fills, rejections, wall clock.
    """

    name: ClassVar[str] = "text"

    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError

    def render_batch(self, batch: BatchResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError


class CsvReporter(Reporter):
    """Writes ``trades.csv``, ``option_trades.csv``, ``equity_curve.csv``,
    ``metrics.csv``. The machine-readable path for downstream analysis."""

    name: ClassVar[str] = "csv"

    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError

    def write(
        self, result: BacktestResult, directory: Path, options: ReportOptions | None = None
    ) -> Sequence[Path]:
        raise NotImplementedError


class JsonReporter(Reporter):
    """Single JSON document: config, params, metrics, stats, warnings.

    Curves are downsampled by default to keep the file readable; the full
    series belong in CSV/Parquet.
    """

    name: ClassVar[str] = "json"

    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError


class HtmlReporter(Reporter):
    """Self-contained HTML tearsheet: equity and drawdown curves, monthly
    return heatmap, trade table. Requires the ``report`` extra."""

    name: ClassVar[str] = "html"

    def __init__(self, template: Path | None = None) -> None:
        raise NotImplementedError

    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError


class CompositeReporter(Reporter):
    """Runs several reporters over one result."""

    name: ClassVar[str] = "composite"

    def __init__(self, reporters: Sequence[Reporter]) -> None:
        raise NotImplementedError

    def render(self, result: BacktestResult, options: ReportOptions | None = None) -> str:
        raise NotImplementedError

    def write(
        self, result: BacktestResult, directory: Path, options: ReportOptions | None = None
    ) -> Sequence[Path]:
        raise NotImplementedError
