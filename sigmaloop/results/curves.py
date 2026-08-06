"""Equity and drawdown curves (Outputs requirement 3).

Columnar for the same reason bars are: a 10-year minute backtest produces
~1M equity points, and a list of dataclasses would cost hundreds of megabytes
and make every metric a Python loop. Buffers are preallocated from the feed
plan's bar estimate and grown geometrically if that estimate is low.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from sigmaloop.domain.account import EquityPoint
from sigmaloop.types import Money, Percent, UtcDatetime

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["EquityCurve", "DrawdownCurve", "DrawdownPeriod"]


class EquityCurve:
    """Per-bar account value — the primary output series.

    One row per bar, appended at phase 7 of the engine loop, so the curve is
    complete even on bars where nothing traded (Accounting requirement #1).
    """

    __slots__ = (
        "_ts",
        "_equity",
        "_cash",
        "_positions_value",
        "_gross_exposure",
        "_net_exposure",
        "_realized_pnl",
        "_open_positions",
        "_size",
        "_capacity",
        "_initial_cash",
    )

    def __init__(self, initial_cash: Money, capacity: int = 4096) -> None:
        raise NotImplementedError

    def append(self, point: EquityPoint) -> None:
        raise NotImplementedError

    # ---- columns (zero-copy views) ------------------------------------------ #

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    @property
    def equity(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def cash(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def positions_value(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    # ---- derived series ------------------------------------------------------ #

    def returns(self) -> npt.NDArray[np.float64]:
        """Simple bar-over-bar returns; length ``n - 1``."""
        raise NotImplementedError

    def log_returns(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def cumulative_returns(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def high_water_mark(self) -> npt.NDArray[np.float64]:
        """Running maximum — the drawdown baseline."""
        raise NotImplementedError

    def drawdown_curve(self) -> DrawdownCurve:
        raise NotImplementedError

    def resample(self, period: str) -> EquityCurve:
        """Downsample to daily/weekly/monthly for reporting and for computing
        monthly return tables."""
        raise NotImplementedError

    # ---- access --------------------------------------------------------------- #

    @property
    def initial_equity(self) -> Money:
        raise NotImplementedError

    @property
    def final_equity(self) -> Money:
        raise NotImplementedError

    def point_at(self, index: int) -> EquityPoint:
        raise NotImplementedError

    def to_frame(self) -> object:
        """``pandas.DataFrame`` for export and plotting."""
        raise NotImplementedError

    def to_csv(self, path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[EquityPoint]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DrawdownPeriod:
    """One peak-to-trough-to-recovery episode.

    ``recovered_at`` is ``None`` for a drawdown still open at the end of the
    run — reported explicitly rather than silently treated as recovered.
    """

    peak_at: UtcDatetime
    trough_at: UtcDatetime
    recovered_at: UtcDatetime | None
    peak_equity: Money
    trough_equity: Money
    depth_pct: Percent
    depth_value: Money

    @property
    def duration(self) -> timedelta:
        """Peak to recovery (or to run end if unrecovered)."""
        raise NotImplementedError

    @property
    def time_to_trough(self) -> timedelta:
        raise NotImplementedError

    @property
    def is_recovered(self) -> bool:
        raise NotImplementedError


class DrawdownCurve:
    """Underwater series plus the episode decomposition."""

    __slots__ = ("_ts", "_drawdown", "_underwater_duration", "_periods")

    def __init__(
        self,
        timestamps: npt.NDArray[np.int64],
        drawdown: npt.NDArray[np.float64],
    ) -> None:
        raise NotImplementedError

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        raise NotImplementedError

    @property
    def drawdown(self) -> npt.NDArray[np.float64]:
        """Fractional distance below the high-water mark; values <= 0."""
        raise NotImplementedError

    def max_drawdown(self) -> Percent:
        raise NotImplementedError

    def periods(self, min_depth_pct: float = 0.0) -> Sequence[DrawdownPeriod]:
        """Episodes deeper than ``min_depth_pct``, worst first."""
        raise NotImplementedError

    def longest_period(self) -> DrawdownPeriod | None:
        """Longest by duration — often a different episode from the deepest."""
        raise NotImplementedError

    def to_frame(self) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
