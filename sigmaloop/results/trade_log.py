"""Trade logs (Outputs requirements 1 and 2).

Two views over the same underlying records:

* :class:`TradeLog` — every closed round trip, with the columns the spec asks
  for (instrument, direction, entry/exit time and price, P&L, costs, reason).
* :class:`OptionTradeLog` — the options view, adding strike, expiry, right, DTE,
  entry greeks and the settlement outcome. Options trades appear in both: the
  base log for portfolio-level accounting, the options log for structure
  analysis.

Kept as a separate object from :class:`~sigmaloop.results.result.BacktestResult`
so it can be exported, filtered and joined independently — the trade log is what
most users actually inspect after a run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

from sigmaloop.domain.position import OptionTrade, Trade
from sigmaloop.types import (
    AssetClass,
    InstrumentId,
    Money,
    OptionRight,
    PositionSide,
    Symbol,
    TradeCloseReason,
    UtcDatetime,
)

__all__ = ["TradeLog", "OptionTradeLog"]


class TradeLog:
    """Append-only collection of closed trades, with filtering and export."""

    __slots__ = ("_trades", "_by_instrument", "_by_tag")

    def __init__(self, trades: Sequence[Trade] = ()) -> None:
        raise NotImplementedError

    def append(self, trade: Trade) -> None:
        raise NotImplementedError

    # ---- filtering (each returns a new log, so filters chain) ---------------- #

    def filter(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        symbol: Symbol | None = None,
        asset_class: AssetClass | None = None,
        direction: PositionSide | None = None,
        winners_only: bool = False,
        losers_only: bool = False,
        close_reason: TradeCloseReason | None = None,
        tag: str | None = None,
        start: UtcDatetime | None = None,
        end: UtcDatetime | None = None,
    ) -> TradeLog:
        raise NotImplementedError

    def by_symbol(self) -> dict[Symbol, TradeLog]:
        """Split for per-symbol attribution — which names actually paid."""
        raise NotImplementedError

    def by_tag(self) -> dict[str, TradeLog]:
        """Split by strategy tag, so a multi-signal strategy can attribute P&L
        to each signal."""
        raise NotImplementedError

    # ---- aggregates ------------------------------------------------------------ #

    @property
    def net_pnl(self) -> Money:
        raise NotImplementedError

    @property
    def gross_pnl(self) -> Money:
        raise NotImplementedError

    @property
    def total_costs(self) -> Money:
        """Commission plus fees across every trade."""
        raise NotImplementedError

    def winners(self) -> TradeLog:
        raise NotImplementedError

    def losers(self) -> TradeLog:
        raise NotImplementedError

    # ---- export ---------------------------------------------------------------- #

    def to_frame(self) -> object:
        """``pandas.DataFrame``, one row per trade, spec columns first."""
        raise NotImplementedError

    def to_csv(self, path: str) -> None:
        raise NotImplementedError

    def to_records(self) -> Sequence[dict[str, object]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Trade]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Trade:
        raise NotImplementedError


class OptionTradeLog(TradeLog):
    """Options-specific trade log with contract-level columns and filters."""

    __slots__ = ("_by_underlying", "_by_expiry")

    def __init__(self, trades: Sequence[OptionTrade] = ()) -> None:
        raise NotImplementedError

    def filter_options(
        self,
        *,
        underlying: Symbol | None = None,
        right: OptionRight | None = None,
        expiry: date | None = None,
        min_dte_at_entry: int | None = None,
        max_dte_at_entry: int | None = None,
        assigned_only: bool = False,
        expired_worthless_only: bool = False,
    ) -> OptionTradeLog:
        raise NotImplementedError

    def by_underlying(self) -> dict[Symbol, OptionTradeLog]:
        raise NotImplementedError

    def by_dte_bucket(self, buckets: Sequence[int] = (0, 7, 30, 90)) -> dict[str, OptionTradeLog]:
        """Group by days-to-expiry at entry — 0DTE behaves nothing like 90DTE,
        and pooling them hides that."""
        raise NotImplementedError

    @property
    def total_premium(self) -> Money:
        """Net premium: positive if the strategy was a net seller."""
        raise NotImplementedError

    @property
    def assignment_count(self) -> int:
        raise NotImplementedError

    @property
    def expired_worthless_count(self) -> int:
        raise NotImplementedError

    def to_frame(self) -> object:
        """Adds underlying, right, strike, expiry, DTE and entry greeks."""
        raise NotImplementedError
