"""``BacktestEngine`` — the single bar loop that runs every strategy mode.

Functional requirement 5 says one engine must run all modes. It does, because
mode-specific behaviour is pushed into three pluggable places — the strategy
base class (how a bar is unpacked), the feed plan (what data is loaded), and the
option-chain/expiry components (whether they are installed at all). The loop
below is identical for single-asset, options and portfolio runs.

Per-bar phase order (fixed, and the reason results are reproducible)
--------------------------------------------------------------------

::

    1.  clock.advance(snapshot.timestamp)
    2.  apply corporate actions with this ex-date
    3.  broker.process_bar(snapshot)          -> fills / expirations
    4.  portfolio.apply_fill(...) for each     -> realised trades
    5.  expiry engine: settle expiries, assignments
    6.  accrue carry (borrow cost, cash interest)
    7.  portfolio.mark_to_market(snapshot)     -> EquityPoint  (Accounting #1)
    8.  indicators.update_all(bar) for each bar in the snapshot
    9.  deliver on_fill / on_order_rejected / on_trade_closed / on_option_expiry
    10. [portfolio mode] re-resolve universe if this is a rebalance bar
    11. strategy.handle_bar(context)           -> OrderIntents
    12. size -> risk-check -> broker.submit(order)   (works from the NEXT bar)
    13. record equity point, emit BarClosed

Steps 3 and 11 are the lookahead firewall: orders raised at step 12 of bar *t*
cannot fill before step 3 of bar *t+1*. Step 7 precedes step 11 so a strategy
reading ``ctx.portfolio.equity`` sees this bar's true value, not last bar's.

Warm-up bars run steps 1-9 only; no orders are accepted until every declared
indicator is ready.
"""

from __future__ import annotations

from collections.abc import Sequence

from sigmaloop.data.feed import DataFeed
from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.order import Order, OrderIntent
from sigmaloop.engine.config import BacktestConfig
from sigmaloop.engine.context import RunContext
from sigmaloop.engine.events import EventBus, EventListener
from sigmaloop.results.result import BacktestResult
from sigmaloop.strategy.base import Strategy
from sigmaloop.types import RunId, RunState

__all__ = ["BacktestEngine", "CancellationToken"]


class CancellationToken:
    """Cooperative cancellation, checked once per bar.

    Needed because a parameter sweep may need to abandon in-flight runs, and a
    long run should be interruptible without killing the process and losing the
    partial result.
    """

    __slots__ = ("_cancelled", "_reason")

    def __init__(self) -> None:
        raise NotImplementedError

    def cancel(self, reason: str = "") -> None:
        raise NotImplementedError

    @property
    def is_cancelled(self) -> bool:
        raise NotImplementedError

    def raise_if_cancelled(self) -> None:
        raise NotImplementedError


class BacktestEngine:
    """Orchestrates one backtest from config + strategy to result.

    Construction is cheap; :meth:`prepare` does the expensive work (plugin
    resolution, data loading, warm-up) and :meth:`run` executes the loop. The
    split exists so a sweep worker can validate a whole batch of configs before
    committing to any I/O, and so tests can drive the loop bar-by-bar via
    :meth:`step`.

    Not thread-safe. Parallelism happens across engines, not within one — see
    :class:`~sigmaloop.engine.runner.BacktestRunner`.
    """

    __slots__ = (
        "_config",
        "_strategy",
        "_context",
        "_feed",
        "_events",
        "_state",
        "_run_id",
        "_cancellation",
        "_expiry_engine",
        "_equity_curve",
        "_trade_log",
        "_started_at",
    )

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        *,
        feed: DataFeed | None = None,
        events: EventBus | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        """``feed`` is normally built from config; injecting one is for tests."""
        raise NotImplementedError

    # ---- lifecycle ----------------------------------------------------------- #

    def prepare(self) -> None:
        """Resolve plugins, build the feed plan, load warm-up data.

        Order matters: config validation, then plugin resolution (so a missing
        provider is reported by name), then strategy declarations (which
        determine warm-up length), then data loading.
        """
        raise NotImplementedError

    def run(self) -> BacktestResult:
        """Execute the full loop and return the result.

        Calls :meth:`prepare` if it has not been called. Always produces a
        result — on failure the result carries ``RunState.FAILED``, the error,
        and every bar processed up to that point, because a partial equity
        curve is far more diagnostic than a bare traceback.
        """
        raise NotImplementedError

    def step(self, snapshot: MarketSnapshot) -> None:
        """Process exactly one bar through the 13 phases documented above."""
        raise NotImplementedError

    def finalise(self) -> BacktestResult:
        """Liquidate if configured, compute metrics, assemble the result."""
        raise NotImplementedError

    def close(self) -> None:
        """Release providers and buffers. Idempotent."""
        raise NotImplementedError

    # ---- observation ---------------------------------------------------------- #

    def subscribe(self, listener: EventListener) -> None:
        raise NotImplementedError

    @property
    def state(self) -> RunState:
        raise NotImplementedError

    @property
    def run_id(self) -> RunId:
        raise NotImplementedError

    @property
    def context(self) -> RunContext:
        raise NotImplementedError

    # ---- phase implementations (internal, one per numbered step) -------------- #

    def _apply_corporate_actions(self, snapshot: MarketSnapshot) -> None:
        raise NotImplementedError

    def _process_broker(self, snapshot: MarketSnapshot) -> None:
        """Steps 3-4: match working orders and post fills to the ledger."""
        raise NotImplementedError

    def _process_expiries(self, snapshot: MarketSnapshot) -> None:
        """Step 5. No-op when the run has no option positions."""
        raise NotImplementedError

    def _accrue_carry(self, snapshot: MarketSnapshot) -> None:
        """Step 6: short-borrow cost and cash interest for one bar."""
        raise NotImplementedError

    def _update_indicators(self, snapshot: MarketSnapshot) -> None:
        raise NotImplementedError

    def _dispatch_callbacks(self) -> None:
        """Step 9. User exceptions are wrapped in ``StrategyError`` with the
        bar timestamp attached, so a failure is locatable."""
        raise NotImplementedError

    def _resolve_universe(self, snapshot: MarketSnapshot) -> None:
        """Step 10; portfolio mode only, and only on rebalance bars."""
        raise NotImplementedError

    def _invoke_strategy(self, snapshot: MarketSnapshot) -> None:
        raise NotImplementedError

    def _process_intents(self, intents: Sequence[OrderIntent]) -> Sequence[Order]:
        """Step 12: size -> risk-check -> submit.

        A rejection here is recorded and delivered to the strategy on the next
        bar; it never raises unless ``AccountingConfig.raise_on_reject`` is set.
        """
        raise NotImplementedError

    def _build_feed(self) -> DataFeed:
        raise NotImplementedError

    def _build_context(self) -> RunContext:
        """Instantiate every component named in the config via the registries."""
        raise NotImplementedError
