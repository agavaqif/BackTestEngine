"""Registers the built-in plugins.

Called once from :mod:`sigmaloop.__init__`. Kept separate from
:mod:`sigmaloop.plugins.registry` because the registry sits *below* every
implementation package in the dependency graph, and registration necessarily
imports from above it — doing both in one module would be a cycle.
"""

from __future__ import annotations

__all__ = ["bootstrap", "is_bootstrapped"]


def bootstrap() -> None:
    """Populate every registry with the built-in implementations.

    Idempotent: safe to call repeatedly, which matters because each subprocess
    in a parallel sweep re-imports the package and re-runs this.

    Registers:

    * data providers  — csv, yahoo, polygon
    * indicators      — sma, ema, stddev, rsi, atr, bbands, macd,
                        rolling_high, rolling_low, roc
    * execution       — next_bar_open (default), next_bar_close, same_bar_close
    * spread models   — fixed_bps, ticks, volatility
    * slippage        — none, fixed_bps, ticks, volume_share, spread_fraction
    * commissions     — zero, per_share, per_trade, percent_value,
                        per_contract, regulatory, tiered
    * sizers          — fixed_quantity, fixed_notional, percent_equity,
                        risk_percent, target_weight
    * risk checks     — capital, shorting, concentration, leverage, max_positions
    * metrics         — returns, drawdown, trades, risk, benchmark
    * reporters       — text, csv, json, html
    * calendars       — nyse, continuous
    """
    raise NotImplementedError


def is_bootstrapped() -> bool:
    raise NotImplementedError
