# SigmaLoop

A programmable, high-performance event-driven backtesting engine for equities
and options.

> **Status: scaffold.** This repository currently contains the full design
> document, type model, dataclasses and abstract interfaces. Method bodies raise
> `NotImplementedError` — see [DESIGN.md](DESIGN.md) for the architecture and
> [Roadmap](#roadmap) for what gets built next.

## What it does

Write a strategy in Python, run it against historical data, get industry-standard
backtest results.

Three modes, one engine:

- **Single-asset** — one underlying. *Buy TQQQ at the open; sell the lot at +10%.*
- **Single-asset + options** — the underlying, its options, or both.
  *Buy the SPY 0DTE 20-delta put and call; test a covered call on AMZN.*
- **Portfolio** — evaluate a universe and act on everything that matches.
  *Buy every breakout stock; buy ITM puts on the top-10% gainers.*

## Design highlights

- **No lookahead, structurally.** Signals from bar *t* execute at bar *t+1*'s
  open by default; the strategy context physically cannot index forward.
- **Honest costs.** Configurable commissions and regulatory fees, slippage with
  volume participation caps, and worst-side (pay the ask, hit the bid) pricing
  by default. Feeds without quotes get a *flagged and counted* synthetic spread.
- **Real options accounting.** Chains with greeks, delta/DTE/moneyness selectors,
  and explicit expiry, exercise and assignment resolution.
- **Sizing is yours.** Fixed quantity, fixed notional, percent of equity,
  risk-based, target weight, or a custom callable — never assumed by the engine.
- **Built for sweeps.** Columnar data, O(1) incremental indicators, heap-merged
  streaming, and process-parallel runs over symbols and parameter grids.
- **Everything is a plugin.** Data providers, indicators, execution models,
  slippage, commissions, sizers, metrics and reporters are resolved by name.

## Install

```bash
pip install -e ".[dev]"          # core + dev tooling
pip install -e ".[yahoo,polygon,report]"   # optional data sources and HTML reports
```

Requires Python 3.11+.

## Usage sketch

```python
from datetime import datetime, timezone

from sigmaloop import (
    BacktestConfig, BacktestEngine, Parameter, ParameterSpec,
    SingleAssetStrategy, StrategyMode,
)


class BuyTheOpen(SingleAssetStrategy):
    symbol = "TQQQ"

    @classmethod
    def param_spec(cls):
        return ParameterSpec.from_parameters(
            Parameter("take_profit_pct", 0.10, "Exit when up this much"),
        )

    def on_bar(self, ctx, bar):
        position = ctx.position(self.instrument_id)
        if position is None:
            ctx.buy(self.instrument_id, percent_equity=1.0)
        elif position.unrealized_pnl_pct >= ctx.params.take_profit_pct:
            ctx.close(self.instrument_id)


config = BacktestConfig(
    strategy_mode=StrategyMode.SINGLE_ASSET,
    start=datetime(2020, 1, 1, tzinfo=timezone.utc),
    end=datetime(2024, 12, 31, tzinfo=timezone.utc),
    symbols=("TQQQ",),
)

result = BacktestEngine(config, BuyTheOpen()).run()
print(result.summary())
```

Options mode overrides `on_bar(self, ctx, bar, chain)` and selects contracts off
the chain (`chain.by_delta(0.20, OptionRight.PUT)`); portfolio mode implements
`select_universe(ctx)` plus `on_bar(ctx, snapshot)`.

## Layout

```
sigmaloop/
  types.py        aliases, ids, enums          errors.py    exception hierarchy
  domain/         Bar, Order, Position, Trade, Instrument, account state
  data/           providers, feed, universe, cache, calendar
  indicators/     Indicator framework + built-in library
  execution/      pricing, slippage, commission, broker, option expiry
  portfolio/      ledger, sizing, risk, margin
  strategy/       base classes, context, parameters, order API
  engine/         config, clock, run context, bar loop, events, parallel runner
  metrics/        performance metric calculators
  results/        equity/drawdown curves, trade logs, result, reporters
  plugins/        registries and built-in registration
  utils/          logging, money, time
```

`DESIGN.md` documents every class, its data, and the interactions between them.

## Roadmap

1. Domain value objects and the CSV provider.
2. Feed, clock and the 13-phase bar loop; single-asset mode end to end.
3. Portfolio accounting, sizing, risk; metrics and the text reporter.
4. Options: chains, selectors, expiry and assignment.
5. Portfolio mode: universes, screens, rebalancing.
6. Yahoo and Polygon providers; caching.
7. Parallel sweeps.

Deferred by design (see DESIGN.md §15): parameter optimisation, walk-forward
analysis, Monte Carlo, futures, news/ML-driven and LLM-defined strategies.
