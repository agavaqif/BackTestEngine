# SigmaLoop — Design Document

**Version:** 0.1.0 (scaffold)
**Status:** Interfaces and data model defined; implementations pending.
**Source requirement:** `../Requirements.pdf`

---

## 1. Scope and design goals

SigmaLoop is a programmable, event-driven backtesting engine. A user writes a
strategy in Python, points it at historical data, and gets an industry-standard
performance report.

Three trading modes, one engine:

| Mode | Base class | Example from the requirements |
|---|---|---|
| Single-asset | `SingleAssetStrategy` | Buy TQQQ at the open; sell the lot at +10% |
| Single-asset + options | `OptionsStrategy` | Buy the SPY 0DTE 20-delta put and call; covered call on AMZN |
| Portfolio | `PortfolioStrategy` | Buy every breakout stock; buy ITM puts on the top-10% gainers |

Five properties the design optimises for, in priority order:

1. **No lookahead.** Enforced structurally, not by convention (§6).
2. **Correct accounting.** Every bar reconciles `equity == cash + Σ market_value`.
3. **Performance.** Columnar data, O(1) incremental indicators, heap-merged
   streaming, process-parallel sweeps.
4. **Extensibility.** Every swappable component is a named plugin (§10).
5. **Honest reporting.** Simulation shortcuts (synthetic spreads, stale marks,
   lookahead-prone models) are counted and surfaced with the results (§9.4).

---

## 2. Layered architecture

```
                       ┌──────────────────────────────┐
                       │  types.py  ·  errors.py      │   no dependencies
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │  domain/                     │   value objects
                       │  Bar · Order · Position ·    │
                       │  Instrument · Trade · Fill   │
                       └──┬────────┬────────┬─────────┘
                          │        │        │
          ┌───────────────▼──┐  ┌──▼─────┐  ├──────────────────┐
          │ data/            │  │ indi-  │  │ execution/       │
          │ providers, feed, │  │ cators/│  │ pricing, slippage│
          │ universe, cache, │  │        │  │ commission,      │
          │ calendar         │  │        │  │ broker, expiry   │
          └───────────────┬──┘  └──┬─────┘  └──────┬───────────┘
                          │        │               │
                          │     ┌──▼───────────────▼───────────┐
                          │     │ portfolio/                   │
                          │     │ accounting · sizing · risk   │
                          │     └──────────────┬───────────────┘
                          │                    │
                       ┌──▼────────────────────▼───────────────┐
                       │ strategy/                             │
                       │ base classes · context · params · api │
                       └──────────────────┬────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────┐
                       │ engine/                               │
                       │ config · clock · context · core loop  │
                       │ events · runner                       │
                       └──────────────────┬────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────┐
                       │ metrics/  ·  results/                 │
                       └───────────────────────────────────────┘

            plugins/ (registries)  ·  utils/ (logging, money, time)
                        cross-cut every layer
```

Dependency arrows point in one direction only. In particular, nothing in
`engine/` imports a concrete provider, indicator or commission model — those are
resolved by name through the plugin registries, which is what makes NFR 4
("plugin architecture") structural rather than aspirational.

### 2.1 Module map

| Module | Contents |
|---|---|
| `sigmaloop/types.py` | Type aliases, `NewType` ids, all enums |
| `sigmaloop/errors.py` | Exception hierarchy rooted at `SigmaLoopError` |
| `sigmaloop/domain/` | `Instrument`, `Bar`, `Order`, `Position`, `Trade`, account state |
| `sigmaloop/data/` | `DataProvider`, `DataFeed`, `Universe`, `DataCache`, `TradingCalendar` |
| `sigmaloop/data/providers/` | CSV, Yahoo, Polygon implementations |
| `sigmaloop/indicators/` | `Indicator` framework + built-in library |
| `sigmaloop/execution/` | Price selection, slippage, commissions, broker, option expiry |
| `sigmaloop/portfolio/` | Ledger, position sizing, pre-trade risk, margin |
| `sigmaloop/strategy/` | Strategy bases, `StrategyContext`, parameters, order API |
| `sigmaloop/engine/` | Config, clock, run context, bar loop, events, parallel runner |
| `sigmaloop/metrics/` | Metric calculators and `PerformanceMetrics` |
| `sigmaloop/results/` | Equity/drawdown curves, trade logs, `BacktestResult`, reporters |
| `sigmaloop/plugins/` | Registries and built-in registration |
| `sigmaloop/utils/` | Structured logging, money rounding, timezone handling |

---

## 3. Data model

### 3.1 Primitive types and conventions

Defined in `sigmaloop/types.py`.

| Alias | Underlying | Notes |
|---|---|---|
| `Symbol` | `NewType(str)` | Display ticker, e.g. `"SPY"`. Not unique across asset classes. |
| `InstrumentId` | `NewType(str)` | Canonical key. `"EQ:SPY"`, `"OPT:SPY:20250117:C:00500000"`. |
| `OrderId`, `IntentId`, `FillId`, `TradeId`, `RunId` | `NewType(str)` | Distinct id spaces, statically checked. |
| `Price`, `Quantity`, `Money` | `float` (float64) | See below. |
| `Percent` | `float` | Fractional: `0.075 == 7.5%`. |
| `Basis` | `float` | Basis points: `1.0 == 0.01%`. |
| `EpochNanos` | `int` | UTC nanoseconds — the columnar time representation. |
| `UtcDatetime` | `datetime` | Always timezone-aware, always UTC. |

**Why float64 and not `Decimal`.** The performance NFR requires vectorised
indicator evaluation over millions of bars and process-parallel parameter
sweeps; `Decimal` is roughly two orders of magnitude slower and cannot be held
in a numpy column. float64 carries ~15 significant digits, ample for simulated
cash. Rounding is applied deliberately at two boundaries — tick-snapping fill
prices, and formatting money for reports (`utils/money.py`) — and nowhere else,
because rounding mid-computation accumulates bias rather than removing it.

**Time.** Every `datetime` crossing a public API is tz-aware UTC; naive values
are rejected, not assumed. Bars are **right-labelled**: `Bar.timestamp` is the
instant the bar *closed*. This is what makes "the 2024-03-01 daily bar becomes
visible only after 2024-03-01's session ends" true by definition rather than by
discipline.

**Options quantities** are counted in **contracts**, never shares.
`Instrument.multiplier` (100 for standard equity options) converts to notional,
and `utils.money.notional()` is the single place that multiplication happens.

### 3.2 Enumerations

| Enum | Members (abridged) | Purpose |
|---|---|---|
| `AssetClass` | EQUITY, ETF, OPTION, INDEX, CASH, FUTURE\* | Margin/settlement/multiplier rules |
| `OptionRight` / `OptionStyle` | CALL, PUT / AMERICAN, EUROPEAN | Contract identity |
| `SettlementType` | PHYSICAL, CASH | Expiry resolution path |
| `Timeframe` | TICK, 1s … 1d, 1w, 1mo | Bar width; drives annualisation |
| `OrderSide` / `PositionSide` | BUY, SELL / LONG, SHORT, FLAT | |
| `OrderType` | MARKET, LIMIT, STOP, STOP_LIMIT, MOO, MOC | |
| `TimeInForce` | DAY, GTC, GTD, IOC, FOK | |
| `OrderStatus` | PENDING_NEW, ACCEPTED, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED | |
| `RejectReason` | INSUFFICIENT_CAPITAL, NO_MARKET_DATA, NOT_SHORTABLE, RISK_LIMIT_BREACHED, LIQUIDITY_CAP, … | Structured rejection cause |
| `ExecutionTiming` | **NEXT_BAR_OPEN** (default), NEXT_BAR_CLOSE, SAME_BAR_CLOSE | Signal→execution model |
| `PriceSelection` | MID, **WORST** (default), BEST, LAST | Which side of the spread |
| `StrategyMode` | SINGLE_ASSET, SINGLE_ASSET_OPTIONS, PORTFOLIO | Selects base class + data plan |
| `SizingMode` | FIXED_QUANTITY, FIXED_NOTIONAL, PERCENT_EQUITY, RISK_PERCENT, TARGET_WEIGHT, CUSTOM | Accounting req. 4 |
| `TradeCloseReason` | SIGNAL, STOP_LOSS, TAKE_PROFIT, OPTION_EXPIRY_WORTHLESS, OPTION_ASSIGNMENT, END_OF_BACKTEST, … | Trade-log attribution |
| `CorporateActionType` | SPLIT, CASH_DIVIDEND, DELISTING, … | |
| `MarginModel` | CASH, REG_T, PORTFOLIO\* | |
| `RunState` | CREATED, WARMING_UP, RUNNING, FINALISING, COMPLETED, FAILED, CANCELLED | |

\* reserved; not simulated in v1.

### 3.3 Instruments — `domain/instrument.py`

```
Instrument (ABC, frozen, slots)
├── instrument_id, symbol, asset_class, currency, exchange
├── multiplier, tick_size, lot_size
├── is_tradeable, is_shortable, borrow_rate_annual
├── notional(price, qty) · is_expired(as_of) · round_price · round_quantity
│
├── Equity          asset_class=EQUITY, multiplier=1, sector, listed/delisted dates
└── OptionContract  underlying_id, right, strike, expiry, style, settlement,
                    multiplier=100, occ_symbol
                    days_to_expiry() · moneyness() · intrinsic_value() · is_itm()
```

`InstrumentRegistry` interns instruments per run so there is exactly one shared
object per instrument. Identity comparisons stay cheap and the portfolio can
resolve a multiplier without calling back into a provider — which matters in
portfolio mode where the registry may hold hundreds of thousands of contracts.

Greeks deliberately do **not** live on `OptionContract`: they are time-varying
market data and belong on `OptionQuote`.

### 3.4 Market data — `domain/bar.py`

Two representations coexist by design:

**Row form** — one frozen object per observation, used in the engine loop.

```
Quote        bid, ask, bid_size, ask_size, is_synthetic
             .mid  .spread  .spread_pct

Bar          instrument_id, timestamp (bar CLOSE), OHLC, volume, timeframe,
             quote?, vwap?, trade_count?
             .typical_price  .range  .is_up
             .price_for(selection, is_buy) -> Price

Greeks       delta, gamma, theta, vega, rho, implied_volatility

OptionQuote  instrument_id, contract, timestamp, quote, last?, volume,
             open_interest, greeks?, underlying_price?
             .mid  .delta  .days_to_expiry  .is_zero_dte  .price_for(...)
```

Both `Bar` and `OptionQuote` satisfy the `PricedInstrument` protocol, which is
what lets one `FillPriceModel` handle equities and options without branching on
asset class.

**Columnar form** — `BarSeries`, a struct-of-arrays over preallocated numpy
buffers with an append cursor:

```
BarSeries    _ts:int64[]  _open/_high/_low/_close/_volume:float64[]
             append() amortised O(1) · tail(n) and slice() are zero-copy views
             from_arrays() adopts validated columns with no copy
```

Indicators vectorise over this; the engine loop uses row form. A daily
10-year history is 8 columns × ~2,500 float64 ≈ 160 KB, versus tens of megabytes
as Python objects — this is the substance of the memory-efficiency NFR.

**Aggregates:**

```
OptionChain     underlying_id, timestamp, underlying_price, quotes[]
                Coarse:  expiries() · strikes() · filter(right, dte, strike,
                         open_interest, spread) -> OptionChain   (chainable)
                Select:  by_delta(target, right)   ← "20-delta put"
                         by_strike() · by_moneyness() · atm() · nearest_expiry()
                Indexed by (expiry, right, strike): O(log n) scans, not O(n).

MarketSnapshot  timestamp, bars: {InstrumentId: Bar}, chains: {…: OptionChain},
                is_session_open/close
                A CLOSED WORLD: absence from `bars` means the instrument did not
                trade this step. Orders against it are rejected NO_MARKET_DATA
                rather than silently filled at a stale price.
```

### 3.5 Orders — `domain/order.py`

The order path is deliberately **two-stage**:

```
Strategy emits            Engine resolves                 Broker works
────────────────          ───────────────                 ────────────
OrderIntent      ──►  PositionSizer ──► RiskManager  ──►  Order  ──►  Fill
(what I want)         (how many)        (am I allowed)    (concrete)   (done)
```

```
SizingRequest   mode: SizingMode, value, sizer_name?, max_quantity?,
                allow_fractional
                The `value` field's meaning is selected by `mode` — shares,
                cash, equity fraction, risk fraction, or target weight.

BracketSpec     stop_loss_price/pct, take_profit_price/pct, trailing_stop_pct
                Offsets resolve against the realised entry fill.

OrderIntent     intent_id, instrument_id, side, sizing, order_type,
                limit/stop price, tif, expires_at, bracket?, reduce_only, tag

Order (mutable) order_id, instrument_id, side, quantity, order_type,
                submitted_at, activated_at, status, filled_quantity,
                avg_fill_price, commission_paid, rejection?, parent_order_id
                .remaining_quantity  .signed_quantity  .apply_fill()

Fill (frozen)   fill_id, order_id, instrument_id, timestamp, side, quantity,
                price, commission, fees, slippage_per_unit, reference_price,
                liquidity, is_partial

Rejection       reason: RejectReason, message, timestamp, required?, available?
```

`submitted_at` vs `activated_at` is the lookahead firewall made visible: under
the default execution model they differ by exactly one bar.

Splitting sizing out of the strategy is what makes Accounting requirement 4
("the user can express position sizing … rather than the engine assuming one")
a first-class concern instead of something each strategy re-implements.

### 3.6 Positions and trades — `domain/position.py`

```
Lot        quantity, price, opened_at, fill_id, commission
           Enables FIFO/LIFO realised-P&L matching, not just average cost.

Position   instrument, quantity (SIGNED: <0 == short), avg_price,
           realized_pnl, commission/fees/borrow/dividends, mark_price,
           lots: deque[Lot], max_favorable_price, max_adverse_price
           .side  .cost_basis  .market_value  .unrealized_pnl  .exposure
           .apply_fill(fill) -> realised P&L   (open / increase / reduce / flip)
           .mark(price, at)  · .accrue_borrow(...)

Trade      trade_id, instrument_id, symbol, asset_class, direction, quantity,
(frozen)   entry_time/price, exit_time/price,
           gross_pnl, commission, fees, net_pnl, return_pct,
           close_reason, mae, mfe, bars_held, tag
           → the row type of the trade log (Outputs req. 1)

OptionTrade(Trade)
           + underlying_symbol, right, strike, expiry, multiplier,
             dte_at_entry/exit, delta_at_entry, iv_at_entry/exit,
             underlying_price_at_entry/exit,
             was_assigned, was_exercised, expired_worthless
           → the options trade log (Outputs req. 2)
```

### 3.7 Account state — `domain/account.py`

```
EquityPoint     timestamp, cash, positions_value, equity,
(frozen)        gross/net_exposure, margin_used, buying_power,
                open_positions, realized_pnl_cum, unrealized_pnl,
                drawdown, high_water_mark   → one per bar (Accounting req. 1)

AccountState    initial_cash, cash, currency, margin_used, reserved_cash,
(mutable)       realized_pnl, total_commission/fees/borrow/dividends,
                high_water_mark
                .available_cash  .buying_power(...)  .can_afford(...)

CashFlow        timestamp, amount, reason, instrument_id?
                Non-trade cash: dividends, borrow, interest. Kept distinct from
                Fill so P&L attribution can separate market return from carry.

CorporateAction action_type, instrument_id, ex_date, ratio, amount, new_symbol
                Applied at the top of the bar, before any trading.
```

---

## 4. Data layer

### 4.1 Providers — `data/provider.py`

```
DataProvider (ABC)
  capabilities -> ProviderCapabilities
  resolve_instrument(symbol, asset_class) -> Instrument
  available_symbols(asset_class) -> [Symbol]
  stream_bars(DataRequest)  -> Iterator[Bar]     ← lazy, memory-bounded
  load_series(symbol, req)  -> BarSeries         ← eager, columnar
  load_many(req) -> {Symbol: BarSeries}
  corporate_actions(symbol, start, end)
  open() / close() / context-manager protocol

OptionsDataProvider(DataProvider)
  get_chain(underlying, as_of) -> OptionChain
  stream_chains(OptionChainRequest) -> Iterator[OptionChain]
  resolve_contract(underlying, expiry, right, strike) -> OptionContract
  expirations(...) · settlement_price(contract)

CompositeDataProvider(OptionsDataProvider)
  Routes by asset class — e.g. equities from CSV, chains from Polygon —
  behind one provider-shaped object.
```

Every provider supplies **both** access paths. Streaming is what large runs use
(NFR 3); eager columnar loading is what indicator warm-up and screeners use.

`ProviderCapabilities` (asset classes, timeframes, options?, greeks?, quotes?,
credentials?) is declared up front so config validation turns "provider silently
returns nothing" into a clear error before the run starts.

**Hard contract:** `stream_bars` must yield in non-decreasing timestamp order.
`DataFeed` k-way merges provider streams and depends on it.

| Implementation | Data req. | Serves | Notes |
|---|---|---|---|
| `CsvDataProvider` | 1a | OHLCV, optional quotes | Configurable `CsvColumnMap`; wide (one file per ticker) or long layout; timezone conversion; optional Parquet cache |
| `YahooDataProvider` | 1b | OHLCV, corporate actions | No bid/ask → requires a `SpreadModel`; no symbol enumeration; heavily cached |
| `PolygonDataProvider` | 1c | OHLCV, NBBO quotes, chains, greeks | The reference `OptionsDataProvider`; pushes chain filters into the API query |

### 4.2 Feed — `data/feed.py`

`DataFeed` is the boundary that turns N per-symbol provider streams into one
chronological sequence of `MarketSnapshot`. Everything downstream sees exactly
one timestamp at a time.

```
FeedPlan          Resolved, validated description of what the run will read:
                  bar_requests, chain_requests, timeframe, warmup_bars.
                  Built once, checked against provider capabilities, so a
                  missing symbol fails at second zero, not on bar 40,000.

DataFeed (ABC)    __iter__ -> Iterator[MarketSnapshot] (strictly increasing)
                  prepare() · close() · instruments() · current
                  corporate_actions_at(ts) · add_instrument(...) (mid-run)

MergedDataFeed    Default. Binary-heap k-way merge keyed on
                  (epoch_ns, instrument_id).
                  Cost: O(total_bars · log n_symbols) time,
                        O(n_symbols) resident memory — independent of
                        history length. This is NFR 3.

PrefetchDataFeed  Bounded background read-ahead; overlaps provider I/O with
                  strategy computation without unbounding memory.

ReplayDataFeed    In-memory, fixture-driven. Tests only.

HistoryWindow     Read-only view of bars already seen for one instrument.
                  Handed to strategies as ctx.history(id) — physically cannot
                  index into the future.
```

`add_instrument` exists because both options mode (a strategy selects a contract
that was not in the initial plan) and portfolio mode (a screener admits a new
name) need mid-run subscription.

### 4.3 Universe — `data/universe.py`

Portfolio strategies name criteria, not tickers.

```
Screen (ABC)          passes(symbol, snapshot, context) -> bool
                      Composable: `liquid & breakout`, `~excluded`
  LiquidityScreen     min price, min average dollar volume
  CallableScreen      wraps a user function

Universe (ABC)        resolve(snapshot, context) -> [Symbol]   (point-in-time)
                      candidate_symbols(start, end) -> [Symbol] (superset, for
                                                       up-front data loading)
                      should_rebalance(snapshot, bar_index)
  StaticUniverse      fixed list
  ScreenedUniverse    candidates narrowed by an ordered screen chain
  RankedUniverse      screen → score → top-N or top-X%   ← "top 10% gainers"
```

**Point-in-time correctness is the whole job.** A universe must answer "what was
investable *as of* this timestamp", never "what is investable today". Ignoring
that is survivorship bias, and it silently inflates every downstream metric.
`candidate_symbols()` is separate from `resolve()` precisely so data loading can
be planned up front while membership still varies per bar.

### 4.4 Cache and calendar

`DataCache` is a two-tier store (`MemoryDataCache` LRU bounded by *bytes*, over
`ParquetDataCache` on disk, composed by `TieredDataCache`). Sweeps and
walk-forward folds re-read the same history dozens of times; without a cache,
wall-clock is dominated by re-parsing. `CacheKey` includes provider name,
symbol, range, timeframe, adjustment flag and schema version — conservative,
because a key collision would silently serve wrong data. Disk writes are atomic
(temp + rename) so a killed run cannot leave a truncated file behind.

`TradingCalendar` (`NyseCalendar`, `ContinuousCalendar`) supplies session
boundaries, holiday handling, `is_session_close` (needed for MOC orders, EOD
liquidation and option expiry) and `sessions_per_year` / `year_fraction` — the
denominators for CAGR, Sharpe annualisation and borrow accrual. Hardcoding 252
would break the first time someone runs an hourly backtest.

---

## 5. Indicators — `indicators/`

Functional requirement 2: a first-class interface for custom indicators.

```
IndicatorSpec       Declarative request: name, params, instrument_id?, alias
                    Strategies DECLARE these; the engine instantiates and
                    updates them.

Indicator[T] (ABC)
  warmup_period -> int
  update(bar) -> T | None      ← O(1) incremental, used in the loop
  compute(series) -> float64[] ← vectorised, used for warm-up and screeners
  value · is_ready · dependencies · reset() · params

RollingIndicator(Indicator[float])
  Fixed-window base: owns the deque and warm-up bookkeeping; subclasses
  implement _aggregate(). Maintains O(1) running state where one exists.

CompositeIndicator(Indicator[T])
  Built from child indicators (MACD = EMA − EMA). Children update first;
  warmup = max(children) + own smoothing.

IndicatorSet
  All instances for one run, keyed by (instrument, alias). Owns update
  ordering (dependencies first) and the ctx.indicator("sma_20") lookup.
```

**Why both `update` and `compute`.** Recomputing an O(window) indicator per bar
per symbol dominates runtime in portfolio mode; vectorising the whole history in
a Python loop is equally bad. Requiring both is a deliberate cost on indicator
authors, mitigated by `RollingIndicator` supplying the incremental half and
`Indicator.compute` having a correct-if-slow default that replays `update`.

**Why the engine owns updates.** Declaring rather than constructing means the
engine knows the total warm-up requirement *before* loading data, so it can
prepend exactly enough history — and it guarantees each indicator sees each bar
exactly once, which is what makes runs reproducible.

Built-ins: `sma`, `ema`, `stddev`, `rsi`, `atr`, `bbands`, `macd`,
`rolling_high`, `rolling_low`, `roc`.

---

## 6. Execution — `execution/`

### 6.1 The lookahead firewall (Execution req. 2)

A signal computed from bar *t*'s close cannot transact at bar *t*'s open,
because that price was already in the past when the signal existed.
`NextBarOpenExecutionModel` — the default — enforces this by construction: an
order submitted during bar *t* becomes eligible only at bar *t+1*.

```
ExecutionModel (ABC)
  timing -> ExecutionTiming
  introduces_lookahead -> bool
  is_eligible(context) -> bool        ← "may this order be considered yet?"
  try_fill(context) -> FillDecision   ← "does this bar trigger it, at what price?"
  should_expire(context) -> bool      ← time-in-force

  NextBarOpenExecutionModel   DEFAULT, lookahead-free
  NextBarCloseExecutionModel  VWAP/TWAP-like working order
  SameBarCloseExecutionModel  LOOKAHEAD-PRONE; comparison only
```

Limit/stop triggering against a bar:

| Order | Triggers when | Fill price |
|---|---|---|
| Buy limit | `low <= limit` | `min(open, limit)` |
| Sell limit | `high >= limit` | `max(open, limit)` |
| Buy stop | `high >= stop` | `max(open, stop)` |
| Sell stop | `low <= stop` | `min(open, stop)` |

**Gap handling matters.** When a bar opens *through* the level, the fill uses
the open, not the level. Assuming the limit price on a gap is the single most
common way backtests manufacture free money.

The intrabar path is unknown, so a bar touching both a stop and a target is
resolved **pessimistically**: the stop is assumed to trigger first.

Models with `introduces_lookahead == True` push a prominent entry into
`BacktestResult.warnings`, so a tainted result can never be mistaken for a clean
one.

### 6.2 Price selection and synthetic spreads (Execution req. 1)

Two separable concerns:

```
FillPriceModel (ABC)  resolve(PricingContext) -> Price
  QuoteFillPriceModel   Prices from bid/ask.
                        MID   → mid
                        WORST → ask when buying, bid when selling   (DEFAULT)
                        BEST  → the inverse
                        LAST  → close, ignoring the spread
  OhlcFillPriceModel    A chosen OHLC field; only valid with LAST.

SpreadModel (ABC)     quote_for(bar, instrument) -> Quote
  FixedBpsSpreadModel   constant bps (per asset class)
  TickSpreadModel       N ticks — for low-priced names where bps < 1 tick
  VolatilitySpreadModel scaled by realised range — wider in stress
```

CSV and Yahoo publish OHLCV only. Without a synthetic spread, `WORST` would
silently collapse to `MID` and quietly overstate returns. Every synthesised
quote is flagged (`Quote.is_synthetic`) and the count of fills that relied on
one is reported in `RunSummaryStats.synthetic_quote_fills`.

### 6.3 Slippage and commissions (Execution req. 3)

```
SlippageModel (ABC)  slippage_per_unit(ctx) -> Price   (always adverse)
                     fillable_quantity(ctx) -> Quantity (participation cap)
  NoSlippageModel · FixedBpsSlippageModel · TickSlippageModel
  VolumeShareSlippageModel      impact = k · price · sqrt(qty / bar_volume),
                                plus a max-participation cap. Keeps portfolio
                                results honest on small caps.
  SpreadFractionSlippageModel   a fraction of the prevailing spread — the
                                natural model for options.

CommissionModel (ABC)  commission(ctx) -> Money · fees(ctx) -> Money
  ZeroCommissionModel · PerShareCommissionModel (floor + cap)
  PerTradeCommissionModel · PercentValueCommissionModel
  PerContractCommissionModel (options; optional close-waiver)
  RegulatoryFeeModel   SEC §31, FINRA TAF, options ORF — sell-side only,
                       which is why fees are modelled apart from commission
  TieredCommissionModel · CompositeCommissionModel (sums children)
```

The realistic default is `Composite(PerShare, Regulatory)`. Slippage is kept
separate from price selection so the trade log can attribute both: it records
the reference price *and* the slippage applied to it.

### 6.4 Broker — `execution/broker.py`

```
SimulatedBroker.process_bar(snapshot) -> BrokerResult
  1. expire orders whose TIF lapsed
  2. for each working order: is_eligible? → try_fill?
  3. reference price → slippage → commission → Fill
  4. activate bracket children once the parent fills (OCO siblings)
```

Working orders are indexed by instrument, so a bar walks only the orders that
could fill on it — O(orders on this instrument), not O(all open orders). In
portfolio mode with thousands of resting brackets, that difference is the run.

**The broker never touches the ledger.** It emits fills; the `Portfolio` applies
them. Keeping matching and accounting apart makes both independently testable
and is why a rejected order can never leave a half-applied cash effect behind.

### 6.5 Option expiry — `execution/expiry.py`

Options positions do not vanish at expiry; they resolve into cash, into shares,
or into nothing. Getting this wrong is the largest single source of error in
options backtests, so it has a dedicated component.

| Held to expiry | Outcome |
|---|---|
| OTM | Expires worthless; premium is the full P&L |
| ITM long, PHYSICAL | Exercised — long call buys at strike, long put sells at strike |
| ITM short, PHYSICAL | Assigned (per `expiry_assignment_probability`) |
| ITM, CASH settled | Intrinsic value credited/debited |

`ExpiryPolicy` also supports closing N bars *before* expiry (avoids modelling
assignment, at a cost in realism) and a per-bar early-assignment probability for
short ITM American options. All draws come from the run's seeded RNG, so an
identical config reproduces identical results — a hard requirement for sweep
points to be comparable.

`ExpiryOutcome` carries the cash impact, any `underlying_quantity_delta` from
physical settlement, and synthetic fills so the trade log shows the full round
trip.

---

## 7. Portfolio — `portfolio/`

### 7.1 Ledger — `accounting.py`

```
PortfolioView (ABC)   READ-ONLY surface handed to strategies
   cash · equity · buying_power · positions_value
   gross_exposure · net_exposure · realized_pnl · unrealized_pnl
   position(id) · open_positions() · quantity(id) · weight(id) · closed_trades()

Portfolio(PortfolioView)   MUTABLE; owned by the engine
   can_afford(order, price) -> Rejection | None      ← Accounting req. 2
   reserve(order, cost) / release(order)
   apply_fill(fill, at) -> Trade | None
   apply_expiry(outcome, at) -> [Trade]
   apply_corporate_action(action, at) · apply_cash_flow(flow)
   mark_to_market(snapshot) -> EquityPoint           ← Accounting req. 1
   liquidate_all(snapshot, reason) -> [Trade]
   validate_invariants()

LedgerPortfolio(Portfolio)   reference implementation
```

Strategies receive `PortfolioView`, never `Portfolio`. User code can observe the
ledger and emit intents; it cannot corrupt state.

**Invariant, checked after every mutation:**
`equity == cash + Σ position.market_value`. A breach raises `AccountingError`
immediately — drift discovered at the end of a run is unattributable.

Cash reservation (`reserve`/`release`) exists so two orders raised on the same
bar cannot both spend the same dollar.

Positions live in a dict keyed by `InstrumentId`; per-bar cost is proportional
to the number of *touched* instruments, not the number held.

### 7.2 Sizing — `sizing.py` (Accounting req. 4)

```
PositionSizer (ABC)   size(SizingContext) -> Quantity
                      apply_constraints(qty, ctx)   ← shared clamping

  FixedQuantitySizer   exactly N units
  FixedNotionalSizer   value / (price · multiplier)
  PercentEquitySizer   equity · pct / notional_per_unit   (compounds)
  RiskPercentSizer     equity · risk_pct / (|entry − stop| · multiplier)
  TargetWeightSizer    trades only the weight delta — the rebalance primitive
  CallableSizer        arbitrary user function — the "custom rule" escape hatch
  CompositeSizer       dispatches on intent.sizing.mode  ← what the engine installs
```

All sizers return a non-negative quantity (direction lives on the order side), a
valid lot multiple, and **contracts** for options. `CompositeSizer` is what the
engine installs, so one strategy can mix percent-of-equity entries with
fixed-quantity hedges.

Sizing runs *before* the fill price is knowable (next-bar-open), so it is
intentionally approximate against the last close; the broker re-checks capital
at fill time.

### 7.3 Risk — `risk.py`

```
RiskCheck (ABC)  check(RiskContext) -> Rejection | None
  CapitalCheck · ShortingCheck · ConcentrationCheck
  LeverageCheck · MaxPositionsCheck

RiskManager      Runs checks cheapest-first, short-circuits on first veto.
                 `flag_only` mode records breaches and lets orders through, so
                 a run can quantify how often a strategy overreaches without
                 truncating its behaviour  ← the "reject OR flag" in req. 2.

MarginCalculator (ABC) · RegTMarginCalculator
                 50% initial / 25% maintenance on equities. Long options paid in
                 full; short options use the naked-option formula — which is why
                 short-premium strategies consume far more buying power than
                 their premium suggests.
```

Most checks skip position-*reducing* orders: closing risk is not new risk, and
blocking an exit is worse than allowing it.

---

## 8. Strategy layer — `strategy/`

### 8.1 Base classes (Functional req. 5)

> *"There can be different Strategy bases for different modes but a single
> engine should run all."*

Achieved by keeping the engine's contract narrow — `Strategy.handle_bar` — and
letting each base class adapt it:

```
Strategy (ABC)                          ← engine-facing; users rarely subclass
  mode · param_spec() · declare_indicators() · declare_instruments()
  handle_bar(ctx)                       ← THE ONLY METHOD THE ENGINE CALLS
  on_start · on_finish
  on_fill · on_order_rejected · on_trade_closed · on_option_expiry
  reset()

├── SingleAssetStrategy      symbol; on_bar(ctx, bar)
│   └── OptionsStrategy      min_dte/max_dte/strike_window_pct/require_greeks;
│                            on_bar(ctx, bar, chain)
└── PortfolioStrategy        rebalance_frequency, max_positions;
                             select_universe(ctx) -> Universe | [Symbol]
                             on_bar(ctx, snapshot)
```

Adding a fourth mode later means adding a base class, not touching the loop.

**Lifecycle:**

```
__init__(params) → param_spec() → declare_indicators() → declare_instruments()
 → on_start(ctx)
 → per bar: [on_fill / on_order_rejected / on_trade_closed / on_option_expiry]
            then handle_bar(ctx)
 → on_finish(ctx)
```

Event callbacks fire **before** `handle_bar` for the bar in which they occurred,
so the strategy reacts to yesterday's fills with today's data — the same
information ordering a live system would see.

Strategies must be stateless across runs: `reset()` is called between sweep
points and walk-forward folds, and state not cleared there leaks between runs
and corrupts comparisons.

### 8.2 `StrategyContext` — the only surface user code touches

A strategy never holds a reference to the engine, portfolio, broker or feed. It
gets a context (`strategy/context.py`, implemented by `engine/context.py`):

| Group | Members |
|---|---|
| Clock | `now`, `bar_index`, `is_warmup`, `is_last_bar` |
| Market | `snapshot`, `bar(id)`, `price(id)`, `history(id)`, `instrument(id)`, `resolve(symbol)` |
| Indicators | `indicator(alias)`, `indicator_value(alias)` |
| Options | `chain(underlying)`, `subscribe_option(contract)` |
| Account | `portfolio` (read-only view), `position(id)`, `positions()`, `working_orders()` |
| Orders | `submit(intent)`, `cancel`, `cancel_all`, `buy`, `sell`, `close`, `close_all`, `order_target_percent`, `rebalance(weights)` |
| Diagnostics | `log(msg, **fields)`, `record(name, value)`, `warn(msg)` |

Two consequences: user code cannot mutate the ledger or reach forward in the
data (every timestamped accessor routes through the clock's lookahead guard),
and the engine can hand a strategy a different context per run or per
walk-forward fold without the strategy noticing.

### 8.3 Parameters (Functional req. 3)

```
Parameter        name, default, description, min/max | choices,
                 value_type, sweep_values
ParameterSpec    the declared surface; .resolve(overrides) -> ParameterSet
                 .grid() -> Iterator[ParamDict]   ← the optimisation hook
ParameterSet     resolved + validated values; mapping AND attribute access
                 .fingerprint() → part of the run id
```

Declaring parameters buys: validation before the run (a bad value is a config
error at second zero); reproducibility (the resolved set is hashed into the run
id, so two results are comparable only if they truly used the same inputs); and
sweepability. Unknown parameter names are an **error**, not a silent no-op —
a typo'd parameter is otherwise invisible.

`sweep_values` is opt-in per parameter, so adding a parameter never silently
multiplies the search space.

### 8.4 Order API — `strategy/api.py`

`StrategyContext` exposes the common cases. The full builder surface lives here:

```
OrderBuilder      fluent: .quantity/.notional/.percent_equity/.risk_percent
                          .limit/.stop/.bracket/.time_in_force/.tag → .build()

OptionLeg         right, side, ratio, and ONE selector
                  (target_delta | strike | moneyness), dte
OptionStructure   named multi-leg structures, resolved against a chain:
                    covered_call() · cash_secured_put() · straddle()
                    strangle()      ← "SPY 0DTE 20-delta put and call"
                    vertical_spread()
                  Resolution is ATOMIC: if any leg is missing from the chain,
                  the whole structure is skipped with a warning, rather than
                  legging into a partial position nobody asked for.

StrategyApi (ABC) order() · submit_intent() · submit_structure()
                  target_weights() · default_sizing()
```

`ratio` is relative, not absolute: the structure is sized once and each leg is
`structure_quantity · ratio`, which keeps a 1×2 ratio spread correct under any
sizing mode.

---

## 9. Engine — `engine/`

### 9.1 The bar loop — `engine/core.py`

**Fixed per-bar phase order.** This ordering *is* the semantics of the engine;
it is why results are reproducible and why lookahead is impossible.

```
 1. clock.advance(snapshot.timestamp)
 2. apply corporate actions with this ex-date
 3. broker.process_bar(snapshot)              → fills, expirations, cancels
 4. portfolio.apply_fill(...) for each        → realised trades
 5. expiry engine: settle expiries, assignments
 6. accrue carry (short borrow, cash interest)
 7. portfolio.mark_to_market(snapshot)        → EquityPoint    ← Accounting req. 1
 8. indicators.update_all(bar) for each bar in the snapshot
 9. deliver on_fill / on_order_rejected / on_trade_closed / on_option_expiry
10. [portfolio mode] re-resolve universe if this is a rebalance bar
11. strategy.handle_bar(context)              → OrderIntents
12. size → risk-check → broker.submit(order)  (works from the NEXT bar)
13. record equity point, emit BarClosed
```

Two properties fall out of this ordering:

* **Steps 3 and 11 are the firewall.** Orders raised at step 12 of bar *t*
  cannot fill before step 3 of bar *t+1*.
* **Step 7 precedes step 11**, so a strategy reading `ctx.portfolio.equity` sees
  *this* bar's true value, not the previous bar's.

Warm-up bars run steps 1–9 only; no orders are accepted until every declared
indicator is ready.

```
BacktestEngine
  __init__(config, strategy, feed?, events?, cancellation?)
  prepare()   validate config → resolve plugins → read strategy declarations
              → build FeedPlan → load warm-up data
  run()       → BacktestResult
  step(snapshot)     one bar through the 13 phases (test seam)
  finalise()  liquidate if configured → compute metrics → assemble result
  subscribe(listener) · state · run_id · context

CancellationToken   cooperative, checked once per bar
```

`run()` **always** produces a result. On failure it carries `RunState.FAILED`,
the error and traceback, and every bar completed up to that point — a partial
equity curve is far more diagnostic than a bare traceback.

`prepare`/`run` are split so a sweep worker can validate a whole batch of
configs before committing to any I/O.

### 9.2 Configuration — `engine/config.py`

One immutable, serialisable `BacktestConfig` fully determines a run.

```
BacktestConfig
  strategy_mode, start, end, symbols, universe?, strategy_params,
  seed, run_id?, tags
  ├── DataConfig        providers[], timeframe, warmup_bars?, adjustments,
  │                     cache dir/size, prefetch, calendar, strict_validation
  ├── ExecutionConfig   timing, price_selection, execution/spread/slippage
  │                     models, commission_models[], max_volume_participation,
  │                     partial-fill policy
  ├── AccountingConfig  initial_cash, default sizing mode+value, allow_short,
  │                     fractional shares, margin model, leverage/position caps,
  │                     on_capital_breach ("reject" | "flag"), lot_matching,
  │                     liquidate_at_end, cash_interest_rate
  ├── OptionsConfig     dte/strike/greeks filters, quote quality gates,
  │                     expiry + assignment policy
  ├── ReportingConfig   output dir, reporters[], benchmark, risk_free_rate,
  │                     what to persist, log level
  └── ParallelConfig    enabled, max_workers, executor, chunk_size, fail_fast

  .validate() -> [problems]     cross-cutting checks (§ below)
  .fingerprint() -> str          identity; hashed into the run id
  .derive(**overrides)           the sweep primitive
  .from_file(path) / .to_dict()
```

`validate()` catches the cross-cutting cases that `__post_init__` cannot,
because they need the plugin registry: options mode wired to an equity-only
provider; `WORST` pricing with a quote-less provider and no spread model; a
percent-equity default outside `(0, 1]`; an unknown plugin name. Failing here
costs milliseconds; failing mid-run costs the whole run.

### 9.3 Clock, context and events

`SimulationClock` is the **single** source of "now". Nothing calls
`datetime.now()` — a backtest has no wall-clock present, and mixing the two is
how "it worked in backtest" bugs are born. It also owns
`assert_visible(timestamp)`, the lookahead guard every context accessor routes
through, and `bars_per_year()`, the annualisation factor derived from timeframe
+ calendar.

`RunContext` holds everything scoped to one run — clock, registry, feed,
portfolio, broker, indicators, sizer, risk manager, event bus, diagnostics — and
*is* the concrete `StrategyContext`, seen by strategies through that narrower
interface.

`RunDiagnostics` accumulates the counters that qualify a result: warnings,
`synthetic_quote_fills`, `stale_marks`, `missing_bars`, `capital_breaches`,
partial fills, plus `ctx.record` series. `warn_once` deduplicates — a per-bar
caveat must not emit 40,000 lines.

The **event system is for observation, not control flow**. A queue-based engine
would cost an allocation and a dispatch per event per bar; instead `EventBus`
fans out synchronously with a no-listener fast path, so instrumentation costs
one dict lookup per phase when nobody is subscribed. Listener exceptions are
logged and swallowed — an observer must never be able to fail a run.

Events: `RunStarted`, `BarOpened`, `CorporateActionApplied`, `OrderFilled`,
`OrderRejected`, `OptionExpired`, `TradeClosed`, `EquityUpdated`,
`OrderSubmitted`, `BarClosed`, `RunFinished`.

### 9.4 Parallelism — `engine/runner.py` (NFR 2)

**Where the parallelism is not:** inside a run. A bar loop is a sequential
dependency chain — bar *t*'s ledger is bar *t−1*'s output — so splitting it
would require locking the portfolio on every fill and would run *slower*.

**Where it is:** across runs, which are embarrassingly parallel — one per
parameter set (sweeps, walk-forward folds) or one per symbol (single-asset
strategies over a watchlist).

```
RunSpec       run_id, config, strategy_CLASS, params, label
              Class, not instance → picklable, and each worker builds fresh,
              uncontaminated state.
RunOutcome    spec, result?, error?, duration — failures are VALUES, so one bad
              parameter combination cannot destroy a 500-point sweep.
BatchResult   outcomes[], .successful() · .failed() · .best_by(metric)
              · .to_frame()  (one row per run: params + every metric)

Executor (ABC) → SerialExecutor | ProcessExecutor | ThreadExecutor
BacktestRunner  run_one() · run_batch() · run_symbols() · run_sweep()
```

Processes, not threads: the loop is CPU-bound pure Python and the GIL would
serialise threads. The cost is that configs, strategy classes and results must
be picklable — which is why `BacktestConfig` is a plain frozen dataclass. The
one shared cost is data loading, amortised by the on-disk Parquet cache: the
first worker to need a series pays for it, the rest read the cached copy.

`run_sweep` is the seam the future parameter-optimisation and walk-forward
features attach to — the fan-out already exists; only the search policy is
missing.

---

## 10. Metrics and outputs

### 10.1 Metrics — `metrics/performance.py`

Two families from two inputs:

* **Return-based** (Sharpe, CAGR, drawdown, volatility) — from the equity
  curve. These describe the *account*.
* **Trade-based** (win rate, expectancy, profit factor) — from the trade log.
  These describe the *strategy*.

They can disagree — a 70% win rate is compatible with a falling equity curve —
and reporting both side by side is the point.

```
MetricCalculator (ABC)  compute(MetricContext) -> {field: value}
  ReturnMetricsCalculator     net profit, total return %, CAGR, volatility,
                              Sharpe, Sortino
  DrawdownMetricsCalculator   max DD depth/dates/duration, recovery factor,
                              Calmar, ulcer index  (vectorised running max, O(n))
  TradeMetricsCalculator      win rate, EXPECTANCY, profit factor, payoff,
                              streaks, holding period
  RiskMetricsCalculator       VaR/CVaR 95, skew, kurtosis, best/worst bar
  BenchmarkMetricsCalculator  alpha, beta, information ratio, correlation

MetricsEngine   registers calculators, assembles PerformanceMetrics
```

Calculators are *grouped* rather than one-per-metric because most share
intermediate work (a return series, a drawdown series); recomputing that per
metric would be wasteful and error-prone.

Every `PerformanceMetrics` field defaults to `None` where a short run cannot
support it — an infinite Sharpe from three bars is worse than no Sharpe.
Annualisation always comes from `clock.bars_per_year()`, never a hardcoded 252.

Requirement coverage: *Net profit, Total Return %, CAGR, Sharpe, Expectancy* —
all present, plus Sortino, Calmar, max drawdown (depth/duration/dates), profit
factor, payoff ratio, win rate, exposure, turnover, cost breakdown, VaR/CVaR and
benchmark-relative stats.

### 10.2 Outputs — `results/`

| Requirement | Type |
|---|---|
| Trade log | `TradeLog` — filter/`by_symbol`/`by_tag`, `to_frame`, `to_csv` |
| Options trade log | `OptionTradeLog` — `filter_options`, `by_underlying`, `by_dte_bucket`, `total_premium`, assignment counts |
| Equity and drawdown curve | `EquityCurve` (columnar, one row per bar) → `DrawdownCurve` with `DrawdownPeriod` decomposition |
| Human-readable summary | `TextReporter.render(result)` |

```
EquityCurve     columnar float64 buffers, one row per bar
                .returns() · .log_returns() · .high_water_mark()
                · .drawdown_curve() · .resample(period) · .to_frame/.to_csv
DrawdownCurve   underwater series + .periods() (worst first) + .longest_period()
DrawdownPeriod  peak_at, trough_at, recovered_at?, depth_pct, duration
                `recovered_at is None` for a drawdown still open at run end —
                reported explicitly, not silently treated as recovered.

BacktestResult  run_id, state, CONFIG, strategy_name, PARAMS, start, end,
                metrics, equity_curve, drawdown_curve,
                trades, option_trades, orders?, fills?, cash_flows,
                stats: RunSummaryStats, WARNINGS, logs, recorded, error?
                .summary() · .save(dir) · .to_dict() · .compare(other)
```

A result is **self-describing**: it carries the config and parameters that
produced it plus the diagnostics that qualify it. A result handed to someone
else should not need a verbal footnote to be interpretable.

`RunSummaryStats` describes the *simulation's fidelity* rather than the
strategy's performance: bars processed, orders submitted/filled/rejected,
partial fills, **synthetic-quote fills**, **stale marks**, capital breaches,
option expiries and assignments, wall clock and bars/second.

Reporters: `TextReporter`, `CsvReporter`, `JsonReporter`, `HtmlReporter`
(tearsheet), `CompositeReporter`. The text summary puts **warnings second, right
after the header** — a caveat printed below the numbers is a caveat nobody
reads.

---

## 11. Plugin architecture — `plugins/` (NFR 4)

Every swappable component is looked up by name:

| Registry | Entry-point group |
|---|---|
| `DATA_PROVIDERS` | `sigmaloop.data_providers` |
| `INDICATORS` | `sigmaloop.indicators` |
| `EXECUTION_MODELS` | `sigmaloop.execution_models` |
| `SPREAD_MODELS` / `SLIPPAGE_MODELS` / `COMMISSION_MODELS` | `sigmaloop.*_models` |
| `POSITION_SIZERS` / `RISK_CHECKS` | `sigmaloop.position_sizers` / `.risk_checks` |
| `METRIC_CALCULATORS` / `REPORTERS` / `CALENDARS` | `sigmaloop.metrics` / `.reporters` / `.calendars` |

Registration is by decorator (`@register(DATA_PROVIDERS)`) for built-ins, or by
entry point for third-party packages — discovered lazily on first lookup, so
import cost is paid only if used. `bootstrap()` registers the built-ins and is
idempotent, because each subprocess in a parallel sweep re-imports the package.

Lookups fail **loudly**: an unknown name raises `PluginNotFoundError` listing
what *is* available. A silent fallback to a default provider would produce a
plausible but wrong backtest.

Because configs reference plugins by string, they stay serialisable, diffable
and hashable into the run fingerprint.

---

## 12. Errors and logging (NFR 6)

```
SigmaLoopError
├── ConfigurationError · ValidationError
├── PluginError → PluginNotFoundError · DuplicatePluginError
├── DataError → DataProviderError · DataNotAvailableError
│               · InstrumentNotFoundError · OptionChainUnavailableError
│               · DataIntegrityError · LookaheadViolationError
├── StrategyError → StrategyContractError
├── IndicatorError → InsufficientHistoryError
├── ExecutionError → OrderRejectedError
├── AccountingError → InsufficientCapitalError · PositionNotFoundError
├── MetricError
└── EngineError → EngineStateError · RunCancelledError
```

Every exception carries a structured `context` dict alongside its message, so an
error states *what* failed, *where in the run*, and *what to do about it*. User
strategy exceptions are wrapped in `StrategyError` with the bar timestamp
attached and `__cause__` preserved, so a failure on bar 40,000 is locatable.

`RunLogger` (`utils/logging.py`) stamps records with **simulation** time, not
wall-clock time, and formats lazily behind a level check — an unconditional
f-string in a million-iteration loop is a measurable cost. Records go both to
the standard `logging` hierarchy and into the run's diagnostics, bounded by
`max_records` with overflow counted rather than stored.

---

## 13. Requirements traceability

### Functional

| # | Requirement | Where |
|---|---|---|
| 1 | Three modes | `StrategyMode`; `SingleAssetStrategy` / `OptionsStrategy` / `PortfolioStrategy`; §8.1 |
| 2 | Custom indicator interface | `indicators/base.py`; `IndicatorSpec` declaration; `INDICATORS` registry; §5 |
| 3 | Parameterised strategies | `Parameter` / `ParameterSpec` / `ParameterSet`; `.grid()`; §8.3 |
| 4 | Python only | Pure-Python package, `>=3.11` |
| 5 | Different bases, one engine | Narrow `Strategy.handle_bar` contract; §8.1, §9.1 |

### Data

| # | Requirement | Where |
|---|---|---|
| 1a | CSV | `CsvDataProvider` + `CsvColumnMap` |
| 1b | Yahoo Finance | `YahooDataProvider` (+ mandatory `SpreadModel`) |
| 1c | API (Polygon) | `PolygonDataProvider` (`OptionsDataProvider`) |

### Execution

| # | Requirement | Where |
|---|---|---|
| 1 | Mid or worst price | `PriceSelection`; `QuoteFillPriceModel`; `SpreadModel` for quote-less feeds; §6.2 |
| 2 | Next-bar-open default, no lookahead | `NextBarOpenExecutionModel`; phase order §9.1; `Clock.assert_visible` |
| 3 | Configurable commissions and fees | `CommissionModel` family + `CompositeCommissionModel`; §6.3 |

### Accounting

| # | Requirement | Where |
|---|---|---|
| 1 | Cash/positions/equity every bar | `Portfolio.mark_to_market` at phase 7 → `EquityPoint` → `EquityCurve` |
| 2 | Reject **or flag** over-capital orders | `CapitalCheck` + `RiskManager(flag_only)` + `AccountingConfig.on_capital_breach` |
| 3 | Long and short, equity and options | Signed `Position.quantity`; `ShortingCheck`; `RegTMarginCalculator`; `ExpiryEngine` |
| 4 | User-expressed sizing | `SizingRequest` + the `PositionSizer` family incl. `CallableSizer`; §7.2 |

### Metrics and outputs

| Requirement | Where |
|---|---|
| Net profit, Total Return %, CAGR, Sharpe, Expectancy, … | `PerformanceMetrics`; §10.1 |
| Trade log | `TradeLog` |
| Options trade log | `OptionTradeLog` |
| Equity and drawdown curve | `EquityCurve`, `DrawdownCurve` |
| Human-readable summary | `TextReporter` / `BacktestResult.summary()` |

### Non-functional

| # | Requirement | Where |
|---|---|---|
| 1 | Performance / data structures | Columnar `BarSeries` & `EquityCurve`; O(1) incremental indicators; heap-merged feed; per-instrument order index; interned registry; two-tier cache |
| 2 | Parallel across symbols and parameter sets | `BacktestRunner` + `ProcessExecutor`; §9.4 |
| 3 | Memory-efficient streaming | `stream_bars` iterators; `MergedDataFeed` O(n_symbols) residency; bounded `PrefetchDataFeed`; byte-bounded LRU |
| 4 | Plugin architecture | `plugins/registry.py`, entry points; §11 |
| 5 | Python-first | `StrategyContext` + strategy bases as the primary API |
| 6 | Clear errors and logging | `errors.py` hierarchy with structured context; `RunLogger` on simulation time; §12 |

---

## 14. Key design decisions, and what they cost

| Decision | Why | Trade-off accepted |
|---|---|---|
| float64, not `Decimal` | Vectorisation and sweep throughput | Rounding must be applied deliberately at boundaries |
| Right-labelled bars | Makes "visible only after close" definitional | Left-labelled source files must be shifted on load |
| Two-stage order path (intent → order) | Makes sizing a swappable policy (Accounting 4) | One extra object per order |
| Broker emits fills; portfolio applies them | Independently testable; no half-applied state | Two-step flow instead of one |
| Read-only `PortfolioView` for strategies | User code cannot corrupt the ledger | An extra interface to maintain |
| Indicators declared, not constructed | Engine can size warm-up before loading data; each bar seen exactly once | Slightly more ceremony for authors |
| Indicators need `update` **and** `compute` | Loop needs O(1); warm-up/screeners need vectorised | Author burden, softened by `RollingIndicator` |
| Parallelism across runs, not within | Bar loops are sequential; locking would be slower | No speed-up for a single long run |
| Events for observation only | No per-bar dispatch cost when unobserved | No dynamic event-driven control flow |
| Fixed 13-phase bar order | Reproducibility; lookahead impossible by construction | Less flexible than a general event queue |
| Synthetic spreads flagged and counted | `WORST` stays meaningful on OHLCV-only feeds without hiding it | Extra bookkeeping per fill |
| Pessimistic intrabar resolution | Ambiguous bars resolve against the strategy | Understates some genuinely good fills |

---

## 15. Deferred (the "Future Improvements" table)

The scaffold leaves explicit seams for each:

| Future requirement | Seam already present |
|---|---|
| News / event-driven strategies | `DataProvider` is generic over stream sources; `MarketSnapshot` can carry extra channels |
| ML-prediction strategies | `Indicator` is `Generic[T]` — a model wrapper is just an indicator returning a prediction |
| Parameter optimisation | `ParameterSpec.grid()` + `BacktestRunner.run_sweep()` + `BatchResult.best_by()` |
| Walk-forward analysis | `BacktestConfig.derive()` for fold windows; `Strategy.reset()` between folds |
| Monte Carlo over trade sequences | `TradeLog.to_records()` — resampling is post-processing over an existing artefact |
| Futures | `AssetClass.FUTURE` reserved; `Instrument.multiplier` and `MarginModel` already generalise |
| LLM-defined strategies | Strategies are ordinary classes over a narrow context; a generator emits one and the engine is unchanged |
