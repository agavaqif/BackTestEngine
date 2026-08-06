"""SigmaLoop — a programmable, high-performance event-driven backtesting engine.

Quick start
-----------

::

    from sigmaloop import (
        BacktestConfig, BacktestEngine, SingleAssetStrategy,
        Parameter, ParameterSpec, StrategyMode,
    )

    class BuyTheOpen(SingleAssetStrategy):
        symbol = "TQQQ"

        @classmethod
        def param_spec(cls):
            return ParameterSpec.from_parameters(
                Parameter("take_profit_pct", 0.10, "Exit when up this much"),
            )

        def on_bar(self, ctx, bar):
            pos = ctx.position(self.instrument_id)
            if pos is None:
                ctx.buy(self.instrument_id, percent_equity=1.0)
            elif pos.unrealized_pnl_pct >= ctx.params.take_profit_pct:
                ctx.close(self.instrument_id)

    config = BacktestConfig(
        strategy_mode=StrategyMode.SINGLE_ASSET,
        start=..., end=..., symbols=("TQQQ",),
    )
    result = BacktestEngine(config, BuyTheOpen()).run()
    print(result.summary())

Architecture
------------
See ``DESIGN.md`` for the full design. The layering, innermost first::

    types / errors
      -> domain          (Bar, Order, Position, Trade, Instrument)
        -> data          (providers, feed, universe, calendar, cache)
        -> indicators
        -> execution     (pricing, slippage, commission, broker, expiry)
        -> portfolio     (accounting, sizing, risk)
          -> strategy    (bases, context, params, order API)
            -> engine    (config, clock, loop, runner)
              -> metrics / results

Every arrow points one way. The engine depends on abstractions; concrete
implementations are resolved by name through :mod:`sigmaloop.plugins`.
"""

from __future__ import annotations

from sigmaloop.domain import (
    Bar,
    BarSeries,
    Equity,
    EquityPoint,
    Fill,
    Greeks,
    Instrument,
    MarketSnapshot,
    OptionChain,
    OptionContract,
    OptionQuote,
    OptionTrade,
    Order,
    OrderIntent,
    Position,
    Quote,
    SizingRequest,
    Trade,
)
from sigmaloop.engine import (
    AccountingConfig,
    BacktestConfig,
    BacktestEngine,
    BacktestRunner,
    DataConfig,
    ExecutionConfig,
    OptionsConfig,
    ParallelConfig,
    ReportingConfig,
)
from sigmaloop.errors import SigmaLoopError
from sigmaloop.indicators import Indicator, IndicatorSpec
from sigmaloop.metrics import PerformanceMetrics
from sigmaloop.results import BacktestResult, EquityCurve, TradeLog
from sigmaloop.strategy import (
    OptionsStrategy,
    Parameter,
    ParameterSet,
    ParameterSpec,
    PortfolioStrategy,
    SingleAssetStrategy,
    Strategy,
    StrategyContext,
)
from sigmaloop.types import (
    AssetClass,
    ExecutionTiming,
    OptionRight,
    OrderSide,
    OrderType,
    PriceSelection,
    SizingMode,
    StrategyMode,
    Timeframe,
)

__version__ = "0.1.0"

__all__ = [
    "AccountingConfig",
    "AssetClass",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunner",
    "Bar",
    "BarSeries",
    "DataConfig",
    "Equity",
    "EquityCurve",
    "EquityPoint",
    "ExecutionConfig",
    "ExecutionTiming",
    "Fill",
    "Greeks",
    "Indicator",
    "IndicatorSpec",
    "Instrument",
    "MarketSnapshot",
    "OptionChain",
    "OptionContract",
    "OptionQuote",
    "OptionRight",
    "OptionTrade",
    "OptionsConfig",
    "OptionsStrategy",
    "Order",
    "OrderIntent",
    "OrderSide",
    "OrderType",
    "ParallelConfig",
    "Parameter",
    "ParameterSet",
    "ParameterSpec",
    "PerformanceMetrics",
    "PortfolioStrategy",
    "Position",
    "PriceSelection",
    "Quote",
    "ReportingConfig",
    "SigmaLoopError",
    "SingleAssetStrategy",
    "SizingMode",
    "SizingRequest",
    "Strategy",
    "StrategyContext",
    "StrategyMode",
    "Timeframe",
    "Trade",
    "TradeLog",
    "__version__",
]
