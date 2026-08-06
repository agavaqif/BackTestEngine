"""Engine orchestration: config, clock, context, loop and parallel runner."""

from __future__ import annotations

from sigmaloop.engine.clock import Clock, ClockState, SimulationClock
from sigmaloop.engine.config import (
    AccountingConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    OptionsConfig,
    ParallelConfig,
    ReportingConfig,
)
from sigmaloop.engine.context import RunContext, RunDiagnostics
from sigmaloop.engine.core import BacktestEngine, CancellationToken
from sigmaloop.engine.events import (
    BarClosed,
    BarOpened,
    EquityUpdated,
    Event,
    EventBus,
    EventListener,
    EventType,
    OptionExpired,
    OrderFilled,
    OrderRejected,
    OrderSubmitted,
    RunFinished,
    RunStarted,
    TradeClosed,
)
from sigmaloop.engine.runner import (
    BacktestRunner,
    BatchResult,
    Executor,
    ProcessExecutor,
    RunOutcome,
    RunSpec,
    SerialExecutor,
    ThreadExecutor,
)

__all__ = [
    "AccountingConfig",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestRunner",
    "BarClosed",
    "BarOpened",
    "BatchResult",
    "CancellationToken",
    "Clock",
    "ClockState",
    "DataConfig",
    "EquityUpdated",
    "Event",
    "EventBus",
    "EventListener",
    "EventType",
    "ExecutionConfig",
    "Executor",
    "OptionExpired",
    "OptionsConfig",
    "OrderFilled",
    "OrderRejected",
    "OrderSubmitted",
    "ParallelConfig",
    "ProcessExecutor",
    "ReportingConfig",
    "RunContext",
    "RunDiagnostics",
    "RunFinished",
    "RunOutcome",
    "RunSpec",
    "RunStarted",
    "SerialExecutor",
    "SimulationClock",
    "ThreadExecutor",
    "TradeClosed",
]
