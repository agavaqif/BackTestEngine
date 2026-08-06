"""Parallel execution across symbols and parameter sets (NFR 2).

Where the parallelism is
------------------------
Not inside a run. A bar loop is a sequential dependency chain — bar *t*'s
ledger is bar *t-1*'s output — so splitting it would require locking the
portfolio on every fill and would run slower, not faster.

Across runs, though, the work is embarrassingly parallel:

* one run per parameter set (sweeps, walk-forward folds),
* one run per symbol (single-asset strategies over a watchlist).

:class:`BacktestRunner` fans those out over a process pool. Processes, not
threads, because the loop is CPU-bound pure Python and the GIL would serialise
threads. The cost of processes is that configs, strategy classes and results
must be picklable — which is why :class:`~sigmaloop.engine.config.BacktestConfig`
is a plain frozen dataclass and strategies are passed as *class + params*, never
as live instances.

Data loading is the one shared cost, and the on-disk
:class:`~sigmaloop.data.cache.ParquetDataCache` amortises it: the first worker
to need a series pays for it, the rest memory-map the cached copy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from sigmaloop.engine.config import BacktestConfig
from sigmaloop.results.result import BacktestResult
from sigmaloop.strategy.base import Strategy
from sigmaloop.types import ParamDict, RunId

__all__ = [
    "RunSpec",
    "RunOutcome",
    "BatchResult",
    "Executor",
    "SerialExecutor",
    "ProcessExecutor",
    "ThreadExecutor",
    "BacktestRunner",
]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One unit of work: a config plus the strategy class to run under it.

    Carries the strategy *class* rather than an instance so the spec stays
    picklable and so each worker constructs fresh, uncontaminated state.
    """

    run_id: RunId
    config: BacktestConfig
    strategy_class: type[Strategy]
    params: ParamDict = field(default_factory=dict)
    label: str = ""

    def fingerprint(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Result of one spec — success or captured failure.

    Failures are values, not exceptions: one bad parameter combination must not
    destroy a 500-point sweep.
    """

    spec: RunSpec
    result: BacktestResult | None
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class BatchResult:
    """Aggregate over a batch of runs."""

    outcomes: list[RunOutcome] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def successful(self) -> Sequence[RunOutcome]:
        raise NotImplementedError

    def failed(self) -> Sequence[RunOutcome]:
        raise NotImplementedError

    def best_by(self, metric: str, maximise: bool = True) -> RunOutcome | None:
        """Top run by a named metric — the hook parameter optimisation uses."""
        raise NotImplementedError

    def to_frame(self) -> object:
        """One row per run: parameters plus every metric. For sweep analysis."""
        raise NotImplementedError


class Executor(ABC):
    """Abstracts how work is distributed, so the runner is testable serially."""

    @abstractmethod
    def map(
        self, fn: Callable[[RunSpec], RunOutcome], specs: Sequence[RunSpec]
    ) -> Iterator[RunOutcome]:
        """Yield outcomes as they complete, not in submission order."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError


class SerialExecutor(Executor):
    """Runs in-process. Debugging, profiling and deterministic tests."""

    def map(
        self, fn: Callable[[RunSpec], RunOutcome], specs: Sequence[RunSpec]
    ) -> Iterator[RunOutcome]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class ProcessExecutor(Executor):
    """``ProcessPoolExecutor`` over run specs. The production path."""

    __slots__ = ("_max_workers", "_pool", "_chunk_size")

    def __init__(self, max_workers: int | None = None, chunk_size: int = 1) -> None:
        raise NotImplementedError

    def map(
        self, fn: Callable[[RunSpec], RunOutcome], specs: Sequence[RunSpec]
    ) -> Iterator[RunOutcome]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class ThreadExecutor(Executor):
    """Thread pool. Only useful when runs are dominated by provider I/O."""

    __slots__ = ("_max_workers", "_pool")

    def __init__(self, max_workers: int | None = None) -> None:
        raise NotImplementedError

    def map(
        self, fn: Callable[[RunSpec], RunOutcome], specs: Sequence[RunSpec]
    ) -> Iterator[RunOutcome]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class BacktestRunner:
    """Builds run specs and executes them, serially or in parallel.

    The single entry point users call:

    * :meth:`run_one`    — one config, one strategy.
    * :meth:`run_symbols` — same strategy over many symbols.
    * :meth:`run_sweep`   — same strategy over a parameter grid (this is the
      seam the future optimisation and walk-forward features attach to; the
      fan-out already exists, only the search policy is missing).
    """

    __slots__ = ("_executor", "_progress", "_fail_fast")

    def __init__(
        self,
        executor: Executor | None = None,
        progress: Callable[[int, int], None] | None = None,
        fail_fast: bool = False,
    ) -> None:
        raise NotImplementedError

    def run_one(
        self,
        config: BacktestConfig,
        strategy_class: type[Strategy],
        params: ParamDict | None = None,
    ) -> BacktestResult:
        raise NotImplementedError

    def run_batch(self, specs: Sequence[RunSpec]) -> BatchResult:
        raise NotImplementedError

    def run_symbols(
        self,
        config: BacktestConfig,
        strategy_class: type[Strategy],
        symbols: Sequence[str],
    ) -> BatchResult:
        """One independent run per symbol. Single-asset mode only — portfolio
        mode is cross-sectional by construction and cannot be split this way."""
        raise NotImplementedError

    def run_sweep(
        self,
        config: BacktestConfig,
        strategy_class: type[Strategy],
        grid: Sequence[ParamDict] | None = None,
    ) -> BatchResult:
        """Run every point in the grid. ``None`` uses the strategy's declared
        :meth:`~sigmaloop.strategy.params.ParameterSpec.grid`."""
        raise NotImplementedError

    @staticmethod
    def _execute(spec: RunSpec) -> RunOutcome:
        """Worker entry point. Must be a module-level-reachable staticmethod so
        it survives pickling into a subprocess."""
        raise NotImplementedError
