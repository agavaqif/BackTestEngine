"""Structured, simulation-time logging (NFR 6).

Two rules that a general-purpose logger does not give you:

1. **Simulation time, not wall-clock time.** Every record is stamped with the
   bar being processed. "This happened at 14:32 on my laptop" is useless; "this
   happened on the 2019-03-14 bar" is actionable.
2. **Cheap when off.** A backtest can execute millions of loop iterations;
   an unconditional f-string in the hot path is a measurable cost. Log calls
   take a message plus keyword fields and format only if the level is enabled.

Records are both emitted to the standard ``logging`` hierarchy and captured into
the run's diagnostics, so :attr:`BacktestResult.logs` reproduces what happened
without the user having to have configured a handler in advance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sigmaloop.types import UtcDatetime

__all__ = ["LogRecord", "RunLogger", "get_logger", "configure_logging"]


@dataclass(frozen=True, slots=True)
class LogRecord:
    """One structured log line."""

    level: int
    message: str
    #: Simulation timestamp; ``None`` outside the bar loop (setup/teardown).
    sim_time: UtcDatetime | None = None
    bar_index: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        raise NotImplementedError


class RunLogger:
    """Logger bound to one run's clock and diagnostics buffer."""

    __slots__ = (
        "_logger",
        "_clock",
        "_records",
        "_level",
        "_capture",
        "_max_records",
        "_suppressed",
    )

    def __init__(
        self,
        name: str,
        clock: object | None = None,
        level: int = 20,
        capture: bool = True,
        max_records: int = 100_000,
    ) -> None:
        """``max_records`` bounds capture so a chatty strategy cannot exhaust
        memory over a million-bar run; overflow is counted, not stored."""
        raise NotImplementedError

    def debug(self, message: str, /, **fields: Any) -> None:
        raise NotImplementedError

    def info(self, message: str, /, **fields: Any) -> None:
        raise NotImplementedError

    def warning(self, message: str, /, **fields: Any) -> None:
        raise NotImplementedError

    def error(self, message: str, /, **fields: Any) -> None:
        raise NotImplementedError

    def is_enabled(self, level: int) -> bool:
        """Guard for expensive log payloads: ``if log.is_enabled(DEBUG): ...``"""
        raise NotImplementedError

    def bind_clock(self, clock: object) -> None:
        raise NotImplementedError

    def records(self) -> Sequence[LogRecord]:
        raise NotImplementedError

    @property
    def suppressed_count(self) -> int:
        """Records dropped after hitting ``max_records``."""
        raise NotImplementedError


def get_logger(name: str) -> RunLogger:
    """Module-level logger, unbound from any clock."""
    raise NotImplementedError


def configure_logging(level: str = "INFO", format_json: bool = False) -> None:
    """Install SigmaLoop's handler on the root ``sigmaloop`` logger."""
    raise NotImplementedError
