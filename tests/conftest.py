"""Shared fixtures: the shipped sample files plus synthetic layout builders."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd
import pytest

from sigmaloop.data.provider import DataRequest
from sigmaloop.types import Symbol, Timeframe

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "DataSamples"

#: The instant every shipped sample covers.
SAMPLE_DAY = date(2023, 3, 28)


class Window(TypedDict):
    """A request window, so ``request_for(..., **WINDOW)`` keeps its types."""

    start: datetime
    end: datetime


@pytest.fixture(scope="session")
def minute_sample() -> Path:
    path = SAMPLES / "stocks_minute_candlesticks_example.csv"
    if not path.exists():  # pragma: no cover - sample data is checked in
        pytest.skip(f"missing sample file {path}")
    return path


@pytest.fixture(scope="session")
def quote_sample() -> Path:
    path = SAMPLES / "stock_quotes_sample.csv"
    if not path.exists():  # pragma: no cover - sample data is checked in
        pytest.skip(f"missing sample file {path}")
    return path


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    if not SAMPLES.is_dir():  # pragma: no cover - sample data is checked in
        pytest.skip(f"missing sample directory {SAMPLES}")
    return SAMPLES


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


#: The window the synthetic fixtures are built inside.
MARCH: Window = {"start": utc(2023, 3, 1), "end": utc(2023, 3, 31)}


def request_for(
    *symbols: str,
    start: datetime,
    end: datetime,
    timeframe: Timeframe = Timeframe.D1,
    **kwargs: object,
) -> DataRequest:
    return DataRequest(
        symbols=tuple(Symbol(s) for s in symbols),
        start=start,
        end=end,
        timeframe=timeframe,
        **kwargs,  # type: ignore[arg-type]
    )


def daily_frame(
    symbol: str,
    days: list[date],
    base: float,
    *,
    symbol_column: str = "ticker",
    date_column: str = "date",
) -> pd.DataFrame:
    """Long-format daily OHLCV whose values encode their offset, so tests can
    assert on content without hard-coding a table."""
    rows = range(len(days))
    return pd.DataFrame(
        {
            symbol_column: symbol,
            date_column: [d.isoformat() for d in days],
            "open": [base + i for i in rows],
            "high": [base + i + 1 for i in rows],
            "low": [base + i - 1 for i in rows],
            "close": [base + i + 0.5 for i in rows],
            "volume": [1_000 + i for i in rows],
        }
    )


def business_days(start: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


@pytest.fixture
def days() -> list[date]:
    return business_days(date(2023, 3, 1), 20)


@pytest.fixture
def wide_layout(tmp_path: Path, days: list[date]) -> Path:
    """One file per ticker: ``MSFT.csv``, ``AAPL.csv``."""
    root = tmp_path / "wide"
    root.mkdir()
    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)
    daily_frame("AAPL", days, 200).to_csv(root / "AAPL.csv", index=False)
    (root / "notes.txt").write_text("not a csv")
    return root


@pytest.fixture
def per_day_layout(tmp_path: Path, days: list[date]) -> Path:
    """One file per session, each holding the whole universe."""
    root = tmp_path / "per_day"
    root.mkdir()
    for day in days:
        frame = pd.concat([daily_frame("MSFT", [day], 100), daily_frame("AAPL", [day], 200)])
        frame.to_csv(root / f"{day.isoformat()}.csv", index=False)
    return root


@pytest.fixture
def long_layout(tmp_path: Path, days: list[date]) -> Path:
    """A single long-format file holding every ticker and every session."""
    root = tmp_path / "long"
    root.mkdir()
    pd.concat([daily_frame("MSFT", days, 100), daily_frame("AAPL", days, 200)]).to_csv(
        root / "universe.csv", index=False
    )
    return root
