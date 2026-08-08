"""The exception tree: structured context alongside a readable message (NFR 6)."""

from __future__ import annotations

import pytest

from sigmaloop.errors import (
    ExecutionError,
    OrderRejectedError,
    PluginNotFoundError,
    SigmaLoopError,
)
from sigmaloop.types import OrderId, RejectReason


def test_context_is_rendered_alongside_the_message() -> None:
    raised = SigmaLoopError("Something gave way.", instrument_id="EQ:MSFT", bar=41_000)

    assert raised.message == "Something gave way."
    assert raised.context == {"instrument_id": "EQ:MSFT", "bar": 41_000}
    assert "instrument_id='EQ:MSFT'" in str(raised)


def test_a_missing_plugin_names_what_was_available_instead() -> None:
    """The whole value of the error is telling the reader what they could have
    typed; a bare "not found" sends them to the source to find out."""
    raised = PluginNotFoundError("data_provider", "polygon", ("csv", "yahoo"))

    assert raised.group == "data_provider"
    assert raised.name == "polygon"
    assert raised.available == ("csv", "yahoo")
    assert "csv, yahoo" in str(raised)


def test_an_empty_registry_says_so_rather_than_printing_nothing() -> None:
    assert "none registered" in str(PluginNotFoundError("indicator", "sma"))


def test_a_rejection_keeps_the_order_and_reason_in_front_of_the_reader() -> None:
    """``__str__`` renders ``self.message``, so the formatted headline must not
    be overwritten by the raw detail."""
    raised = OrderRejectedError(
        OrderId("O-1"), RejectReason.INSUFFICIENT_CAPITAL, "needs 5000, has 100"
    )

    rendered = str(raised)
    assert "Order O-1 rejected (insufficient_capital)" in rendered
    assert "needs 5000, has 100" in rendered
    assert raised.detail == "needs 5000, has 100"
    assert raised.reason is RejectReason.INSUFFICIENT_CAPITAL


@pytest.mark.parametrize(
    "raised",
    [
        PluginNotFoundError("data_provider", "polygon"),
        OrderRejectedError(OrderId("O-1"), RejectReason.MARKET_CLOSED, "shut"),
    ],
)
def test_every_error_is_catchable_as_one_root_type(raised: SigmaLoopError) -> None:
    """A caller can wrap an entire run in one ``except``."""
    assert isinstance(raised, SigmaLoopError)


def test_a_rejection_is_an_execution_failure() -> None:
    assert isinstance(
        OrderRejectedError(OrderId("O-1"), RejectReason.STALE_QUOTE, "old"), ExecutionError
    )
