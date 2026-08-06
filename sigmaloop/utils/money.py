"""Money and price rounding helpers.

The engine computes in float64 (see :mod:`sigmaloop.types` for why). These
helpers apply rounding at the two boundaries where it matters: snapping fill
prices to a valid tick, and presenting cash amounts in a report. Rounding
anywhere else would accumulate bias rather than remove it.
"""

from __future__ import annotations

from sigmaloop.types import Money, Percent, Price, Quantity

__all__ = [
    "round_money",
    "round_to_tick",
    "round_to_lot",
    "is_close",
    "safe_divide",
    "pct_change",
    "notional",
]


def round_money(amount: Money, places: int = 2) -> Money:
    """Round to the currency's minor unit, half-away-from-zero.

    Banker's rounding (Python's default) is wrong for money reporting: it makes
    ``round(0.125, 2)`` and ``round(0.135, 2)`` disagree in a way readers do not
    expect.
    """
    raise NotImplementedError


def round_to_tick(price: Price, tick_size: Price, mode: str = "nearest") -> Price:
    """Snap to a valid tick. ``mode``: ``"nearest"``, ``"up"``, ``"down"``.

    Fills use directional rounding — always against the trader — so tick
    rounding can never manufacture a better price than the market offered.
    """
    raise NotImplementedError


def round_to_lot(quantity: Quantity, lot_size: Quantity, allow_fractional: bool = False) -> Quantity:
    """Floor toward zero to a valid lot multiple.

    Floor, not round: rounding up would let a sizer spend more capital than the
    account has.
    """
    raise NotImplementedError


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Float comparison for accounting invariants."""
    raise NotImplementedError


def safe_divide(numerator: float, denominator: float, default: float | None = None) -> float | None:
    """Division that returns ``default`` instead of raising or producing inf.

    Used throughout metrics: profit factor with no losses, Sharpe with zero
    volatility. Returning ``None`` there is honest; returning ``inf`` is not.
    """
    raise NotImplementedError


def pct_change(old: float, new: float) -> Percent:
    raise NotImplementedError


def notional(price: Price, quantity: Quantity, multiplier: float = 1.0) -> Money:
    """``price * quantity * multiplier`` — the one place the option multiplier
    is applied, so it cannot be forgotten in one code path and not another."""
    raise NotImplementedError
