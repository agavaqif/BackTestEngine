"""Money and price rounding helpers.

The engine computes in float64 (see :mod:`sigmaloop.types` for why). These
helpers apply rounding at the two boundaries where it matters: snapping fill
prices to a valid tick, and presenting cash amounts in a report. Rounding
anywhere else would accumulate bias rather than remove it.
"""

from __future__ import annotations

import math

from sigmaloop.errors import ValidationError
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

#: Decimal places kept when re-rounding a ``ticks * tick_size`` product. Tick
#: sizes such as 0.01 are not exactly representable, so the product drifts into
#: the 1e-17 range and would print as 100.03000000000001.
_PRICE_DECIMALS = 10

#: Absorbs representation error in the tick count before a directional round:
#: ``100.03 / 0.01`` is 10002.999999999998, and a bare ``ceil`` would snap a
#: price that is already on the grid up to the next tick.
_TICK_EPSILON = 1e-9

_ROUNDING_MODES = ("nearest", "up", "down")


def round_money(amount: Money, places: int = 2) -> Money:
    """Round to the currency's minor unit, half-away-from-zero.

    Banker's rounding (Python's default) is wrong for money reporting: it makes
    ``round(0.125, 2)`` and ``round(0.135, 2)`` disagree in a way readers do not
    expect.
    """
    if not math.isfinite(amount):
        # A non-finite balance is already a bug upstream; rounding it would
        # raise here and hide where it actually came from.
        return amount
    factor = 10.0**places
    magnitude = math.floor(abs(amount) * factor + 0.5) / factor
    return -magnitude if amount < 0.0 else magnitude


def round_to_tick(price: Price, tick_size: Price, mode: str = "nearest") -> Price:
    """Snap to a valid tick. ``mode``: ``"nearest"``, ``"up"``, ``"down"``.

    Fills use directional rounding — always against the trader — so tick
    rounding can never manufacture a better price than the market offered.
    ``"up"`` and ``"down"`` move along the number line, not away from zero, so
    a buy always rounds up and a sell always rounds down whatever the sign.
    """
    if mode not in _ROUNDING_MODES:
        raise ValidationError(
            f"Unknown rounding mode {mode!r}; expected one of {_ROUNDING_MODES}.",
            mode=mode,
        )
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        raise ValidationError(
            "tick_size must be a positive, finite price increment.", tick_size=tick_size
        )
    if not math.isfinite(price):
        return price
    ratio = price / tick_size
    if mode == "up":
        ticks = float(math.ceil(ratio - _TICK_EPSILON))
    elif mode == "down":
        ticks = float(math.floor(ratio + _TICK_EPSILON))
    else:
        ticks = math.copysign(math.floor(abs(ratio) + 0.5), ratio)
    snapped = round(ticks * tick_size, _PRICE_DECIMALS)
    # `or 0.0` kills the -0.0 that a small negative rounds to, which prints as
    # "-0" in every report and compares equal to 0.0 so no test would catch it.
    return snapped or 0.0


def round_to_lot(
    quantity: Quantity, lot_size: Quantity, allow_fractional: bool = False
) -> Quantity:
    """Floor toward zero to a valid lot multiple.

    Floor, not round: rounding up would let a sizer spend more capital than the
    account has. ``allow_fractional`` passes the quantity through untouched, for
    brokers that trade share fractions.
    """
    if not math.isfinite(lot_size) or lot_size <= 0.0:
        raise ValidationError(
            "lot_size must be a positive, finite quantity increment.", lot_size=lot_size
        )
    if allow_fractional or not math.isfinite(quantity):
        return quantity
    lots = math.floor(abs(quantity) / lot_size + _TICK_EPSILON)
    magnitude = round(lots * lot_size, _PRICE_DECIMALS)
    return -magnitude if quantity < 0.0 and magnitude else magnitude


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Float comparison for accounting invariants."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def safe_divide(numerator: float, denominator: float, default: float | None = None) -> float | None:
    """Division that returns ``default`` instead of raising or producing inf.

    Used throughout metrics: profit factor with no losses, Sharpe with zero
    volatility. Returning ``None`` there is honest; returning ``inf`` is not.
    """
    if denominator == 0.0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def pct_change(old: float, new: float) -> Percent:
    """Fractional change from ``old`` to ``new``.

    Growth from nothing is infinite, and reporting it as such keeps the
    degenerate case visible rather than rounding it into a plausible number —
    the same call :attr:`~sigmaloop.domain.account.EquityPoint.leverage` makes.
    """
    if old == 0.0:
        return 0.0 if new == 0.0 else math.copysign(math.inf, new)
    return (new - old) / old


def notional(price: Price, quantity: Quantity, multiplier: float = 1.0) -> Money:
    """``price * quantity * multiplier`` — the one place the option multiplier
    is applied, so it cannot be forgotten in one code path and not another."""
    return price * quantity * multiplier
