"""Option expiry, exercise and assignment.

Options positions do not simply vanish at expiry — they resolve into cash, into
underlying shares, or into nothing, and each path has a different P&L and a
different follow-on position. Getting this wrong is the single largest source
of error in options backtests, so it is handled by a dedicated component rather
than buried in the broker.

Resolution at expiry, for a contract held to the close of its expiry date:

* OTM (and unassigned)  -> expires worthless; premium is the full P&L.
* ITM long, PHYSICAL    -> exercised: long call buys shares at strike,
                           long put sells shares at strike.
* ITM short, PHYSICAL   -> assigned (subject to :attr:`ExpiryPolicy.assignment_probability`).
* ITM, CASH settled     -> intrinsic value credited/debited.

American-style early assignment is approximated, not simulated tick-by-tick:
short ITM options are assigned with configurable probability, raised near
ex-dividend dates for short calls.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.instrument import OptionContract
from sigmaloop.domain.order import Fill
from sigmaloop.domain.position import Position
from sigmaloop.errors import DataNotAvailableError, ValidationError
from sigmaloop.types import (
    FillId,
    InstrumentId,
    Money,
    OptionRight,
    OptionStyle,
    OrderId,
    OrderSide,
    Price,
    Quantity,
    SettlementType,
    TradeCloseReason,
    UtcDatetime,
)

__all__ = ["ExpiryPolicy", "ExpiryOutcome", "ExpiryEngine", "StandardExpiryEngine"]

#: Quantities at or below this are float dust, not a holding.
_FLAT_TOLERANCE: Quantity = 1e-9


@dataclass(frozen=True, slots=True)
class ExpiryPolicy:
    """Configurable rules for how expiring options are resolved."""

    #: Close positions this many bars before expiry instead of letting them
    #: expire. Avoids modelling assignment at the cost of realism.
    close_before_expiry_bars: int | None = None
    #: Probability a short ITM American option is assigned early, per bar ITM.
    early_assignment_probability: float = 0.0
    #: Assign short ITM options at expiry with this probability (1.0 == always).
    expiry_assignment_probability: float = 1.0
    #: Auto-exercise long options ITM by at least this much per share.
    auto_exercise_threshold: Price = 0.01
    #: If False, physical settlement is converted to cash-equivalent instead of
    #: creating an underlying position (useful when no equity data is loaded).
    allow_physical_settlement: bool = True
    #: Charge this per contract on exercise/assignment.
    exercise_fee_per_contract: Money = 0.0


@dataclass(frozen=True, slots=True)
class ExpiryOutcome:
    """What happened to one expiring position.

    :attr:`fills` is the ledger-bearing part: applying them closes the contract
    and, under physical settlement, opens or closes the underlying leg.
    :attr:`cash_impact` restates the same movement as a single number for
    reporting and for assertions — a portfolio that books both would count the
    settlement twice.
    """

    contract: OptionContract
    quantity: Quantity
    reason: TradeCloseReason
    settlement_price: Price
    underlying_price: Price
    cash_impact: Money
    #: Non-zero when PHYSICAL settlement creates/removes an underlying position.
    underlying_quantity_delta: Quantity = 0.0
    fees: Money = 0.0
    #: Synthetic fills recorded so the trade log shows the full round trip.
    fills: tuple[Fill, ...] = ()


class ExpiryEngine(ABC):
    """Resolves expiring and assigned option positions each bar."""

    @abstractmethod
    def expiring_positions(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        """Positions whose contract expires at (or before) this timestamp."""
        raise NotImplementedError

    @abstractmethod
    def resolve(self, position: Position, snapshot: MarketSnapshot) -> ExpiryOutcome:
        raise NotImplementedError

    @abstractmethod
    def check_early_assignment(
        self, position: Position, snapshot: MarketSnapshot
    ) -> ExpiryOutcome | None:
        """Called every bar on short option positions; ``None`` if untouched."""
        raise NotImplementedError

    def positions_to_close_early(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        """Positions the policy wants flattened before expiry."""
        return ()


class StandardExpiryEngine(ExpiryEngine):
    """Reference implementation of the resolution rules documented above.

    Deterministic given a seed: assignment draws come from the run's seeded RNG
    so an identical config reproduces identical results, which is a hard
    requirement for parameter sweeps to be comparable.
    """

    __slots__ = ("_policy", "_rng", "_settlement_source", "_next_fill_seq")

    def __init__(self, policy: ExpiryPolicy, seed: int = 0) -> None:
        self._policy = policy
        self._rng = random.Random(seed)
        #: Last observed price per underlying, carried forward. An expiry bar
        #: does not always carry the underlying (a holiday-shortened session, a
        #: halt, or an options-only feed), and settlement still has to resolve.
        self._settlement_source: dict[InstrumentId, Price] = {}
        self._next_fill_seq = 0

    @property
    def policy(self) -> ExpiryPolicy:
        return self._policy

    def expiring_positions(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        """Contracts at or past their expiry date, resolved at the session close.

        Also refreshes the underlying price cache, since this runs every bar and
        is the last chance to observe a price before the expiry bar itself.
        """
        expiring: list[Position] = []
        for position in positions:
            contract = _as_contract(position)
            if contract is None or not position.is_open:
                continue
            observed = snapshot.price(contract.underlying_id)
            if observed is not None:
                self._settlement_source[contract.underlying_id] = observed
            if not snapshot.is_session_close:
                # Settlement is struck at the close. Resolving intraday would
                # book the 0DTE contract at whatever the last bar happened to
                # print and free its margin hours early.
                continue
            if snapshot.timestamp.date() >= contract.expiry:
                expiring.append(position)
        return tuple(expiring)

    def positions_to_close_early(
        self, positions: Sequence[Position], snapshot: MarketSnapshot
    ) -> Sequence[Position]:
        horizon = self._policy.close_before_expiry_bars
        if horizon is None:
            return ()
        due: list[Position] = []
        for position in positions:
            contract = _as_contract(position)
            if contract is None or not position.is_open:
                continue
            if 0 <= contract.days_to_expiry(snapshot.timestamp) <= horizon:
                due.append(position)
        return tuple(due)

    def resolve(self, position: Position, snapshot: MarketSnapshot) -> ExpiryOutcome:
        contract = _require_contract(position)
        underlying_price = self._settlement_price(contract, snapshot)
        return self._resolve_at(position, underlying_price, snapshot.timestamp)

    def check_early_assignment(
        self, position: Position, snapshot: MarketSnapshot
    ) -> ExpiryOutcome | None:
        """Draw for early assignment on a short ITM American contract.

        The draw is skipped entirely at probability 0 or 1 so that turning the
        feature on does not shift the RNG stream for every other stochastic
        component in the run.

        The refinement the docs mention — a higher rate for short calls into an
        ex-dividend date — needs a dividend calendar the execution layer is not
        given; it belongs here once the data layer carries one.
        """
        contract = _as_contract(position)
        if contract is None or not position.is_open or not position.is_short:
            return None
        if contract.style is not OptionStyle.AMERICAN:
            return None
        probability = self._policy.early_assignment_probability
        if probability <= 0.0:
            return None
        underlying_price = snapshot.price(contract.underlying_id)
        if underlying_price is None:
            return None
        self._settlement_source[contract.underlying_id] = underlying_price
        if not contract.is_itm(underlying_price):
            return None
        if probability < 1.0 and self._rng.random() >= probability:
            return None
        return self._resolve_at(
            position, underlying_price, snapshot.timestamp, force_assignment=True
        )

    # ---- internals ----------------------------------------------------------- #

    def _settlement_price(self, contract: OptionContract, snapshot: MarketSnapshot) -> Price:
        """Underlying's official close on the expiry date."""
        observed = snapshot.price(contract.underlying_id)
        if observed is not None:
            self._settlement_source[contract.underlying_id] = observed
            return observed
        carried = self._settlement_source.get(contract.underlying_id)
        if carried is not None:
            return carried
        raise DataNotAvailableError(
            contract.underlying_id,
            timestamp=snapshot.timestamp.isoformat(),
            contract=contract.instrument_id,
            hint=(
                "The expiring contract's underlying has never been priced in this "
                "run, so intrinsic value at settlement is unknowable. Subscribe to "
                "the underlying, or run with OptionsConfig settlement data loaded."
            ),
        )

    def _resolve_at(
        self,
        position: Position,
        underlying_price: Price,
        at: UtcDatetime,
        *,
        force_assignment: bool = False,
    ) -> ExpiryOutcome:
        contract = _require_contract(position)
        contracts = abs(position.quantity)
        if contracts <= _FLAT_TOLERANCE:
            raise ValidationError(
                "Cannot resolve an expiry for a flat position.",
                instrument_id=contract.instrument_id,
            )
        is_long = position.quantity > 0.0
        intrinsic = contract.intrinsic_value(underlying_price)
        policy = self._policy

        exercised = is_long and intrinsic >= policy.auto_exercise_threshold
        assigned = force_assignment or (not is_long and intrinsic > 0.0 and self._draw_assignment())
        if not exercised and not assigned:
            # OTM, or ITM by less than the auto-exercise threshold, or a short
            # the holder chose not to exercise. The premium is the whole P&L.
            return self._outcome(
                contract=contract,
                position=position,
                reason=TradeCloseReason.OPTION_EXPIRY_WORTHLESS,
                settlement_price=0.0,
                underlying_price=underlying_price,
                option_close_price=0.0,
                underlying_delta=0.0,
                fees=0.0,
                at=at,
            )

        reason = TradeCloseReason.OPTION_EXERCISE if is_long else TradeCloseReason.OPTION_ASSIGNMENT
        fees = policy.exercise_fee_per_contract * contracts
        physical = (
            contract.settlement is SettlementType.PHYSICAL and policy.allow_physical_settlement
        )
        if not physical:
            # Cash settlement: the contract itself pays out its intrinsic value.
            return self._outcome(
                contract=contract,
                position=position,
                reason=reason,
                settlement_price=intrinsic,
                underlying_price=underlying_price,
                option_close_price=intrinsic,
                underlying_delta=0.0,
                fees=fees,
                at=at,
            )

        # Physical delivery. The contract itself expires at nothing; the value
        # arrives as shares transacted at the strike, which is the actual cash
        # flow — closing at intrinsic *and* delivering shares would count it
        # twice.
        shares = contracts * contract.multiplier
        # Who ends up holding the shares: a long call and a short put receive
        # them; a long put and a short call deliver them.
        receives = (contract.right is OptionRight.CALL) == is_long
        delta = shares if receives else -shares
        return self._outcome(
            contract=contract,
            position=position,
            reason=reason,
            settlement_price=intrinsic,
            underlying_price=underlying_price,
            option_close_price=0.0,
            underlying_delta=delta,
            fees=fees,
            at=at,
        )

    def _draw_assignment(self) -> bool:
        probability = self._policy.expiry_assignment_probability
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return self._rng.random() < probability

    def _outcome(
        self,
        *,
        contract: OptionContract,
        position: Position,
        reason: TradeCloseReason,
        settlement_price: Price,
        underlying_price: Price,
        option_close_price: Price,
        underlying_delta: Quantity,
        fees: Money,
        at: UtcDatetime,
    ) -> ExpiryOutcome:
        """Assemble the synthetic fills and the net cash the resolution moves."""
        contracts = abs(position.quantity)
        is_long = position.quantity > 0.0
        fills = [
            self._fill(
                instrument_id=contract.instrument_id,
                side=OrderSide.SELL if is_long else OrderSide.BUY,
                quantity=contracts,
                price=option_close_price,
                at=at,
                fees=0.0 if underlying_delta else fees,
            )
        ]
        cash = (1.0 if is_long else -1.0) * option_close_price * contracts * contract.multiplier
        if underlying_delta:
            fills.append(
                self._fill(
                    instrument_id=contract.underlying_id,
                    side=OrderSide.BUY if underlying_delta > 0.0 else OrderSide.SELL,
                    quantity=abs(underlying_delta),
                    price=contract.strike,
                    at=at,
                    fees=fees,
                )
            )
            cash -= underlying_delta * contract.strike
        return ExpiryOutcome(
            contract=contract,
            quantity=position.quantity,
            reason=reason,
            settlement_price=settlement_price,
            underlying_price=underlying_price,
            cash_impact=cash - fees,
            underlying_quantity_delta=underlying_delta,
            fees=fees,
            fills=tuple(fills),
        )

    def _fill(
        self,
        *,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        price: Price,
        at: UtcDatetime,
        fees: Money,
    ) -> Fill:
        """A fill with no order behind it — settlement is not a trade anyone placed."""
        self._next_fill_seq += 1
        return Fill(
            fill_id=FillId(f"X{self._next_fill_seq}"),
            order_id=OrderId(f"EXPIRY:{instrument_id}"),
            instrument_id=instrument_id,
            timestamp=at,
            side=side,
            quantity=quantity,
            price=price,
            fees=fees,
            reference_price=price,
        )


def _as_contract(position: Position) -> OptionContract | None:
    instrument = position.instrument
    return instrument if isinstance(instrument, OptionContract) else None


def _require_contract(position: Position) -> OptionContract:
    contract = _as_contract(position)
    if contract is None:
        raise ValidationError(
            "The expiry engine was handed a position that is not an option; only "
            "contracts expire, and treating an equity as one would flatten a "
            "holding the strategy never closed.",
            instrument_id=position.instrument_id,
            asset_class=position.instrument.asset_class.value,
        )
    return contract
