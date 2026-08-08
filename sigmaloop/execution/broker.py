"""Simulated broker — the order book, matching loop and fill factory.

Responsibilities, in the order they run each bar:

1. Expire orders whose time-in-force has lapsed.
2. For each working order, ask the :class:`~sigmaloop.execution.models.ExecutionModel`
   whether it is eligible and whether the bar triggers it.
3. Resolve the reference price
   (:class:`~sigmaloop.execution.pricing.FillPriceModel`), apply slippage and
   commission, and emit a :class:`~sigmaloop.domain.order.Fill`.
4. Activate any bracket children once their parent fills.

The broker deliberately does NOT touch the ledger. It emits fills; the
:class:`~sigmaloop.portfolio.accounting.Portfolio` applies them. Keeping matching
and accounting apart is what makes both independently testable, and it is why
a rejected order can never leave a half-applied cash effect behind.

Brackets travel on :attr:`Order.metadata` under :data:`BRACKET_METADATA_KEY`.
An :class:`~sigmaloop.domain.order.Order` carries no bracket field — it is the
sized, broker-visible instruction, and the protective legs are a property of the
*intent* — so the sizing layer copies the spec across when it builds the order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionQuote, Quote
from sigmaloop.domain.instrument import Instrument, InstrumentRegistry, OptionContract
from sigmaloop.domain.order import BracketSpec, Fill, Order, Rejection
from sigmaloop.errors import ExecutionError
from sigmaloop.execution.commission import CommissionContext, CommissionModel
from sigmaloop.execution.models import ExecutionContext, ExecutionModel, FillDecision
from sigmaloop.execution.pricing import FillPriceModel, PricingContext
from sigmaloop.execution.slippage import SlippageContext, SlippageModel
from sigmaloop.types import (
    FillId,
    FillLiquidity,
    InstrumentId,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    PriceSelection,
    Quantity,
    RejectReason,
    TimeInForce,
    TradeCloseReason,
    UtcDatetime,
)
from sigmaloop.utils.money import round_to_tick

__all__ = [
    "BRACKET_METADATA_KEY",
    "OCO_METADATA_KEY",
    "CLOSE_REASON_METADATA_KEY",
    "TRAILING_PCT_METADATA_KEY",
    "BrokerResult",
    "Broker",
    "SimulatedBroker",
]

#: ``Order.metadata`` key holding the :class:`BracketSpec` for an entry order.
BRACKET_METADATA_KEY = "bracket"
#: ``Order.metadata`` key holding the id of a bracket child's OCO sibling.
OCO_METADATA_KEY = "oco_sibling"
#: ``Order.metadata`` key naming why a protective child closes the position, so
#: the trade log can say "stop loss" rather than "signal".
CLOSE_REASON_METADATA_KEY = "close_reason"
#: ``Order.metadata`` key carrying a trailing stop's ratchet distance.
TRAILING_PCT_METADATA_KEY = "trailing_stop_pct"

#: Quantities at or below this are float dust, not a fill.
_QUANTITY_TOLERANCE: Quantity = 1e-9

#: Order types evaluated before the rest within one bar. The intrabar path is
#: unknown, so a bar touching both a bracket's stop and its target is resolved
#: pessimistically — the stop is assumed to have traded first.
_STOP_TYPES = frozenset({OrderType.STOP, OrderType.STOP_LIMIT})

#: Time-in-force values that get exactly one look at the market.
_ONE_SHOT_TIF = frozenset({TimeInForce.IOC, TimeInForce.FOK})

#: Selections that transact on a side of the book, so a missing book means the
#: fill crossed the spread for free. MID and LAST ask for no side and cost
#: nothing to serve without one.
_SPREAD_SENSITIVE_SELECTIONS = frozenset({PriceSelection.WORST, PriceSelection.BEST})


@dataclass(slots=True)
class BrokerResult:
    """Everything the broker produced during one bar."""

    fills: list[Fill] = field(default_factory=list)
    rejections: list[tuple[OrderId, Rejection]] = field(default_factory=list)
    expirations: list[OrderId] = field(default_factory=list)
    cancellations: list[OrderId] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.fills or self.rejections or self.expirations or self.cancellations)


class Broker(ABC):
    """Order intake and matching contract."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept an order into the working book (status -> ACCEPTED)."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, order_id: OrderId, at: UtcDatetime, reason: str = "") -> bool:
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self, at: UtcDatetime, instrument_id: InstrumentId | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def process_bar(self, snapshot: MarketSnapshot) -> BrokerResult:
        """Advance every working order against this bar."""
        raise NotImplementedError

    @abstractmethod
    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: OrderId) -> Order | None:
        raise NotImplementedError


class SimulatedBroker(Broker):
    """Bar-driven matching engine.

    Working orders are indexed by instrument so a bar only walks the orders
    that could possibly fill on it — O(orders_on_this_instrument) rather than
    O(all_open_orders). In portfolio mode with thousands of resting brackets,
    that difference is the run.
    """

    __slots__ = (
        "_execution_model",
        "_price_model",
        "_slippage_model",
        "_commission_model",
        "_registry",
        "_price_selection",
        "_orders",
        "_by_instrument",
        "_brackets",
        "_next_fill_seq",
        "_oco",
        "_pending_rejections",
        "_carry_unfilled_remainder",
        "_allow_partial_fills",
        "_period_key",
        "_period_volume",
        "_synthetic_quote_fills",
        "_uncosted_spread_fills",
        "_partial_fills",
    )

    def __init__(
        self,
        execution_model: ExecutionModel,
        price_model: FillPriceModel,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
        registry: InstrumentRegistry,
        price_selection: PriceSelection = PriceSelection.WORST,
        *,
        carry_unfilled_remainder: bool = True,
        allow_partial_fills: bool = True,
    ) -> None:
        """The two keyword flags mirror ``ExecutionConfig``.

        ``carry_unfilled_remainder`` decides what happens to the part of an
        order a liquidity cap refused: carry it to the next bar, or cancel it so
        the strategy's exposure never quietly lags its intent.
        """
        self._execution_model = execution_model
        self._price_model = price_model
        self._slippage_model = slippage_model
        self._commission_model = commission_model
        self._registry = registry
        self._price_selection = price_selection
        self._carry_unfilled_remainder = carry_unfilled_remainder
        self._allow_partial_fills = allow_partial_fills
        self._orders: dict[OrderId, Order] = {}
        # Insertion-ordered inner dicts: O(1) add and remove, and a deterministic
        # walk, which is what makes two runs of one config byte-identical.
        self._by_instrument: dict[InstrumentId, dict[OrderId, Order]] = {}
        self._brackets: dict[OrderId, BracketSpec] = {}
        self._oco: dict[OrderId, OrderId] = {}
        self._pending_rejections: list[tuple[OrderId, Rejection]] = []
        self._next_fill_seq = 0
        self._period_key: tuple[int, int] | None = None
        self._period_volume: Quantity = 0.0
        self._synthetic_quote_fills = 0
        self._uncosted_spread_fills = 0
        self._partial_fills = 0

    # ---- diagnostics --------------------------------------------------------- #

    @property
    def synthetic_quote_fills(self) -> int:
        """Fills priced off a synthesised, not observed, book.

        Feeds :attr:`~sigmaloop.results.result.RunSummaryStats.synthetic_quote_fills`
        so a reader can see how much of a result rests on a modelled spread.
        """
        return self._synthetic_quote_fills

    @property
    def uncosted_spread_fills(self) -> int:
        """Fills that crossed a spread the run never charged for.

        Reached when ``price_selection`` asks for a side of the book but the feed
        carries no quotes and no ``SpreadModel`` was configured, so ``WORST``
        quietly degrades to the bar's own price. A non-zero count means part of
        the result was priced as if trading were free, and the engine turns it
        into a run warning rather than leaving it to be inferred.
        """
        return self._uncosted_spread_fills

    @property
    def partial_fills(self) -> int:
        return self._partial_fills

    # ---- intake -------------------------------------------------------------- #

    def submit(self, order: Order) -> None:
        if order.status is not OrderStatus.PENDING_NEW:
            raise ExecutionError(
                "Only a PENDING_NEW order can be submitted; this one has already "
                "been through the book.",
                order_id=order.order_id,
                status=order.status.value,
            )
        if order.order_id in self._orders:
            raise ExecutionError(
                "Duplicate order id. Ids key the working book, the fill records "
                "and the trade log, so reusing one would merge two orders.",
                order_id=order.order_id,
            )

        instrument = self._registry.try_get(order.instrument_id)
        if instrument is None:
            self._reject(
                order,
                RejectReason.INSTRUMENT_NOT_TRADEABLE,
                "Instrument is not in the run's registry, so the broker has no "
                "contract terms (multiplier, tick, lot) to execute against.",
            )
            return
        if not instrument.is_tradeable:
            self._reject(
                order,
                RejectReason.INSTRUMENT_NOT_TRADEABLE,
                f"{instrument.symbol} is marked non-tradeable for this run.",
            )
            return
        if instrument.is_expired(order.submitted_at):
            self._reject(
                order,
                RejectReason.INSTRUMENT_EXPIRED,
                f"{instrument.symbol} no longer trades as of {order.submitted_at.isoformat()}.",
            )
            return
        if order.quantity <= _QUANTITY_TOLERANCE:
            self._reject(
                order,
                RejectReason.ZERO_OR_NEGATIVE_QUANTITY,
                "Order quantity must be positive; direction lives on the side.",
            )
            return
        if instrument.round_quantity(order.quantity) <= 0.0:
            self._reject(
                order,
                RejectReason.BELOW_MIN_LOT,
                f"Quantity {order.quantity} is below {instrument.symbol}'s lot "
                f"size of {instrument.lot_size}.",
            )
            return
        # The portfolio knows whether this sell opens a short; the broker only
        # knows the instrument cannot be borrowed at all, which is a refusal
        # either way for anything but a position-reducing exit.
        if order.side is OrderSide.SELL and not instrument.is_shortable and not order.reduce_only:
            self._reject(
                order,
                RejectReason.NOT_SHORTABLE,
                f"{instrument.symbol} is not shortable and the order is not marked reduce_only.",
            )
            return

        order.status = OrderStatus.ACCEPTED
        self._orders[order.order_id] = order
        self._by_instrument.setdefault(order.instrument_id, {})[order.order_id] = order
        bracket = order.metadata.get(BRACKET_METADATA_KEY)
        if isinstance(bracket, BracketSpec):
            self._brackets[order.order_id] = bracket

    def cancel(self, order_id: OrderId, at: UtcDatetime, reason: str = "") -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.is_open:
            return False
        order.cancel(at, reason)
        self._forget(order)
        return True

    def cancel_all(self, at: UtcDatetime, instrument_id: InstrumentId | None = None) -> int:
        targets = [order.order_id for order in self.working_orders(instrument_id)]
        return sum(1 for order_id in targets if self.cancel(order_id, at, "cancel_all"))

    # ---- matching ------------------------------------------------------------ #

    def process_bar(self, snapshot: MarketSnapshot) -> BrokerResult:
        result = BrokerResult(rejections=list(self._pending_rejections))
        self._pending_rejections.clear()

        for instrument_id in list(self._by_instrument):
            book = self._by_instrument.get(instrument_id)
            if not book:
                self._by_instrument.pop(instrument_id, None)
                continue
            instrument = self._registry.get(instrument_id)
            option_quote = self._option_quote(snapshot, instrument)
            bar = self._market_bar(snapshot, instrument_id, option_quote)
            # Snapshot the book: activating a bracket adds children to it, and
            # a child must not be worked on the bar that created its parent.
            for order in sorted(book.values(), key=_matching_priority):
                if order.is_open:
                    self._work_order(order, instrument, bar, option_quote, snapshot, result)
            self._prune(instrument_id)
        return result

    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        if instrument_id is not None:
            book = self._by_instrument.get(instrument_id, {})
            return tuple(order for order in book.values() if order.is_open)
        return tuple(order for order in self._orders.values() if order.is_open)

    def get_order(self, order_id: OrderId) -> Order | None:
        return self._orders.get(order_id)

    # ---- internals ---------------------------------------------------------- #

    def _work_order(
        self,
        order: Order,
        instrument: Instrument,
        bar: Bar | None,
        option_quote: OptionQuote | None,
        snapshot: MarketSnapshot,
        result: BrokerResult,
    ) -> None:
        """Run one working order through expiry, eligibility and matching."""
        context = ExecutionContext(
            order=order,
            instrument=instrument,
            timestamp=snapshot.timestamp,
            bar=bar,
            option_quote=option_quote,
            is_session_close=snapshot.is_session_close,
        )
        if self._execution_model.should_expire(context):
            self._expire(order, snapshot.timestamp, "time in force lapsed")
            result.expirations.append(order.order_id)
            return
        if not self._execution_model.is_eligible(context):
            return
        if order.activated_at is None:
            # First bar this order could have traded on. Time-in-force counts
            # from here, not from submission — see ExecutionModel.should_expire.
            order.activated_at = snapshot.timestamp

        decision = self._execution_model.try_fill(context)
        quantity = self._fillable(order, instrument, decision, bar) if decision.should_fill else 0.0

        if quantity <= _QUANTITY_TOLERANCE:
            self._ratchet_trailing_stop(order, bar)
            self._retire_one_shot(order, snapshot.timestamp, result, "no fill on its only bar")
            return

        fill = self._build_fill(order, instrument, decision, quantity, bar, option_quote, snapshot)
        order.apply_fill(fill)
        result.fills.append(fill)
        if fill.is_partial:
            self._partial_fills += 1

        # Before any remainder is dropped: what filled is a real position, and
        # it wants its protective legs whether or not the entry ever completes.
        self._activate_brackets(order, fill)
        if order.is_filled:
            self._settle_oco(order, snapshot.timestamp, result)
        elif order.time_in_force in _ONE_SHOT_TIF or not self._carry_unfilled_remainder:
            # A truncated fill leaves the strategy's exposure short of its
            # intent. Carrying the rest is one honest answer; cancelling it and
            # letting the next bar's signal re-decide is the other, and the
            # config picks. IOC never gets a second look either way.
            self._cancel_remainder(order, snapshot.timestamp, result)
        self._ratchet_trailing_stop(order, bar)

    def _fillable(
        self, order: Order, instrument: Instrument, decision: FillDecision, bar: Bar | None
    ) -> Quantity:
        """Quantity the bar can actually absorb, after the participation cap."""
        wanted = min(decision.quantity or order.remaining_quantity, order.remaining_quantity)
        if bar is None:
            return wanted
        capped = self._slippage_model.fillable_quantity(
            SlippageContext(
                instrument=instrument,
                side=order.side,
                quantity=wanted,
                reference_price=decision.reference_price,
                bar=bar,
            )
        )
        capped = max(min(wanted, capped), 0.0)
        if capped < wanted:
            # Only snap a truncated quantity: an order sized in whole lots is
            # already valid, and re-rounding a legitimate fractional order
            # against a whole-share lot size would silently zero it.
            capped = instrument.round_quantity(capped)
            if not self._allow_partial_fills:
                return 0.0
            if order.time_in_force is TimeInForce.FOK:
                # Fill-or-kill: a truncated fill is not a fill.
                return 0.0
        return capped

    def _build_fill(
        self,
        order: Order,
        instrument: Instrument,
        decision: FillDecision,
        quantity: Quantity,
        bar: Bar | None,
        option_quote: OptionQuote | None,
        snapshot: MarketSnapshot,
    ) -> Fill:
        """Price -> slippage -> commission -> :class:`Fill`.

        The execution model has already decided *where in the bar* this trades;
        the price model adds the side of the book and the slippage model the
        impact. Both are recorded on the fill so the trade log can attribute
        them separately.
        """
        pricing = PricingContext(
            instrument=instrument,
            side=order.side,
            selection=self._price_selection,
            bar=bar,
            option_quote=option_quote,
            # A synthesised book is built around the level this fill transacts
            # at, not around the bar's close: on a bar that opened at 100 and
            # closed at 140, the close's spread is not the one an opening fill
            # crossed.
            reference_price=max(decision.reference_price, 0.0),
        )
        quote = self._price_model.quote_used(pricing)
        if quote is not None:
            pricing = replace(pricing, synthetic_quote=quote)
            if quote.is_synthetic:
                self._synthetic_quote_fills += 1
        elif self._price_selection in _SPREAD_SENSITIVE_SELECTIONS:
            # Asked to transact on a side of the book, but there is no book and
            # no spread model to invent one, so the fill crosses for free. That
            # is the case the synthetic spread exists to prevent; count it so the
            # run can say how much of its result was priced at zero spread.
            self._uncosted_spread_fills += 1

        reference = decision.reference_price
        if reference <= 0.0:
            reference = self._price_model.resolve(pricing)
        price = reference + self._price_model.spread_adjustment(pricing)

        slippage = 0.0
        if bar is not None:
            slippage = max(
                self._slippage_model.slippage_per_unit(
                    SlippageContext(
                        instrument=instrument,
                        side=order.side,
                        quantity=quantity,
                        # The clean level, not the spread-adjusted price: the two
                        # costs are measured off the same base so the trade log
                        # can attribute them independently.
                        reference_price=reference,
                        # The book that priced this fill, so a spread-fraction
                        # model sees the same quote the price model used.
                        bar=bar if bar.quote is not None else replace(bar, quote=quote),
                    )
                ),
                0.0,
            )
        price += order.side.sign * slippage
        price = self._settle_price(price, order, instrument, bar, quote)

        commission_context = CommissionContext(
            instrument=instrument,
            side=order.side,
            quantity=quantity,
            price=price,
            period_volume=self._period_volume_before(snapshot.timestamp),
            # The broker cannot see the ledger, so reduce_only is the only
            # closing signal it has. Sizers set it on every exit.
            is_closing=order.reduce_only,
            # Charged before this fill is applied, so per-order floors, caps and
            # flat fees bill the order once however many bars it takes to fill.
            filled_before=order.filled_quantity,
        )
        commission = max(self._commission_model.commission(commission_context), 0.0)
        fees = max(self._commission_model.fees(commission_context), 0.0)
        self._period_volume += quantity

        self._next_fill_seq += 1
        return Fill(
            fill_id=FillId(f"F{self._next_fill_seq}"),
            order_id=order.order_id,
            instrument_id=order.instrument_id,
            timestamp=snapshot.timestamp,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            fees=fees,
            slippage_per_unit=slippage,
            # The clean level the execution model derived, so the log can show
            # what the market offered before costs were layered on.
            reference_price=reference,
            liquidity=(
                FillLiquidity.MAKER if order.order_type is OrderType.LIMIT else FillLiquidity.TAKER
            ),
            is_partial=quantity < order.remaining_quantity - _QUANTITY_TOLERANCE,
        )

    def _settle_price(
        self,
        price: Price,
        order: Order,
        instrument: Instrument,
        bar: Bar | None,
        quote: Quote | None,
    ) -> Price:
        """Snap to a tick against the trader, then bound by the market and the limit.

        Three guards, applied in that order:

        * **Tick.** Directional rounding cannot manufacture a better price than
          the market offered.
        * **Market.** Spread and slippage stack on top of a level that may
          already sit at the bar's extreme — a stop triggered at the high, say —
          and left alone they print a buy above every price the bar traded at.
          A quote can legitimately sit outside the traded range (the ask is
          never below the last print), so the bound is the range widened by the
          prevailing half-spread, and no further: past that, slippage is
          inventing liquidity that was never observed.
        * **Limit.** A limit order may fill at its limit or better, never
          through it — :meth:`Order.apply_fill` rejects the alternative.

        The market bound is itself snapped inward to the tick grid, so clamping
        cannot land the fill between ticks.
        """
        is_buy = order.side is OrderSide.BUY
        tick = instrument.tick_size
        price = max(round_to_tick(price, tick, "up" if is_buy else "down"), 0.0)

        # A bar with no range is not evidence of anything — an option priced off
        # its chain gets a flat one-point "bar" at the contract's mid, and
        # bounding to that would clamp away the spread-fraction slippage that is
        # the whole cost model for options.
        if bar is not None and bar.high > bar.low:
            half_spread = 0.0 if quote is None else max(quote.spread, 0.0) * 0.5
            if is_buy:
                price = min(price, round_to_tick(bar.high + half_spread, tick, "down"))
            else:
                price = max(price, round_to_tick(max(bar.low - half_spread, 0.0), tick, "up"))

        if not order.accepts_price(price) and order.limit_price is not None:
            return order.limit_price
        return price

    def _activate_brackets(self, parent: Order, fill: Fill) -> list[Order]:
        """Materialise stop-loss / take-profit children from the parent's fill.

        Children are OCO: filling one cancels its sibling.

        They are stamped with the fill's timestamp, so under the default
        next-bar-open model they cannot trigger on the bar that opened the
        position — a stop cannot be hit by the same print that filled the entry.

        Sized to what the parent has filled *so far*, and resized as more of it
        fills. A partially filled entry is a real position: leaving it bare
        until the entry happened to complete would drop the stop on exactly the
        order a thin market refused to fill in one go.
        """
        spec = self._brackets.get(parent.order_id)
        if spec is None:
            return []
        working = self._existing_children(parent)
        if working:
            for child in working:
                child.quantity = parent.filled_quantity
            return working
        entry = parent.avg_fill_price or fill.price
        is_long = parent.side is OrderSide.BUY
        quantity = parent.filled_quantity
        children: list[Order] = []

        stop_price = spec.stop_loss_price
        if stop_price is None and spec.stop_loss_pct is not None:
            stop_price = entry * (1.0 - spec.stop_loss_pct if is_long else 1.0 + spec.stop_loss_pct)
        trailing = spec.trailing_stop_pct
        if stop_price is None and trailing is not None:
            stop_price = entry * (1.0 - trailing if is_long else 1.0 + trailing)
        if stop_price is not None and stop_price > 0.0:
            child = self._child_order(
                parent,
                suffix="SL",
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                fill=fill,
                close_reason=(
                    TradeCloseReason.TRAILING_STOP if trailing else TradeCloseReason.STOP_LOSS
                ),
            )
            if trailing is not None:
                child.metadata[TRAILING_PCT_METADATA_KEY] = trailing
            children.append(child)

        take_profit = spec.take_profit_price
        if take_profit is None and spec.take_profit_pct is not None:
            take_profit = entry * (
                1.0 + spec.take_profit_pct if is_long else 1.0 - spec.take_profit_pct
            )
        if take_profit is not None and take_profit > 0.0:
            children.append(
                self._child_order(
                    parent,
                    suffix="TP",
                    order_type=OrderType.LIMIT,
                    quantity=quantity,
                    limit_price=take_profit,
                    fill=fill,
                    close_reason=TradeCloseReason.TAKE_PROFIT,
                )
            )

        for child in children:
            self.submit(child)
        if len(children) == 2:
            first, second = children
            self._oco[first.order_id] = second.order_id
            self._oco[second.order_id] = first.order_id
            first.metadata[OCO_METADATA_KEY] = second.order_id
            second.metadata[OCO_METADATA_KEY] = first.order_id
        return children

    def _existing_children(self, parent: Order) -> list[Order]:
        """This parent's protective legs that are still open and unfilled.

        A leg that has begun filling is left alone: its size is already
        committed to a print, and rewriting it would move quantity the market
        has seen.

        Children are found by reconstructing their ids rather than by tracking
        them, which keeps the book the single source of truth. The cost is that
        ``-SL``/``-TP`` is reserved: a caller that submits its own order ending
        in one of those suffixes on the same parent id would collide with a
        bracket leg.
        """
        candidates = (
            self._orders.get(OrderId(f"{parent.order_id}-{suffix}")) for suffix in ("SL", "TP")
        )
        return [
            child
            for child in candidates
            if child is not None and child.is_open and child.filled_quantity == 0.0
        ]

    def _child_order(
        self,
        parent: Order,
        *,
        suffix: str,
        order_type: OrderType,
        quantity: Quantity,
        fill: Fill,
        close_reason: TradeCloseReason,
        limit_price: Price | None = None,
        stop_price: Price | None = None,
    ) -> Order:
        instrument = self._registry.get(parent.instrument_id)
        metadata: dict[str, Any] = {CLOSE_REASON_METADATA_KEY: close_reason}
        return Order(
            order_id=OrderId(f"{parent.order_id}-{suffix}"),
            instrument_id=parent.instrument_id,
            side=parent.side.opposite,
            quantity=quantity,
            order_type=order_type,
            submitted_at=fill.timestamp,
            parent_order_id=parent.order_id,
            limit_price=None if limit_price is None else instrument.round_price(limit_price),
            stop_price=None if stop_price is None else instrument.round_price(stop_price),
            # Protective legs outlive the session that opened the position;
            # a DAY bracket would silently leave the trade unprotected tomorrow.
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            tag=parent.tag,
            metadata=metadata,
        )

    def _settle_oco(self, order: Order, at: UtcDatetime, result: BrokerResult) -> None:
        """One leg filled, so its sibling is no longer wanted."""
        sibling_id = self._oco.pop(order.order_id, None)
        if sibling_id is None:
            return
        self._oco.pop(sibling_id, None)
        if self.cancel(sibling_id, at, f"OCO sibling {order.order_id} filled"):
            result.cancellations.append(sibling_id)

    def _cancel_remainder(self, order: Order, at: UtcDatetime, result: BrokerResult) -> None:
        if order.remaining_quantity <= _QUANTITY_TOLERANCE:
            return
        order.cancel(at, "unfilled remainder cancelled by execution policy")
        self._forget(order)
        result.cancellations.append(order.order_id)

    def _retire_one_shot(
        self, order: Order, at: UtcDatetime, result: BrokerResult, reason: str
    ) -> None:
        """IOC/FOK get one look at the market; an unfilled one is done."""
        if order.time_in_force not in _ONE_SHOT_TIF or order.activated_at is None:
            return
        self._expire(order, at, reason)
        result.expirations.append(order.order_id)

    def _ratchet_trailing_stop(self, order: Order, bar: Bar | None) -> None:
        """Move a trailing stop up (or down) after this bar has been evaluated.

        After, never before: ratcheting on the same bar's extreme and then
        testing the trigger against that same bar would let a high that printed
        *after* the low decide whether the low stopped the trade out.
        """
        if bar is None or not order.is_open or order.stop_price is None:
            return
        raw = order.metadata.get(TRAILING_PCT_METADATA_KEY)
        # `bool` subclasses `int`, so it needs excluding before the numeric test
        # rather than after it: a stray `True` in metadata would otherwise read
        # as a 100% trailing distance and pin the stop to zero.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return
        pct = float(raw)
        if order.side is OrderSide.SELL:
            # Protecting a long: the stop only ever ratchets up.
            order.stop_price = max(order.stop_price, bar.high * (1.0 - pct))
        else:
            order.stop_price = min(order.stop_price, bar.low * (1.0 + pct))

    def _option_quote(self, snapshot: MarketSnapshot, instrument: Instrument) -> OptionQuote | None:
        if not isinstance(instrument, OptionContract):
            return None
        chain = snapshot.chain(instrument.underlying_id)
        return None if chain is None else chain.get(instrument.instrument_id)

    def _market_bar(
        self,
        snapshot: MarketSnapshot,
        instrument_id: InstrumentId,
        option_quote: OptionQuote | None,
    ) -> Bar | None:
        """The bar to match against, synthesising one for options.

        Option chains carry quotes, not bars, but the execution and slippage
        models are written against a bar. A flat bar at the contract's mid,
        carrying its real book, lets one code path serve both asset classes
        rather than branching every model on it.
        """
        bar = snapshot.bar(instrument_id)
        if bar is not None or option_quote is None:
            return bar
        mid = option_quote.quote.mid
        return Bar(
            instrument_id=option_quote.instrument_id,
            timestamp=option_quote.timestamp,
            open=mid,
            high=mid,
            low=mid,
            close=mid,
            volume=option_quote.volume,
            quote=option_quote.quote,
        )

    def _period_volume_before(self, at: UtcDatetime) -> Quantity:
        """Volume traded so far in ``at``'s calendar month, for tiered rates."""
        key = (at.year, at.month)
        if key != self._period_key:
            self._period_key = key
            self._period_volume = 0.0
        return self._period_volume

    def _reject(self, order: Order, reason: RejectReason, message: str) -> None:
        """Refuse an order at intake and queue the record for the next bar.

        Rejections surface through the next :class:`BrokerResult` rather than
        raising, because a refused order is a normal event a strategy is
        entitled to hear about (``on_order_rejected``) and carry on from.
        """
        rejection = Rejection(reason=reason, message=message, timestamp=order.submitted_at)
        order.reject(rejection)
        self._orders[order.order_id] = order
        self._pending_rejections.append((order.order_id, rejection))

    def _expire(self, order: Order, at: UtcDatetime, reason: str) -> None:
        if order.status.is_terminal:
            raise ExecutionError(
                "Cannot expire an order that has already reached a terminal state.",
                order_id=order.order_id,
                status=order.status.value,
            )
        order.status = OrderStatus.EXPIRED
        order.metadata["expired_at"] = at
        order.metadata["expiry_reason"] = reason
        self._forget(order)

    def _forget(self, order: Order) -> None:
        """Drop a now-terminal order from the per-instrument index."""
        book = self._by_instrument.get(order.instrument_id)
        if book is not None:
            book.pop(order.order_id, None)
            if not book:
                self._by_instrument.pop(order.instrument_id, None)
        self._brackets.pop(order.order_id, None)
        sibling = self._oco.pop(order.order_id, None)
        if sibling is not None:
            self._oco.pop(sibling, None)

    def _prune(self, instrument_id: InstrumentId) -> None:
        book = self._by_instrument.get(instrument_id)
        if book is None:
            return
        for order_id in [oid for oid, order in book.items() if not order.is_open]:
            del book[order_id]
            # A settled order will never be worked again, so its bracket spec is
            # dead weight. `_forget` clears the cancel and expire paths; this is
            # the fill path, which reaches terminal state without going through
            # it, and in portfolio mode leaks one entry per completed entry order.
            self._brackets.pop(order_id, None)
        if not book:
            self._by_instrument.pop(instrument_id, None)


def _matching_priority(order: Order) -> int:
    """Stops before everything else, so a stop-and-target bar stops out."""
    return 0 if order.order_type in _STOP_TYPES else 1
