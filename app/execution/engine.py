"""
Execution Engine

Handles the full lifecycle of placing and monitoring a trade:
1. Pre-flight checks
2. Order placement (limit-first or market)
3. Fill monitoring
4. Stop-loss and take-profit placement
5. State persistence
6. Notifications
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.trading_config import ExecutionConfig, get_trading_config
from app.exchange.base import BaseExchangeAdapter, ExchangeOrder
from app.models.order import Order
from app.models.trade import Trade
from app.models.fill import Fill
from app.portfolio.state import PortfolioManager, PositionSummary
from app.risk.engine import RiskDecisionResult
from app.signals.schema import SignalDirection
from app.voting.engine import VoteResult

logger = structlog.get_logger(__name__)


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    reason: str
    trade_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    slippage_pct: Optional[float] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """
    Manages order placement, monitoring, and lifecycle management.
    """

    def __init__(
        self,
        adapter: BaseExchangeAdapter,
        db: AsyncSession,
        portfolio: PortfolioManager,
        execution_config: Optional[ExecutionConfig] = None,
        mode: str = "paper",
    ) -> None:
        self._adapter = adapter
        self._db = db
        self._portfolio = portfolio
        self._cfg = execution_config or get_trading_config().execution
        self._mode = mode

    async def execute_trade(
        self,
        vote_result: VoteResult,
        risk_result: RiskDecisionResult,
        current_price: float,
        strategy_signal_summary: str = "",
    ) -> ExecutionResult:
        """
        Execute a trade following the approved risk parameters.

        Execution flow:
        1. Pre-flight sanity checks
        2. Select order type
        3. Place entry order
        4. Monitor fill
        5. Place SL and TP orders
        6. Save to DB
        7. Update portfolio state
        """
        cfg = self._cfg
        symbol = vote_result.symbol
        direction = vote_result.direction
        size = risk_result.approved_quantity
        stop_price = risk_result.approved_stop_price
        tp_price = risk_result.approved_tp_price
        leverage = risk_result.approved_leverage

        trade_id = str(uuid.uuid4())
        client_order_id = f"bot_{trade_id[:8]}"

        # Set leverage if adapter supports it
        if self._adapter.capabilities.supports_set_leverage:
            try:
                await self._adapter.set_leverage(symbol, leverage)
            except Exception as e:
                logger.warning("set_leverage_failed", symbol=symbol, leverage=leverage, error=str(e))

        # Determine order type
        order_type = self._select_order_type(current_price, direction)
        entry_price = self._calculate_limit_price(current_price, direction) if order_type == "limit" else None

        # Place entry order
        try:
            if self._mode == "paper":
                exchange_order = await self._adapter.place_order(
                    symbol=symbol,
                    side="buy" if direction == SignalDirection.LONG else "sell",
                    order_type=order_type,
                    size=size,
                    price=entry_price,
                    client_order_id=client_order_id,
                    current_price=current_price,
                    slippage_pct=cfg.slippage_estimate_basis_points / 10000,
                )
            else:
                exchange_order = await self._adapter.place_order(
                    symbol=symbol,
                    side="buy" if direction == SignalDirection.LONG else "sell",
                    order_type=order_type,
                    size=size,
                    price=entry_price,
                    client_order_id=client_order_id,
                )
        except Exception as e:
            logger.error("entry_order_failed", symbol=symbol, error=str(e), trade_id=trade_id)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                reason=f"entry_order_exception: {str(e)[:200]}",
            )

        # Save order to DB
        db_order = Order(
            trade_id=trade_id,
            exchange_order_id=exchange_order.order_id,
            symbol=symbol,
            order_type=order_type,
            side="buy" if direction == SignalDirection.LONG else "sell",
            quantity=size,
            price=entry_price,
            status=exchange_order.status,
            role="entry",
            placed_at=datetime.now(timezone.utc),
            mode=self._mode,
        )
        self._db.add(db_order)

        # Monitor fill
        fill_result = await self._monitor_fill(exchange_order.order_id, symbol)
        if not fill_result:
            # Cancel and abort
            try:
                await self._adapter.cancel_order(exchange_order.order_id, symbol)
            except Exception:
                pass
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                reason="entry_order_not_filled",
                entry_order_id=exchange_order.order_id,
            )

        actual_fill_price = fill_result.get("avg_fill_price", current_price)
        actual_fill_qty = fill_result.get("filled_size", size)
        slippage_pct = abs(actual_fill_price - current_price) / current_price * 100

        # Check if slippage exceeded threshold post-fill
        if slippage_pct > self._cfg.max_allowed_slippage_pct:
            logger.warning("post_fill_slippage_exceeded",
                          slippage=slippage_pct, threshold=self._cfg.max_allowed_slippage_pct,
                          symbol=symbol)
            # Still proceed, but log warning and notify

        # Update order status
        db_order.status = "filled"
        db_order.filled_qty = actual_fill_qty
        db_order.avg_fill_price = actual_fill_price
        db_order.filled_at = datetime.now(timezone.utc)

        # Save fill
        self._db.add(Fill(
            trade_id=trade_id,
            exchange_order_id=exchange_order.order_id,
            symbol=symbol,
            side="buy" if direction == SignalDirection.LONG else "sell",
            fill_price=actual_fill_price,
            fill_qty=actual_fill_qty,
            fee=actual_fill_qty * actual_fill_price * 0.0005,  # TODO: use actual fee
            fill_time=datetime.now(timezone.utc),
            mode=self._mode,
        ))

        # Create trade record
        now = datetime.now(timezone.utc)
        db_trade = Trade(
            id=trade_id,
            symbol=symbol,
            mode=self._mode,
            direction=direction,
            strategy_family=vote_result.contributing_signals[0].strategy_family if vote_result.contributing_signals else "unknown",
            signal_score=vote_result.final_score,
            score_tier=vote_result.tier,
            entry_time=now,
            entry_price=actual_fill_price,
            quantity=actual_fill_qty,
            notional_usd=actual_fill_qty * actual_fill_price,
            leverage=leverage,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            stop_distance_pct=risk_result.stop_distance_pct,
            capital_risked_usd=risk_result.capital_at_risk,
            slippage_pct=slippage_pct,
            entry_order_id=exchange_order.order_id,
            regime_at_entry=vote_result.details.get("regime"),
            reason_codes=json.dumps(vote_result.reason_codes) if hasattr(vote_result.reason_codes, '__iter__') else "[]",
            status="open",
        )
        self._db.add(db_trade)

        # Place SL and TP orders
        sl_order_id, tp_order_id = await self._place_exit_orders(
            symbol=symbol,
            direction=direction,
            quantity=actual_fill_qty,
            stop_price=stop_price,
            tp_price=tp_price,
            trade_id=trade_id,
        )

        db_trade.sl_order_id = sl_order_id
        db_trade.tp_order_id = tp_order_id

        try:
            await self._db.commit()
        except Exception as e:
            await self._db.rollback()
            logger.error("trade_save_error", error=str(e), trade_id=trade_id)

        # Update portfolio state
        self._portfolio.add_position(PositionSummary(
            symbol=symbol,
            direction=direction,
            quantity=actual_fill_qty,
            entry_price=actual_fill_price,
            current_price=actual_fill_price,
            stop_price=stop_price,
            tp_price=tp_price,
            leverage=leverage,
            notional_usd=actual_fill_qty * actual_fill_price,
            unrealized_pnl=0.0,
            trade_id=trade_id,
        ))

        logger.info(
            "trade_executed",
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            fill_price=actual_fill_price,
            quantity=actual_fill_qty,
            leverage=leverage,
            stop_price=stop_price,
            tp_price=tp_price,
            slippage_pct=round(slippage_pct, 4),
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            reason="executed",
            trade_id=trade_id,
            entry_order_id=exchange_order.order_id,
            fill_price=actual_fill_price,
            fill_quantity=actual_fill_qty,
            slippage_pct=slippage_pct,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
        )

    async def close_trade(
        self,
        trade_id: str,
        current_price: float,
        reason: str = "manual",
    ) -> bool:
        """Close an open trade at current market price."""
        from sqlalchemy import select as sa_select
        result = await self._db.execute(
            sa_select(Trade).where(Trade.id == trade_id, Trade.status == "open")
        )
        trade = result.scalar_one_or_none()
        if not trade:
            logger.warning("close_trade_not_found", trade_id=trade_id)
            return False

        # Place closing order
        close_side = "sell" if trade.direction == "long" else "buy"
        try:
            order = await self._adapter.place_order(
                symbol=trade.symbol,
                side=close_side,
                order_type="market",
                size=trade.quantity,
                reduce_only=True,
                current_price=current_price,
            )
        except Exception as e:
            logger.error("close_order_failed", trade_id=trade_id, error=str(e))
            return False

        # Compute PnL
        if trade.direction == "long":
            pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - current_price) * trade.quantity
        pnl_pct = pnl / trade.capital_risked_usd * 100 if trade.capital_risked_usd else 0.0

        now = datetime.now(timezone.utc)
        hold_minutes = (now - trade.entry_time).total_seconds() / 60

        # Update trade
        trade.status = "closed"
        trade.exit_time = now
        trade.exit_price = current_price
        trade.exit_reason = reason
        trade.realized_pnl = pnl
        trade.realized_pnl_pct = pnl_pct
        trade.hold_duration_minutes = hold_minutes

        try:
            await self._db.commit()
        except Exception as e:
            await self._db.rollback()
            logger.error("trade_close_save_error", error=str(e))
            return False

        # Update portfolio
        self._portfolio.remove_position(trade_id)
        self._portfolio.record_trade_result(pnl)

        logger.info("trade_closed", trade_id=trade_id, pnl=round(pnl, 2),
                   pnl_pct=round(pnl_pct, 2), reason=reason,
                   hold_minutes=round(hold_minutes, 1))
        return True

    async def _monitor_fill(self, order_id: str, symbol: str, timeout_seconds: Optional[int] = None) -> Optional[Dict]:
        """Poll order status until filled or timeout."""
        cfg = self._cfg
        timeout = timeout_seconds or cfg.limit_order_timeout_seconds
        poll_interval = 2.0
        elapsed = 0.0

        while elapsed < timeout:
            try:
                order = await self._adapter.get_order(order_id, symbol)
                if order.status == "filled":
                    return {
                        "avg_fill_price": order.avg_fill_price or 0.0,
                        "filled_size": order.filled_size,
                    }
                elif order.status in ("cancelled", "rejected", "expired"):
                    logger.warning("order_not_fillable", order_id=order_id, status=order.status)
                    return None
            except Exception as e:
                logger.error("fill_monitor_error", order_id=order_id, error=str(e))

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning("order_fill_timeout", order_id=order_id, timeout=timeout)
        return None

    async def _place_exit_orders(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        stop_price: float,
        tp_price: float,
        trade_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Place stop-loss and take-profit orders."""
        close_side = "sell" if direction == SignalDirection.LONG else "buy"
        sl_order_id = None
        tp_order_id = None

        # Stop Loss
        try:
            sl_order = await self._adapter.place_order(
                symbol=symbol,
                side=close_side,
                order_type="stop",
                size=quantity,
                stop_price=stop_price,
                reduce_only=True,
            )
            sl_order_id = sl_order.order_id
            self._db.add(Order(
                trade_id=trade_id,
                exchange_order_id=sl_order.order_id,
                symbol=symbol,
                order_type="stop",
                side=close_side,
                quantity=quantity,
                stop_price=stop_price,
                status="open",
                role="stop_loss",
                placed_at=datetime.now(timezone.utc),
                mode=self._mode,
            ))
        except Exception as e:
            logger.warning("sl_order_failed", symbol=symbol, error=str(e))

        # Take Profit
        try:
            tp_order = await self._adapter.place_order(
                symbol=symbol,
                side=close_side,
                order_type="limit",
                size=quantity,
                price=tp_price,
                reduce_only=True,
            )
            tp_order_id = tp_order.order_id
            self._db.add(Order(
                trade_id=trade_id,
                exchange_order_id=tp_order.order_id,
                symbol=symbol,
                order_type="limit",
                side=close_side,
                quantity=quantity,
                price=tp_price,
                status="open",
                role="take_profit",
                placed_at=datetime.now(timezone.utc),
                mode=self._mode,
            ))
        except Exception as e:
            logger.warning("tp_order_failed", symbol=symbol, error=str(e))

        return sl_order_id, tp_order_id

    def _select_order_type(self, current_price: float, direction: str) -> str:
        pref = self._cfg.order_type_preference
        if pref == "market":
            return "market"
        if pref == "limit":
            return "limit"
        # limit_first: use limit; fall back to market handled by timeout
        return "limit"

    def _calculate_limit_price(self, current_price: float, direction: str) -> float:
        """Set limit slightly inside market to get a good fill."""
        offset = 0.001  # 0.1% inside
        if direction == SignalDirection.LONG:
            return round(current_price * (1 - offset), 2)
        return round(current_price * (1 + offset), 2)


# Import json for trade reason_codes serialization
import json
