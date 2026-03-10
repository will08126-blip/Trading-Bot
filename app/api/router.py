"""
FastAPI router: all REST API endpoints for the dashboard and monitoring.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthService, get_current_user
from app.db.database import get_db
from app.models.trade import Trade
from app.models.order import Order
from app.models.equity_snapshot import EquitySnapshot
from app.models.error_event import ErrorEvent
from app.models.notification_log import NotificationLog
from app.models.strategy_health import StrategyHealth
from app.models.cooldown_state import CooldownState
from app.models.report import Report
from app.models.signal import Signal

router = APIRouter()
auth_service = AuthService()


# ------------------------------------------------------------------ Schemas

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mode: str


# ------------------------------------------------------------------ Auth

@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(body: LoginRequest) -> TokenResponse:
    token = auth_service.authenticate(body.username, body.password)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=token)


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def login_form(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    token = auth_service.authenticate(form.username, form.password)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(access_token=token)


# ------------------------------------------------------------------ Health

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    from app.config.settings import get_settings
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=get_settings().bot_mode,
    )


@router.get("/status", tags=["system"])
async def system_status(username: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Full system status snapshot."""
    from app.config.settings import get_settings
    settings = get_settings()
    return {
        "mode": settings.bot_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# ------------------------------------------------------------------ Positions

@router.get("/positions", tags=["trading"])
async def get_positions(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Trade).where(Trade.status == "open").order_by(desc(Trade.entry_time))
    )
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


# ------------------------------------------------------------------ Orders

@router.get("/orders", tags=["trading"])
async def get_orders(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 50,
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Order).where(Order.status.in_(["open", "pending", "partial"]))
        .order_by(desc(Order.placed_at)).limit(limit)
    )
    orders = result.scalars().all()
    return [_order_to_dict(o) for o in orders]


# ------------------------------------------------------------------ Trades/Fills

@router.get("/trades", tags=["trading"])
async def get_recent_trades(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 50,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = select(Trade).order_by(desc(Trade.entry_time)).limit(limit)
    if symbol:
        query = query.where(Trade.symbol == symbol)
    result = await db.execute(query)
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


# ------------------------------------------------------------------ Equity / PnL

@router.get("/equity", tags=["metrics"])
async def get_equity_curve(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    days: int = 30,
) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(EquitySnapshot)
        .where(EquitySnapshot.snapshot_time >= cutoff)
        .order_by(EquitySnapshot.snapshot_time)
    )
    snapshots = result.scalars().all()
    return [
        {
            "time": s.snapshot_time.isoformat(),
            "total_equity": s.total_equity,
            "drawdown_pct": s.drawdown_pct,
            "realized_pnl_total": s.realized_pnl_total,
        }
        for s in snapshots
    ]


@router.get("/metrics", tags=["metrics"])
async def get_metrics(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Aggregate metrics for dashboard."""
    # Latest snapshot
    result = await db.execute(
        select(EquitySnapshot).order_by(desc(EquitySnapshot.snapshot_time)).limit(1)
    )
    snap = result.scalar_one_or_none()

    # Trade stats
    closed_result = await db.execute(
        select(Trade).where(Trade.status == "closed")
    )
    closed_trades = closed_result.scalars().all()
    pnls = [t.realized_pnl or 0.0 for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        "total_equity": snap.total_equity if snap else 0.0,
        "daily_pnl": snap.realized_pnl_today if snap else 0.0,
        "total_pnl": snap.realized_pnl_total if snap else 0.0,
        "drawdown_pct": snap.drawdown_pct if snap else 0.0,
        "open_positions": snap.open_position_count if snap else 0,
        "total_trades": len(closed_trades),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


# ------------------------------------------------------------------ Strategy health

@router.get("/strategy-health", tags=["metrics"])
async def get_strategy_health(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(StrategyHealth)
        .order_by(StrategyHealth.strategy_name, desc(StrategyHealth.computed_at))
    )
    rows = result.scalars().all()
    seen = set()
    health_list = []
    for row in rows:
        if row.strategy_name not in seen:
            seen.add(row.strategy_name)
            health_list.append({
                "strategy": row.strategy_name,
                "win_rate": row.win_rate,
                "profit_factor": row.profit_factor,
                "expectancy": row.expectancy,
                "window_trades": row.window_trades,
                "consecutive_losses": row.consecutive_losses,
                "is_disabled": row.is_disabled,
                "disabled_until": row.disabled_until.isoformat() if row.disabled_until else None,
            })
    return health_list


# ------------------------------------------------------------------ Signals

@router.get("/signals", tags=["trading"])
async def get_recent_signals(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 20,
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Signal).order_by(desc(Signal.signal_time)).limit(limit)
    )
    signals = result.scalars().all()
    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "strategy": s.strategy_family,
            "direction": s.direction,
            "score": s.final_score,
            "tier": s.score_tier,
            "time": s.signal_time.isoformat(),
            "acted_on": s.acted_on,
        }
        for s in signals
    ]


# ------------------------------------------------------------------ Logs / Events

@router.get("/logs", tags=["monitoring"])
async def get_error_events(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 50,
    severity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = select(ErrorEvent).order_by(desc(ErrorEvent.occurred_at)).limit(limit)
    if severity:
        query = query.where(ErrorEvent.severity == severity)
    result = await db.execute(query)
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "severity": e.severity,
            "component": e.component,
            "message": e.message,
            "time": e.occurred_at.isoformat(),
            "kill_switch": e.kill_switch_triggered,
        }
        for e in events
    ]


@router.get("/notifications", tags=["monitoring"])
async def get_notifications(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 30,
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(NotificationLog).order_by(desc(NotificationLog.sent_at)).limit(limit)
    )
    notifs = result.scalars().all()
    return [
        {
            "event": n.event_type,
            "message": n.message[:200],
            "time": n.sent_at.isoformat(),
            "success": n.success,
        }
        for n in notifs
    ]


# ------------------------------------------------------------------ Reports

@router.get("/reports", tags=["reporting"])
async def get_reports(
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
    limit: int = 10,
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Report).order_by(desc(Report.generated_at)).limit(limit)
    )
    reports = result.scalars().all()
    return [
        {
            "id": r.id,
            "type": r.report_type,
            "generated_at": r.generated_at.isoformat(),
            "content": r.content[:500],
        }
        for r in reports
    ]


@router.get("/reports/{report_id}", tags=["reporting"])
async def get_report(
    report_id: str,
    db: AsyncSession = Depends(get_db),
    username: str = Depends(get_current_user),
) -> Dict[str, Any]:
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"id": report.id, "type": report.report_type,
            "generated_at": report.generated_at.isoformat(), "content": report.content}


# ------------------------------------------------------------------ Manual controls

@router.post("/control/pause", tags=["control"])
async def pause_trading(username: str = Depends(get_current_user)) -> Dict[str, str]:
    """Pause new trade opens. Does not close existing positions."""
    # TODO: Signal trading worker to pause via shared state or DB flag
    return {"status": "paused", "note": "Implementation: set pause flag in shared state"}


@router.post("/control/resume", tags=["control"])
async def resume_trading(username: str = Depends(get_current_user)) -> Dict[str, str]:
    return {"status": "resumed"}


@router.post("/control/disable-strategy/{strategy_name}", tags=["control"])
async def disable_strategy(
    strategy_name: str,
    username: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    """Manually disable a strategy. Respects safety boundaries."""
    allowed = ["trend_pullback", "breakout_retest", "liquidity_sweep_reversal", "volatility_expansion"]
    if strategy_name not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy_name}")
    # TODO: Write disable flag to DB and signal worker
    return {"status": "disabled", "strategy": strategy_name}


# ------------------------------------------------------------------ Helpers

def _trade_to_dict(t: Trade) -> Dict[str, Any]:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "direction": t.direction,
        "strategy": t.strategy_family,
        "score": t.signal_score,
        "tier": t.score_tier,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "entry_price": t.entry_price,
        "quantity": t.quantity,
        "notional_usd": t.notional_usd,
        "leverage": t.leverage,
        "stop_price": t.stop_loss_price,
        "tp_price": t.take_profit_price,
        "status": t.status,
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "realized_pnl": t.realized_pnl,
        "slippage_pct": t.slippage_pct,
        "regime": t.regime_at_entry,
    }


def _order_to_dict(o: Order) -> Dict[str, Any]:
    return {
        "id": o.id,
        "exchange_id": o.exchange_order_id,
        "symbol": o.symbol,
        "type": o.order_type,
        "side": o.side,
        "quantity": o.quantity,
        "price": o.price,
        "filled": o.filled_qty,
        "status": o.status,
        "role": o.role,
    }
