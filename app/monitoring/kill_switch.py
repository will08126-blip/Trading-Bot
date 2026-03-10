"""
Kill Switch System

Monitors for critical conditions and stops trading if triggered.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.trading_config import get_trading_config
from app.models.error_event import ErrorEvent

logger = structlog.get_logger(__name__)


class KillSwitch:
    """
    Monitors critical conditions and sets the kill switch flag.
    Once triggered, only manual intervention or config change resets it.
    """

    def __init__(self, db: AsyncSession, portfolio_manager) -> None:
        self._db = db
        self._portfolio = portfolio_manager
        self._cfg = get_trading_config()
        self._triggered = False
        self._trigger_reason: Optional[str] = None
        self._trigger_time: Optional[datetime] = None

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> Optional[str]:
        return self._trigger_reason

    async def check_and_trigger(
        self,
        reason: str,
        details: str = "",
        component: str = "unknown",
    ) -> bool:
        """
        Trigger the kill switch.
        Returns True if newly triggered.
        """
        if self._triggered:
            return False  # already triggered

        if not self._cfg.kill_switch_enabled:
            logger.warning("kill_switch_disabled", reason=reason)
            return False

        self._triggered = True
        self._trigger_reason = reason
        self._trigger_time = datetime.now(timezone.utc)

        logger.critical(
            "KILL_SWITCH_TRIGGERED",
            reason=reason,
            details=details,
            component=component,
            time=self._trigger_time.isoformat(),
        )

        # Update portfolio risk mode
        self._portfolio.get_state().risk_mode = "kill_switch"

        # Save to DB
        self._db.add(ErrorEvent(
            event_type="kill_switch",
            severity="critical",
            component=component,
            message=f"Kill switch triggered: {reason}",
            details=details,
            occurred_at=self._trigger_time,
            kill_switch_triggered=True,
        ))
        try:
            await self._db.commit()
        except Exception as e:
            await self._db.rollback()
            logger.error("kill_switch_db_error", error=str(e))

        return True

    async def check_conditions(self) -> None:
        """
        Periodically check all kill switch conditions.
        Called by the main trading loop.
        """
        state = self._portfolio.get_state()
        risk_cfg = self._cfg.risk

        # Drawdown check
        if state.drawdown_pct >= risk_cfg.max_account_drawdown_pct:
            await self.check_and_trigger(
                reason="max_drawdown_exceeded",
                details=f"Drawdown: {state.drawdown_pct:.1f}% >= {risk_cfg.max_account_drawdown_pct:.1f}%",
                component="risk_engine",
            )
            return

        # Daily loss check
        if state.total_equity > 0:
            daily_loss_pct = -state.realized_pnl_today / state.total_equity * 100
            if daily_loss_pct >= risk_cfg.max_daily_loss_pct:
                await self.check_and_trigger(
                    reason="daily_loss_limit",
                    details=f"Daily loss: {daily_loss_pct:.1f}% >= {risk_cfg.max_daily_loss_pct:.1f}%",
                    component="risk_engine",
                )

    def reset(self) -> None:
        """
        Reset the kill switch.
        Should only be called after manual operator review.
        """
        logger.warning("kill_switch_reset", previous_reason=self._trigger_reason)
        self._triggered = False
        self._trigger_reason = None
        self._trigger_time = None
        state = self._portfolio.get_state()
        if state.risk_mode == "kill_switch":
            state.risk_mode = "reduced"
