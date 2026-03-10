"""
Session classifier for trading liquidity windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, time as dtime
from typing import Optional

from app.config.trading_config import SessionConfig, get_trading_config


@dataclass
class SessionInfo:
    name: str           # asia | london | new_york | overlap | off_hours
    quality_score: float  # 0–1
    is_active: bool


class SessionClassifier:
    """
    Classifies the current trading session and returns a quality score.
    All times in UTC.
    """

    # Session time windows in UTC (start_hour, end_hour)
    SESSION_WINDOWS = {
        "asia": (0, 9),           # 00:00–09:00 UTC
        "london": (7, 16),        # 07:00–16:00 UTC
        "new_york": (13, 22),     # 13:00–22:00 UTC
        "overlap": (13, 16),      # London/NY overlap
    }

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self._cfg = config or get_trading_config().session

    def classify(self, dt: Optional[datetime] = None) -> SessionInfo:
        """Classify the current or given datetime into a session."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour
        cfg = self._cfg

        # Check overlap first (highest quality)
        overlap_start, overlap_end = self.SESSION_WINDOWS["overlap"]
        if overlap_start <= hour < overlap_end:
            return SessionInfo(
                name="overlap",
                quality_score=cfg.overlap_weight / 1.5,  # normalize to 0–1 range
                is_active=True,
            )

        # Check NY session
        ny_start, ny_end = self.SESSION_WINDOWS["new_york"]
        if ny_start <= hour < ny_end:
            return SessionInfo(
                name="new_york",
                quality_score=min(1.0, cfg.ny_open_weight / 1.4),
                is_active=True,
            )

        # Check London session
        lon_start, lon_end = self.SESSION_WINDOWS["london"]
        if lon_start <= hour < lon_end:
            return SessionInfo(
                name="london",
                quality_score=min(1.0, cfg.london_open_weight / 1.3),
                is_active=True,
            )

        # Check Asia session
        asia_start, asia_end = self.SESSION_WINDOWS["asia"]
        if asia_start <= hour < asia_end:
            return SessionInfo(
                name="asia",
                quality_score=cfg.asia_weight / 1.0,
                is_active=True,
            )

        # Off hours
        return SessionInfo(
            name="off_hours",
            quality_score=cfg.off_hours_weight / 1.0,
            is_active=not cfg.hard_block_off_hours,
        )

    def get_session_score(self, dt: Optional[datetime] = None) -> float:
        """Return just the quality score for the current session."""
        return self.classify(dt).quality_score

    def is_trading_allowed(self, dt: Optional[datetime] = None) -> bool:
        """Check if trading is allowed in current session."""
        info = self.classify(dt)
        if not info.is_active:
            return False
        if info.quality_score < self._cfg.min_session_score:
            return False
        return True
