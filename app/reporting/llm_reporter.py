"""
LLM Reporting Layer (Claude API)

IMPORTANT SAFETY RULES:
- Claude NEVER places orders
- Claude NEVER overrides risk controls
- Claude NEVER changes leverage or strategy parameters
- Claude only generates readable summaries and analysis text

Usage: Analysis, summaries, anomaly explanations only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import structlog

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config

logger = structlog.get_logger(__name__)


class LLMReporter:
    """
    Uses Claude API to generate human-readable reports.
    All LLM outputs are TEXT ONLY - no trading decisions.
    Gracefully degrades if API is unavailable.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.anthropic_api_key
        self._model = settings.anthropic_model
        self._cfg = get_trading_config().llm
        self._enabled = bool(self._api_key) and self._cfg.enabled
        self._client = None
        self._last_report_time: Dict[str, datetime] = {}

        if not self._enabled:
            logger.info("llm_reporter_disabled", reason="no_api_key_or_disabled")
        else:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
                logger.info("llm_reporter_initialized", model=self._model)
            except ImportError:
                logger.warning("anthropic_not_installed")
                self._enabled = False
            except Exception as e:
                logger.warning("llm_reporter_init_failed", error=str(e))
                self._enabled = False

    def _is_cache_valid(self, report_type: str) -> bool:
        """Avoid regenerating reports too frequently."""
        last = self._last_report_time.get(report_type)
        if not last:
            return False
        cache_hours = self._cfg.cache_report_hours
        return (datetime.now(timezone.utc) - last).total_seconds() < cache_hours * 3600

    async def generate_daily_report(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Generate a daily performance summary."""
        if not self._enabled or not self._cfg.daily_report_enabled:
            return None
        if self._is_cache_valid("daily"):
            return None

        prompt = self._build_daily_report_prompt(metrics)
        report = await self._call_claude(prompt, "daily_report")
        if report:
            self._last_report_time["daily"] = datetime.now(timezone.utc)
        return report

    async def generate_weekly_report(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Generate a weekly performance summary."""
        if not self._enabled or not self._cfg.weekly_report_enabled:
            return None
        if self._is_cache_valid("weekly"):
            return None

        prompt = self._build_weekly_report_prompt(metrics)
        report = await self._call_claude(prompt, "weekly_report")
        if report:
            self._last_report_time["weekly"] = datetime.now(timezone.utc)
        return report

    async def generate_anomaly_summary(self, anomaly_data: Dict[str, Any]) -> Optional[str]:
        """Explain an anomalous trading event."""
        if not self._enabled or not self._cfg.anomaly_summary_enabled:
            return None

        prompt = self._build_anomaly_prompt(anomaly_data)
        return await self._call_claude(prompt, "anomaly")

    async def generate_status_narrative(self, status_data: Dict[str, Any]) -> Optional[str]:
        """Generate a brief narrative of current bot status."""
        if not self._enabled:
            return None

        prompt = self._build_status_prompt(status_data)
        return await self._call_claude(prompt, "status", max_tokens=500)

    async def _call_claude(
        self,
        prompt: str,
        report_type: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Call Claude API with error handling and graceful degradation."""
        if not self._client:
            return None

        max_tok = max_tokens or self._cfg.max_tokens_per_report
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tok,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text if response.content else ""
            logger.info("llm_report_generated",
                       report_type=report_type,
                       tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else 0)
            return content
        except Exception as e:
            if self._cfg.graceful_degrade_on_failure:
                logger.warning("llm_report_failed", report_type=report_type, error=str(e))
                return None
            raise

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_daily_report_prompt(self, metrics: Dict[str, Any]) -> str:
        return f"""You are the reporting module for an automated crypto futures trading bot.
Analyze the following trading metrics from today and provide a concise, professional daily summary.

IMPORTANT: Your role is ANALYSIS ONLY. You must NOT suggest changing risk controls, leverage settings,
or strategy parameters. You may explain what happened and identify patterns.

Today's metrics:
{json.dumps(metrics, indent=2, default=str)}

Provide a summary covering:
1. Overall day performance (PnL, win rate, trade count)
2. Which strategies performed well vs. poorly
3. Session quality observations
4. Any notable patterns in winning vs. losing trades
5. Risk utilization summary
6. One or two brief observations for informational purposes only

Keep it concise (under 400 words) and factual. Do not speculate about trades to take tomorrow.
Do not override any risk or strategy settings."""

    def _build_weekly_report_prompt(self, metrics: Dict[str, Any]) -> str:
        return f"""You are the reporting module for an automated crypto futures trading bot.
Analyze the following weekly trading metrics and provide a comprehensive weekly summary.

IMPORTANT: Your role is ANALYSIS ONLY. Do not suggest changing risk controls, leverage,
or strategy parameters.

Weekly metrics:
{json.dumps(metrics, indent=2, default=str)}

Provide a summary covering:
1. Weekly PnL, return %, max drawdown
2. Strategy breakdown (which worked best/worst and why based on data)
3. Regime analysis (what market regimes dominated)
4. Session analysis
5. Slippage and execution quality
6. Equity curve observations

Keep factual and analytical. Maximum 600 words."""

    def _build_anomaly_prompt(self, anomaly_data: Dict[str, Any]) -> str:
        return f"""You are the reporting module for an automated crypto futures trading bot.
An anomalous event was detected. Analyze and explain it clearly.

IMPORTANT: You are providing an EXPLANATION only. You cannot and must not suggest changing risk controls,
overriding safety systems, or altering trade parameters.

Anomaly data:
{json.dumps(anomaly_data, indent=2, default=str)}

Explain:
1. What likely happened
2. Why this is flagged as an anomaly
3. What it might indicate about market conditions
4. Whether this appears to be a systemic issue or one-off

Be concise (under 200 words). Do not recommend manual overrides of safety controls."""

    def _build_status_prompt(self, status_data: Dict[str, Any]) -> str:
        return f"""Provide a brief 2-3 sentence narrative description of the trading bot's current status
based on this data. Be factual and concise. Do not recommend any changes.

Status data:
{json.dumps(status_data, indent=2, default=str)}"""
