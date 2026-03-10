"""
Trading Worker

The core trading loop. Runs as a separate service from the dashboard.
Handles: market data refresh, signal evaluation, voting, risk, execution, monitoring.

Run with: python -m app.main --mode paper
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config
from app.db.database import AsyncSessionLocal, init_db
from app.exchange.factory import create_exchange_adapter
from app.exchange.paper_exchange import PaperExchangeAdapter
from app.market_data.candles import CandleManager
from app.market_data.features import (
    compute_all_features, get_atr_value, get_trend_direction,
    get_rsi_value, get_adx_value,
)
from app.market_data.service import MarketDataService
from app.ml.regime_classifier import RegimeClassifier
from app.ml.quality_scorer import TradeQualityScorer
from app.ml.strategy_monitor import StrategyPerformanceMonitor
from app.monitoring.session import SessionClassifier
from app.monitoring.kill_switch import KillSwitch
from app.notifications.discord import DiscordNotifier
from app.portfolio.state import PortfolioManager
from app.risk.engine import RiskEngine
from app.execution.engine import ExecutionEngine
from app.signals.schema import SignalDirection
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.breakout_retest import BreakoutRetestStrategy
from app.strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from app.strategies.volatility_expansion import VolatilityExpansionStrategy
from app.voting.engine import VotingEngine
from app.models.signal import Signal
from app.models.score_breakdown import ScoreBreakdown
import json

logger = structlog.get_logger(__name__)


class TradingWorker:
    """
    Main trading loop worker.
    Orchestrates all layers of the trading system.
    """

    LOOP_INTERVAL_SECONDS = 30  # evaluate signals every 30 seconds

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode
        self._settings = get_settings()
        self._cfg = get_trading_config()
        self._running = False

    async def run(self) -> None:
        """Main entry point. Initializes everything and starts the loop."""
        logger.info("trading_worker_starting", mode=self._mode)

        # Initialize DB
        await init_db()

        async with AsyncSessionLocal() as db:
            # Set up exchange adapter
            if self._mode == "paper":
                # Paper mode: use real data but simulated orders
                real_adapter = create_exchange_adapter("coinbase")
                paper_adapter = PaperExchangeAdapter()
                # Try to connect to real exchange for market data
                try:
                    if self._settings.coinbase_api_key:
                        await real_adapter.connect()
                        paper_adapter.set_market_adapter(real_adapter)
                    exchange = paper_adapter
                    await exchange.connect()
                except Exception as e:
                    logger.warning("exchange_connect_failed_using_paper_only", error=str(e))
                    exchange = paper_adapter
                    await exchange.connect()
            else:
                exchange = create_exchange_adapter("coinbase")
                await exchange.connect()

            # Initialize components
            candle_mgr = CandleManager(exchange, db)
            market_data = MarketDataService(exchange, candle_mgr)

            portfolio = PortfolioManager(db, mode=self._mode)
            await portfolio.initialize()

            kill_switch = KillSwitch(db, portfolio)
            discord = DiscordNotifier(db)

            strategy_monitor = StrategyPerformanceMonitor(db, mode=self._mode)
            await strategy_monitor.refresh()

            regime_classifier = RegimeClassifier()
            regime_classifier.load_model()

            quality_scorer = TradeQualityScorer()
            quality_scorer.load_model()

            session_classifier = SessionClassifier()
            voting_engine = VotingEngine()
            risk_engine = RiskEngine()

            execution_engine = ExecutionEngine(
                adapter=exchange,
                db=db,
                portfolio=portfolio,
                mode=self._mode,
            )

            # Strategy instances
            strategies = {
                "trend_pullback": TrendPullbackStrategy(),
                "breakout_retest": BreakoutRetestStrategy(),
                "liquidity_sweep_reversal": LiquiditySweepReversalStrategy(),
                "volatility_expansion": VolatilityExpansionStrategy(),
            }

            # Pre-load market data
            await market_data.initialize()
            await market_data.start_streaming()

            # Notify startup
            await discord.bot_started(self._mode)

            self._running = True
            logger.info("trading_worker_started", mode=self._mode)

            try:
                # Periodic tasks counters
                loop_count = 0
                equity_snapshot_interval = 60  # every 60 loops = ~30 min
                strategy_monitor_interval = 120  # every 60 min

                while self._running:
                    loop_start = datetime.now(timezone.utc)
                    loop_count += 1

                    # Kill switch check
                    await kill_switch.check_conditions()
                    if kill_switch.is_triggered:
                        logger.critical("trading_halted_kill_switch", reason=kill_switch.reason)
                        await asyncio.sleep(60)
                        continue

                    # Periodic equity snapshot
                    if loop_count % equity_snapshot_interval == 0:
                        await portfolio.save_equity_snapshot()

                    # Periodic strategy monitor refresh
                    if loop_count % strategy_monitor_interval == 0:
                        await strategy_monitor.refresh()

                    # Process each symbol
                    for symbol in self._cfg.universe.symbols:
                        try:
                            await self._process_symbol(
                                symbol=symbol,
                                market_data=market_data,
                                session_classifier=session_classifier,
                                regime_classifier=regime_classifier,
                                quality_scorer=quality_scorer,
                                strategy_monitor=strategy_monitor,
                                strategies=strategies,
                                voting_engine=voting_engine,
                                risk_engine=risk_engine,
                                execution_engine=execution_engine,
                                portfolio=portfolio,
                                discord=discord,
                                kill_switch=kill_switch,
                                db=db,
                            )
                        except Exception as e:
                            logger.error("symbol_processing_error", symbol=symbol, error=str(e))

                    # Sleep remainder of loop interval
                    elapsed = (datetime.now(timezone.utc) - loop_start).total_seconds()
                    sleep_time = max(0.0, self.LOOP_INTERVAL_SECONDS - elapsed)
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("trading_worker_cancelled")
            except Exception as e:
                logger.critical("trading_worker_fatal_error", error=str(e))
                await discord.error_alert("trading_worker", str(e))
                raise
            finally:
                self._running = False
                await market_data.stop_streaming()
                await exchange.disconnect()
                await discord.bot_stopped("shutdown")
                logger.info("trading_worker_stopped")

    async def _process_symbol(
        self,
        symbol: str,
        market_data: MarketDataService,
        session_classifier: SessionClassifier,
        regime_classifier: RegimeClassifier,
        quality_scorer: TradeQualityScorer,
        strategy_monitor: StrategyPerformanceMonitor,
        strategies: Dict,
        voting_engine: VotingEngine,
        risk_engine: RiskEngine,
        execution_engine: ExecutionEngine,
        portfolio: PortfolioManager,
        discord: DiscordNotifier,
        kill_switch: KillSwitch,
        db,
    ) -> None:
        """Process one symbol: evaluate signals and potentially execute a trade."""

        # Get current price
        ticker = market_data.get_ticker(symbol)
        if ticker is None:
            return
        current_price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else ticker.last
        if current_price <= 0:
            return

        # Session filter
        session = session_classifier.classify()
        if self._cfg.session.hard_block_off_hours and not session.is_active:
            return

        # Check global cooldown
        if await portfolio.check_cooldown_active("global"):
            return

        # Check symbol cooldown
        if await portfolio.check_cooldown_active("symbol", symbol):
            return

        # Gather candles for all timeframes
        candles: Dict[str, Any] = {}
        for tf in self._cfg.timeframes.enabled:
            df = market_data.get_candles(symbol, tf)
            if df is not None and len(df) >= 30:
                candles[tf] = compute_all_features(df)

        if not candles:
            return

        # Regime classification
        df_primary = candles.get("5m") or candles.get("15m")
        if df_primary is None:
            return

        regime_result = regime_classifier.classify(df_primary)
        regime = regime_result.regime

        # Get allowed strategies for this regime
        regime_map = self._cfg.regime_strategy_map
        allowed_in_regime = getattr(regime_map, regime, regime_map.unknown)

        # Evaluate each enabled strategy
        all_signals = []
        for name, strategy in strategies.items():
            if not getattr(self._cfg.strategy_enable, name, True):
                continue
            if name not in allowed_in_regime:
                continue
            if strategy_monitor.is_disabled(name):
                continue

            try:
                sig = await strategy.evaluate(symbol, candles, current_price, regime)
                if sig:
                    all_signals.append(sig)
                    if self._cfg.verbose_signal_logging:
                        logger.debug("signal_generated", symbol=symbol,
                                    strategy=name, direction=sig.direction,
                                    confidence=sig.confidence)
            except Exception as e:
                logger.error("strategy_evaluate_error", strategy=name, symbol=symbol, error=str(e))

        if not all_signals:
            return

        # Build context for voting
        df_4h = candles.get("4h")
        df_15m = candles.get("15m")
        df_5m = candles.get("5m")

        context = {
            "trend_4h": get_trend_direction(df_4h) if df_4h is not None else "neutral",
            "trend_15m": get_trend_direction(df_15m) if df_15m is not None else "neutral",
            "rsi": get_rsi_value(df_5m) if df_5m is not None else 50.0,
            "adx": get_adx_value(df_5m) if df_5m is not None else 0.0,
            "volatility_state": df_5m.iloc[-1].get("volatility_state", "medium") if df_5m is not None else "medium",
            "spread_pct": market_data.get_spread_pct(symbol),
            "volume_24h_usd": ticker.volume_24h if ticker else 0.0,
            "order_book_depth_usd": 500_000,  # TODO: compute from actual order book
            "session_score": session.quality_score,
            "slippage_estimate_pct": self._cfg.execution.slippage_estimate_basis_points / 100,
            "macd_hist": float(df_5m.iloc[-1].get("macd_hist", 0)) if df_5m is not None else 0.0,
        }

        # Vote for each direction
        best_vote = None
        for direction in [SignalDirection.LONG, SignalDirection.SHORT]:
            dir_signals = [s for s in all_signals if s.direction == direction]
            if not dir_signals:
                continue

            # ML quality score
            ml_score = None
            if quality_scorer._loaded:
                features = quality_scorer.build_feature_vector(
                    trend_alignment=context["trend_4h"] == ("bullish" if direction == SignalDirection.LONG else "bearish"),
                    momentum=context["rsi"] / 100.0,
                    volatility_state=str(context["volatility_state"]),
                    spread_pct=context["spread_pct"],
                    session_score=context["session_score"],
                    signal_score=max(s.confidence for s in dir_signals) * 100,
                    adx=context["adx"],
                    rsi=context["rsi"],
                    volume_ratio=1.0,
                    regime=regime,
                    strategy_family=dir_signals[0].strategy_family,
                )
                ml_score = quality_scorer.score(features)

            strategy_health = strategy_monitor.get_all_health_dict()
            vote = voting_engine.vote(symbol, all_signals, direction, context,
                                      strategy_health=strategy_health,
                                      ml_quality_score=ml_score)

            if best_vote is None or vote.final_score > best_vote.final_score:
                best_vote = vote

        if best_vote is None or best_vote.tier == "no_trade":
            return

        # Save signal to DB
        best_signal = best_vote.contributing_signals[0] if best_vote.contributing_signals else None
        if best_signal:
            db_signal = Signal(
                symbol=symbol,
                strategy_family=best_signal.strategy_family,
                direction=best_vote.direction,
                confidence=max(s.confidence for s in best_vote.contributing_signals),
                final_score=best_vote.final_score,
                score_tier=best_vote.tier,
                entry_price_suggestion=best_signal.entry_price,
                stop_suggestion=best_signal.stop_price,
                tp_suggestion=best_signal.tp_price,
                regime=regime,
                session=session.name,
                score_breakdown_json=json.dumps(best_vote.to_dict()),
                reason_codes_json=json.dumps(best_vote.reason_codes),
                mode=self._mode,
            )
            db.add(db_signal)
            try:
                await db.commit()
            except Exception:
                await db.rollback()

        # Risk check
        stop_price = best_signal.stop_price if best_signal else 0.0
        tp_price = best_signal.tp_price if best_signal else 0.0

        if not stop_price or not tp_price:
            return

        portfolio_state = portfolio.get_state().to_dict()
        portfolio_state["open_positions"] = [
            {"symbol": p.symbol, "notional": p.notional_usd}
            for p in portfolio.get_state().open_positions
        ]

        risk_result = risk_engine.evaluate(
            best_vote, current_price, stop_price, tp_price,
            portfolio_state, {"slippage_estimate_pct": context["slippage_estimate_pct"]},
        )

        if risk_result.decision == "reject":
            logger.info("trade_rejected_by_risk", symbol=symbol,
                       reason=risk_result.reason, score=best_vote.final_score)
            return

        # Execute trade
        exec_result = await execution_engine.execute_trade(
            vote_result=best_vote,
            risk_result=risk_result,
            current_price=current_price,
        )

        if exec_result.status.value == "success":
            # Notify Discord
            await discord.trade_opened(
                symbol=symbol,
                direction=best_vote.direction,
                entry_price=exec_result.fill_price or current_price,
                size=exec_result.fill_quantity or risk_result.approved_quantity,
                stop_price=risk_result.approved_stop_price,
                tp_price=risk_result.approved_tp_price,
                score=best_vote.final_score,
                strategy=best_signal.strategy_family if best_signal else "unknown",
            )

            # Mark signal as acted on
            if db_signal:
                db_signal.acted_on = True
                db_signal.trade_id = exec_result.trade_id
                try:
                    await db.commit()
                except Exception:
                    await db.rollback()

            # Check if cooldown should be triggered after consecutive losses
            state = portfolio.get_state()
            cooldown_cfg = self._cfg.cooldown
            if state.consecutive_losses >= cooldown_cfg.consecutive_losses_long:
                await portfolio.create_cooldown(
                    "global", None,
                    f"consecutive_losses_{state.consecutive_losses}",
                    cooldown_cfg.pause_long_hours,
                    state.consecutive_losses,
                )
                await discord.error_alert("cooldown",
                    f"Global cooldown: {state.consecutive_losses} consecutive losses. "
                    f"Paused for {cooldown_cfg.pause_long_hours}h")
            elif state.consecutive_losses >= cooldown_cfg.consecutive_losses_short:
                await portfolio.create_cooldown(
                    "global", None,
                    f"consecutive_losses_{state.consecutive_losses}",
                    cooldown_cfg.pause_short_hours,
                    state.consecutive_losses,
                )
        else:
            logger.warning("execution_failed", symbol=symbol, status=exec_result.status,
                          reason=exec_result.reason)

    def stop(self) -> None:
        self._running = False
