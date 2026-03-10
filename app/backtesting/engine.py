"""
Backtesting Engine

Simulates the full trading pipeline on historical data.
Shares the same strategy/voting/risk logic as live trading.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from app.config.trading_config import BacktestConfig, get_trading_config
from app.market_data.features import compute_all_features, get_atr_value, get_trend_direction, get_rsi_value, get_adx_value
from app.monitoring.session import SessionClassifier
from app.ml.regime_classifier import RegimeClassifier
from app.signals.schema import SignalDirection, SignalTier
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.breakout_retest import BreakoutRetestStrategy
from app.strategies.liquidity_sweep import LiquiditySweepReversalStrategy
from app.strategies.volatility_expansion import VolatilityExpansionStrategy
from app.voting.engine import VotingEngine
from app.risk.engine import RiskEngine

logger = structlog.get_logger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    strategy: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    quantity: float
    leverage: float
    stop_price: float
    tp_price: float
    capital_at_risk: float
    realized_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    regime: str = "unknown"
    score: float = 0.0
    hold_minutes: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_usd: float
    avg_loss_usd: float
    profit_factor: float
    expectancy_usd: float
    avg_hold_minutes: float
    total_fees: float
    total_slippage: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    performance_by_strategy: Dict[str, Dict] = field(default_factory=dict)
    performance_by_regime: Dict[str, Dict] = field(default_factory=dict)
    performance_by_session: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 3),
            "avg_win_usd": round(self.avg_win_usd, 2),
            "avg_loss_usd": round(self.avg_loss_usd, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_usd": round(self.expectancy_usd, 2),
            "avg_hold_minutes": round(self.avg_hold_minutes, 1),
            "total_fees": round(self.total_fees, 2),
            "total_slippage": round(self.total_slippage, 2),
            "performance_by_strategy": self.performance_by_strategy,
            "performance_by_regime": self.performance_by_regime,
            "performance_by_session": self.performance_by_session,
        }


class BacktestEngine:
    """
    Full strategy backtesting engine.
    Uses the same strategy/voting/risk pipeline as live mode.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        initial_capital: float = 10000.0,
    ) -> None:
        self._cfg = config or get_trading_config().backtest
        self._trading_cfg = get_trading_config()
        self._initial_capital = initial_capital
        self._session_classifier = SessionClassifier()
        self._regime_classifier = RegimeClassifier()
        self._voting_engine = VotingEngine()
        self._risk_engine = RiskEngine()

        # Strategy instances
        self._strategies = {
            "trend_pullback": TrendPullbackStrategy(),
            "breakout_retest": BreakoutRetestStrategy(),
            "liquidity_sweep_reversal": LiquiditySweepReversalStrategy(),
            "volatility_expansion": VolatilityExpansionStrategy(),
        }

    async def run(
        self,
        symbol: str,
        candles_by_tf: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run backtest for a single symbol.

        Args:
            symbol: trading symbol
            candles_by_tf: dict mapping timeframe -> full historical DataFrame
            start_date: backtest start
            end_date: backtest end
        """
        cfg = self._cfg
        fee_pct = cfg.fee_pct / 100
        slippage_pct = cfg.slippage_pct / 100
        latency_candles = max(1, cfg.assumed_latency_ms // 60000 + 1)  # in 1m candles

        # Use 5m candles as primary timeframe for bar-by-bar iteration
        df_primary = candles_by_tf.get("5m")
        if df_primary is None or len(df_primary) < 100:
            logger.error("backtest_insufficient_data", symbol=symbol)
            return self._empty_result(symbol, str(start_date), str(end_date))

        if start_date:
            df_primary = df_primary[df_primary.index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df_primary = df_primary[df_primary.index <= pd.Timestamp(end_date, tz="UTC")]

        logger.info("backtest_started", symbol=symbol, bars=len(df_primary),
                   start=df_primary.index[0].isoformat() if len(df_primary) > 0 else "")

        equity = self._initial_capital
        peak_equity = equity
        max_drawdown_pct = 0.0
        equity_curve: List[Tuple[datetime, float]] = [(df_primary.index[0].to_pydatetime(), equity)]
        trades: List[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None

        # Warm-up period
        warmup = 100

        for i in range(warmup, len(df_primary)):
            bar_time = df_primary.index[i].to_pydatetime()
            current_price = float(df_primary.iloc[i]["close"])

            # Check open trade exit
            if open_trade is not None:
                open_trade, exit_happened = self._check_exits(
                    open_trade, df_primary.iloc[i], fee_pct, slippage_pct
                )
                if exit_happened:
                    equity += open_trade.realized_pnl
                    if equity > peak_equity:
                        peak_equity = equity
                    dd = (peak_equity - equity) / peak_equity * 100
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd
                    trades.append(open_trade)
                    open_trade = None
                    equity_curve.append((bar_time, equity))

            # Skip if we have an open trade (single-position backtest)
            if open_trade is not None:
                continue

            # Build multi-timeframe candles slice up to current bar
            candles_slice = self._build_candles_slice(candles_by_tf, bar_time)
            if not candles_slice or len(candles_slice.get("5m", pd.DataFrame())) < 50:
                continue

            # Compute features
            for tf, df in candles_slice.items():
                candles_slice[tf] = compute_all_features(df)

            # Regime classification
            regime_result = self._regime_classifier.classify(candles_slice.get("5m"))
            regime = regime_result.regime

            # Session
            session = self._session_classifier.classify(bar_time)
            if self._trading_cfg.session.hard_block_off_hours and not session.is_active:
                continue

            # Strategy evaluation
            signals = []
            for name, strategy in self._strategies.items():
                cfg_enable = getattr(self._trading_cfg.strategy_enable, name, True)
                if not cfg_enable:
                    continue
                try:
                    sig = await strategy.evaluate(symbol, candles_slice, current_price, regime)
                    if sig:
                        signals.append(sig)
                except Exception:
                    pass

            if not signals:
                continue

            # Voting for each direction
            for direction in [SignalDirection.LONG, SignalDirection.SHORT]:
                dir_signals = [s for s in signals if s.direction == direction]
                if not dir_signals:
                    continue

                # Build context
                df_5m_enriched = candles_slice.get("5m")
                df_4h_enriched = candles_slice.get("4h")
                context = {
                    "trend_4h": get_trend_direction(df_4h_enriched) if df_4h_enriched is not None else "neutral",
                    "trend_15m": get_trend_direction(candles_slice.get("15m")) if candles_slice.get("15m") is not None else "neutral",
                    "rsi": get_rsi_value(df_5m_enriched),
                    "adx": get_adx_value(df_5m_enriched),
                    "volatility_state": df_5m_enriched.iloc[-1].get("volatility_state", "medium") if df_5m_enriched is not None else "medium",
                    "spread_pct": 0.01,
                    "volume_24h_usd": 1_000_000_000,
                    "order_book_depth_usd": 500_000,
                    "session_score": session.quality_score,
                    "slippage_estimate_pct": slippage_pct * 100,
                }

                vote = self._voting_engine.vote(symbol, signals, direction, context)

                if vote.tier == "no_trade":
                    continue

                # Risk sizing
                best_signal = dir_signals[0]
                stop = best_signal.stop_price
                tp = best_signal.tp_price

                if not stop or not tp:
                    continue

                portfolio_state = {
                    "total_equity": equity,
                    "allocated_capital": 0.0,
                    "daily_pnl": 0.0,
                    "peak_equity": peak_equity,
                    "open_positions": [],
                    "consecutive_losses": 0,
                    "drawdown_pct": (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0,
                    "risk_mode": "normal",
                }
                market_context = {"slippage_estimate_pct": slippage_pct * 100}

                risk_result = self._risk_engine.evaluate(
                    vote, current_price, stop, tp, portfolio_state, market_context
                )

                if risk_result.decision == "reject":
                    continue

                # Apply latency offset to entry price
                entry_price = current_price * (1 + slippage_pct if direction == SignalDirection.LONG else 1 - slippage_pct)
                fees_entry = entry_price * risk_result.approved_quantity * fee_pct

                open_trade = BacktestTrade(
                    symbol=symbol,
                    direction=direction,
                    strategy=best_signal.strategy_family,
                    entry_time=bar_time,
                    entry_price=entry_price,
                    exit_time=None,
                    exit_price=None,
                    exit_reason=None,
                    quantity=risk_result.approved_quantity,
                    leverage=risk_result.approved_leverage,
                    stop_price=risk_result.approved_stop_price,
                    tp_price=risk_result.approved_tp_price,
                    capital_at_risk=risk_result.capital_at_risk,
                    fees=fees_entry,
                    regime=regime,
                    score=vote.final_score,
                )
                break  # one direction per bar

        # Close any remaining open trade at end
        if open_trade is not None:
            last_price = float(df_primary.iloc[-1]["close"])
            if open_trade.direction == SignalDirection.LONG:
                pnl = (last_price - open_trade.entry_price) * open_trade.quantity
            else:
                pnl = (open_trade.entry_price - last_price) * open_trade.quantity
            fees_exit = last_price * open_trade.quantity * fee_pct
            open_trade.realized_pnl = pnl - fees_exit - open_trade.fees
            open_trade.fees += fees_exit
            open_trade.exit_price = last_price
            open_trade.exit_time = df_primary.index[-1].to_pydatetime()
            open_trade.exit_reason = "end_of_backtest"
            open_trade.hold_minutes = (open_trade.exit_time - open_trade.entry_time).total_seconds() / 60
            trades.append(open_trade)
            equity += open_trade.realized_pnl

        return self._compute_result(symbol, trades, equity, peak_equity, max_drawdown_pct, equity_curve)

    def _check_exits(
        self,
        trade: BacktestTrade,
        bar: pd.Series,
        fee_pct: float,
        slippage_pct: float,
    ) -> Tuple[BacktestTrade, bool]:
        """Check if a bar hits SL or TP."""
        is_long = trade.direction == SignalDirection.LONG

        if is_long:
            sl_hit = bar["low"] <= trade.stop_price
            tp_hit = bar["high"] >= trade.tp_price
        else:
            sl_hit = bar["high"] >= trade.stop_price
            tp_hit = bar["low"] <= trade.tp_price

        if not sl_hit and not tp_hit:
            return trade, False

        # Determine which hit first (assume worst case: SL if both)
        if sl_hit and tp_hit:
            exit_price = trade.stop_price  # conservative
            exit_reason = "stop_loss"
        elif sl_hit:
            exit_price = trade.stop_price
            exit_reason = "stop_loss"
        else:
            exit_price = trade.tp_price
            exit_reason = "take_profit"

        # Apply slippage on exit
        if exit_reason == "stop_loss":
            exit_price *= (1 - slippage_pct) if is_long else (1 + slippage_pct)

        if is_long:
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        fees_exit = exit_price * trade.quantity * fee_pct
        trade.realized_pnl = pnl - fees_exit - trade.fees
        trade.fees += fees_exit
        trade.exit_price = exit_price
        trade.exit_time = bar.name.to_pydatetime() if hasattr(bar.name, 'to_pydatetime') else datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.hold_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60

        return trade, True

    def _build_candles_slice(
        self,
        candles_by_tf: Dict[str, pd.DataFrame],
        up_to: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Return historical candles up to the given timestamp."""
        result = {}
        ts = pd.Timestamp(up_to, tz="UTC")
        for tf, df in candles_by_tf.items():
            subset = df[df.index <= ts]
            if len(subset) >= 30:
                result[tf] = subset.tail(300).copy()
        return result

    def _compute_result(
        self,
        symbol: str,
        trades: List[BacktestTrade],
        final_equity: float,
        peak_equity: float,
        max_dd_pct: float,
        equity_curve: List,
    ) -> BacktestResult:
        if not trades:
            return self._empty_result(symbol, "", "")

        pnls = [t.realized_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Sharpe ratio (annualized, assuming daily returns from equity curve)
        daily_pnl_arr = np.array(pnls)
        sharpe = float(np.mean(daily_pnl_arr) / (np.std(daily_pnl_arr) + 1e-8) * np.sqrt(252)) if len(daily_pnl_arr) > 1 else 0.0

        total_return_pct = (final_equity - self._initial_capital) / self._initial_capital * 100

        # Per-strategy breakdown
        by_strategy: Dict[str, Dict] = {}
        for t in trades:
            s = t.strategy
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["pnl"] += t.realized_pnl
            if t.realized_pnl > 0:
                by_strategy[s]["wins"] += 1

        # Per-regime breakdown
        by_regime: Dict[str, Dict] = {}
        for t in trades:
            r = t.regime
            if r not in by_regime:
                by_regime[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_regime[r]["trades"] += 1
            by_regime[r]["pnl"] += t.realized_pnl
            if t.realized_pnl > 0:
                by_regime[r]["wins"] += 1

        start_str = trades[0].entry_time.date().isoformat() if trades else ""
        end_str = trades[-1].exit_time.date().isoformat() if trades and trades[-1].exit_time else ""

        return BacktestResult(
            symbol=symbol,
            start_date=start_str,
            end_date=end_str,
            initial_capital=self._initial_capital,
            final_capital=final_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win_usd=float(avg_win),
            avg_loss_usd=float(avg_loss),
            profit_factor=profit_factor,
            expectancy_usd=expectancy,
            avg_hold_minutes=float(np.mean([t.hold_minutes for t in trades])),
            total_fees=float(sum(t.fees for t in trades)),
            total_slippage=float(sum(t.slippage for t in trades)),
            trades=trades,
            equity_curve=equity_curve,
            performance_by_strategy=by_strategy,
            performance_by_regime=by_regime,
        )

    def _empty_result(self, symbol: str, start: str, end: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, start_date=start, end_date=end,
            initial_capital=self._initial_capital, final_capital=self._initial_capital,
            total_return_pct=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            avg_win_usd=0.0, avg_loss_usd=0.0, profit_factor=0.0,
            expectancy_usd=0.0, avg_hold_minutes=0.0, total_fees=0.0, total_slippage=0.0,
        )
