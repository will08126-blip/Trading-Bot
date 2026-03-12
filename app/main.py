"""
Main entry point for the trading bot.

Usage:
    python -m app.main --mode paper
    python -m app.main --mode live
    python -m app.main --mode backtest --symbol BTC-PERP
    python -m app.main --mode retrain
    python -m app.main --mode report-daily
    python -m app.main --mode report-weekly
    python -m app.main --mode api   (start dashboard/API only)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_environment() -> None:
    """Load environment variables and configure logging."""
    from dotenv import load_dotenv
    load_dotenv()

    from app.config.settings import get_settings
    settings = get_settings()

    from app.monitoring.logger import setup_logging
    setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)


def run_api_server() -> None:
    """Start the FastAPI dashboard/API server."""
    import uvicorn

    async def create_app():
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from app.api.router import router
        from app.db.database import init_db

        app = FastAPI(
            title="Crypto Trading Bot",
            description="Automated crypto futures trading bot dashboard",
            version="1.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(router, prefix="/api/v1")

        @app.on_event("startup")
        async def startup():
            await init_db()

        return app

    import uvicorn
    import asyncio

    async def run():
        app = await create_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(run())


async def run_backtest(symbol: str, start_date: str = "", end_date: str = "") -> None:
    """Run a backtest for a symbol."""
    import structlog
    logger = structlog.get_logger(__name__)

    from app.db.database import AsyncSessionLocal, init_db
    from app.config.trading_config import get_trading_config
    from app.exchange.factory import create_exchange_adapter
    from app.market_data.candles import CandleManager
    from app.backtesting.engine import BacktestEngine
    from datetime import datetime, timezone

    await init_db()

    cfg = get_trading_config()
    bt_cfg = cfg.backtest
    start = start_date or bt_cfg.default_start_date
    end = end_date or bt_cfg.default_end_date

    logger.info("backtest_starting", symbol=symbol, start=start, end=end)

    async with AsyncSessionLocal() as db:
        exchange = create_exchange_adapter("coinbase")
        try:
            await exchange.connect()
        except Exception as e:
            logger.warning("backtest_exchange_connect_failed", error=str(e))

        candle_mgr = CandleManager(exchange, db)

        # Fetch historical data
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

        candles_by_tf = {}
        for tf in cfg.timeframes.enabled:
            try:
                df = await candle_mgr.get_candles(symbol, tf, limit=5000)
                if df is not None and len(df) > 0:
                    candles_by_tf[tf] = df
                    logger.info("candles_loaded", symbol=symbol, tf=tf, count=len(df))
            except Exception as e:
                logger.error("candle_fetch_error", symbol=symbol, tf=tf, error=str(e))

        if not candles_by_tf:
            logger.error("no_candles_for_backtest", symbol=symbol)
            return

        engine = BacktestEngine(initial_capital=10000.0)
        result = await engine.run(
            symbol=symbol,
            candles_by_tf=candles_by_tf,
            start_date=start_dt,
            end_date=end_dt,
        )

        print("\n" + "="*60)
        print(f"BACKTEST RESULT: {symbol}")
        print("="*60)
        import json
        print(json.dumps(result.to_dict(), indent=2))
        print("="*60)

        try:
            await exchange.disconnect()
        except Exception:
            pass


async def run_retrain() -> None:
    """Retrain ML models."""
    import structlog
    logger = structlog.get_logger(__name__)

    from app.db.database import AsyncSessionLocal, init_db
    from app.ml.retrainer import MLRetrainer

    await init_db()
    logger.info("retraining_ml_models")

    async with AsyncSessionLocal() as db:
        retrainer = MLRetrainer(db)
        results = await retrainer.retrain_all()
        logger.info("retraining_complete", results=results)
        print("Retraining results:", results)


async def run_report(report_type: str) -> None:
    """Generate and save a report."""
    import structlog
    logger = structlog.get_logger(__name__)

    from app.db.database import AsyncSessionLocal, init_db
    from app.reporting.llm_reporter import LLMReporter
    from app.models.report import Report
    from datetime import datetime, timezone

    await init_db()

    async with AsyncSessionLocal() as db:
        reporter = LLMReporter()

        metrics = {"report_type": report_type, "generated_at": datetime.now(timezone.utc).isoformat()}
        if report_type == "daily":
            content = await reporter.generate_daily_report(metrics)
        else:
            content = await reporter.generate_weekly_report(metrics)

        if content:
            db.add(Report(
                report_type=report_type,
                content=content,
                generated_at=datetime.now(timezone.utc),
                model_used=reporter._model,
            ))
            await db.commit()
            print(content)
        else:
            print(f"Report generation failed or LLM disabled. Check ANTHROPIC_API_KEY.")


def main() -> None:
    setup_environment()

    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest", "retrain", "report-daily", "report-weekly", "api", "discord-bot"],
        default="paper",
        help="Operating mode",
    )
    parser.add_argument("--symbol", default="", help="Symbol for backtest mode")
    parser.add_argument("--start", default="", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end", default="", help="End date for backtest (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.mode == "api":
        run_api_server()

    elif args.mode == "backtest":
        from app.config.trading_config import get_trading_config
        cfg = get_trading_config()
        symbol = args.symbol or cfg.universe.symbols[0]
        asyncio.run(run_backtest(symbol, args.start, args.end))

    elif args.mode == "retrain":
        asyncio.run(run_retrain())

    elif args.mode in ("report-daily", "report-weekly"):
        report_type = "daily" if args.mode == "report-daily" else "weekly"
        asyncio.run(run_report(report_type))

    elif args.mode == "discord-bot":
        from app.discord_bot.bot import run_discord_bot
        asyncio.run(run_discord_bot())

    elif args.mode in ("paper", "live"):
        from app.trading_worker import TradingWorker
        worker = TradingWorker(mode=args.mode)
        asyncio.run(worker.run())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
