"""
Standalone backtest script.
Usage: python scripts/backtest.py --symbol BTC-PERP --start 2023-01-01
"""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.monitoring.logger import setup_logging
from app.config.settings import get_settings

settings = get_settings()
setup_logging(settings.log_level, settings.log_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC-PERP")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="")
    args = parser.parse_args()

    from app.main import run_backtest
    asyncio.run(run_backtest(args.symbol, args.start, args.end))
