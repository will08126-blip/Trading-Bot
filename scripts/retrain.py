"""
Standalone ML retraining script.
Usage: python scripts/retrain.py
"""
import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    from app.main import run_retrain
    asyncio.run(run_retrain())
