"""
Exchange adapter factory.
Add new exchanges here without touching business logic.
"""
from app.exchange.base import BaseExchangeAdapter


def create_exchange_adapter(exchange_name: str = "coinbase") -> BaseExchangeAdapter:
    """
    Return the appropriate exchange adapter instance.
    Future: support "bybit", "binance", "okx", etc.
    """
    name = exchange_name.lower()
    if name in ("coinbase", "coinbase_advanced", "coinbase_derivatives"):
        from app.exchange.coinbase import CoinbaseAdapter
        return CoinbaseAdapter()
    elif name == "paper":
        from app.exchange.paper_exchange import PaperExchangeAdapter
        return PaperExchangeAdapter()
    else:
        raise ValueError(f"Unknown exchange: {exchange_name}. Supported: coinbase, paper")
