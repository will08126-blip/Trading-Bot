"""
Exchange adapter base interface.
All exchange implementations must implement this interface.
New exchanges can be added by creating a subclass without touching the rest of the system.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures returned by the adapter
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    is_closed: bool = True


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    mark_price: Optional[float]
    index_price: Optional[float]
    volume_24h: float
    timestamp: datetime


@dataclass
class OrderBook:
    symbol: str
    bids: List[tuple[float, float]]  # (price, size)
    asks: List[tuple[float, float]]
    timestamp: datetime


@dataclass
class ProductInfo:
    symbol: str
    base_currency: str
    quote_currency: str
    product_type: str              # spot | futures | perpetual
    is_tradable: bool
    min_order_size: float
    max_order_size: float
    size_increment: float
    price_increment: float
    max_leverage: float            # exchange max leverage (may vary by account)
    maker_fee_rate: float
    taker_fee_rate: float
    settlement_type: Optional[str]
    expiry_date: Optional[datetime]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountBalance:
    currency: str
    total: float
    available: float
    hold: float


@dataclass
class ExchangePosition:
    symbol: str
    side: str                       # long | short
    size: float
    entry_price: float
    mark_price: Optional[float]
    unrealized_pnl: float
    leverage: float
    margin_used: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeOrder:
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    order_type: str                 # market | limit | stop | stop_limit
    side: str                       # buy | sell
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    filled_size: float
    avg_fill_price: Optional[float]
    status: str                     # pending | open | filled | partial | cancelled | rejected
    created_at: datetime
    updated_at: Optional[datetime]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeFill:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    fill_price: float
    fill_size: float
    fee: float
    fee_currency: str
    fill_time: datetime
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capability flags – exchange may not support all features
# ---------------------------------------------------------------------------

@dataclass
class ExchangeCapabilities:
    supports_ws_candles: bool = False
    supports_ws_ticker: bool = False
    supports_ws_orderbook: bool = False
    supports_set_leverage: bool = False
    supports_amend_order: bool = False
    supports_stop_orders: bool = False
    supports_tp_sl_bundled: bool = False
    supports_mark_price: bool = False
    supports_funding_rate: bool = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseExchangeAdapter(abc.ABC):
    """
    Abstract base class for exchange adapters.
    Implement all abstract methods when adding a new exchange.
    Methods that are not supported should raise NotImplementedError.
    """

    capabilities: ExchangeCapabilities = ExchangeCapabilities()

    # ------------------------------------------------------------------ auth

    @abc.abstractmethod
    async def connect(self) -> None:
        """Authenticate and verify connectivity."""
        ...

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close connections."""
        ...

    # ---------------------------------------------------------- account data

    @abc.abstractmethod
    async def get_account_balances(self) -> List[AccountBalance]:
        """Return list of balances across currencies / margin accounts."""
        ...

    @abc.abstractmethod
    async def get_positions(self) -> List[ExchangePosition]:
        """Return all open positions."""
        ...

    @abc.abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Return all open orders, optionally filtered by symbol."""
        ...

    @abc.abstractmethod
    async def get_recent_fills(self, symbol: Optional[str] = None, limit: int = 50) -> List[ExchangeFill]:
        """Return recent fill history."""
        ...

    # -------------------------------------------------------- market / products

    @abc.abstractmethod
    async def get_products(self) -> List[ProductInfo]:
        """Return all tradable products."""
        ...

    @abc.abstractmethod
    async def get_product(self, symbol: str) -> ProductInfo:
        """Return metadata for a specific product."""
        ...

    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Return current ticker for a symbol."""
        ...

    @abc.abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Return order book snapshot."""
        ...

    @abc.abstractmethod
    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 300,
    ) -> List[Candle]:
        """
        Fetch historical OHLCV candles.
        timeframe examples: 1m, 5m, 15m, 1h, 4h, 1d
        """
        ...

    # ----------------------------------------------------------------- orders

    @abc.abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
        **kwargs: Any,
    ) -> ExchangeOrder:
        """Place an order. Returns the created order."""
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order. Returns True if cancelled."""
        ...

    @abc.abstractmethod
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> ExchangeOrder:
        """Fetch a single order by ID."""
        ...

    # ------------------------------------------------------------------ misc

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for a symbol. Returns True if successful."""
        raise NotImplementedError("This exchange does not support set_leverage")

    async def amend_order(
        self,
        order_id: str,
        symbol: str,
        price: Optional[float] = None,
        size: Optional[float] = None,
    ) -> ExchangeOrder:
        """Amend an existing order. Not supported on all exchanges."""
        raise NotImplementedError("This exchange does not support amend_order")

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate. Only relevant for perpetuals."""
        return None

    # ----------------------------------------------------------- websockets

    async def subscribe_candles(self, symbol: str, timeframe: str, callback) -> None:
        """Subscribe to live candle updates via websocket."""
        raise NotImplementedError("WS candles not supported")

    async def subscribe_ticker(self, symbol: str, callback) -> None:
        """Subscribe to live ticker updates via websocket."""
        raise NotImplementedError("WS ticker not supported")

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all websocket channels."""
        pass
