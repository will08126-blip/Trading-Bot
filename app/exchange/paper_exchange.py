"""
Paper trading exchange adapter.
Simulates exchange behavior with configurable slippage and fees.
Uses the market data layer for pricing but doesn't touch real orders.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from app.config.settings import get_settings
from app.exchange.base import (
    AccountBalance,
    BaseExchangeAdapter,
    Candle,
    ExchangeCapabilities,
    ExchangeFill,
    ExchangeOrder,
    ExchangePosition,
    OrderBook,
    ProductInfo,
    Ticker,
)

logger = structlog.get_logger(__name__)


class PaperExchangeAdapter(BaseExchangeAdapter):
    """
    Simulated exchange for paper trading.
    Does not connect to any external service for order execution.
    Market data still comes through the real exchange adapter or cached candles.
    """

    capabilities = ExchangeCapabilities(
        supports_ws_candles=False,
        supports_ws_ticker=False,
        supports_ws_orderbook=False,
        supports_set_leverage=True,
        supports_amend_order=False,
        supports_stop_orders=True,
        supports_tp_sl_bundled=False,
        supports_mark_price=False,
        supports_funding_rate=False,
    )

    def __init__(self) -> None:
        settings = get_settings()
        self._balance = settings.paper_initial_balance
        self._positions: Dict[str, ExchangePosition] = {}
        self._orders: Dict[str, ExchangeOrder] = {}
        self._fills: List[ExchangeFill] = []
        self._leverages: Dict[str, float] = {}
        # For real market data we delegate to the coinbase adapter
        self._market_adapter: Optional[BaseExchangeAdapter] = None

    async def connect(self) -> None:
        logger.info("paper_exchange_connected", initial_balance=self._balance)

    async def disconnect(self) -> None:
        logger.info("paper_exchange_disconnected")

    async def get_account_balances(self) -> List[AccountBalance]:
        total = self._balance
        in_use = sum(p.margin_used for p in self._positions.values())
        return [AccountBalance(
            currency="USD",
            total=total,
            available=max(0.0, total - in_use),
            hold=in_use,
        )]

    async def get_positions(self) -> List[ExchangePosition]:
        return list(self._positions.values())

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        orders = [o for o in self._orders.values() if o.status in ("open", "pending")]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_recent_fills(self, symbol: Optional[str] = None, limit: int = 50) -> List[ExchangeFill]:
        fills = list(reversed(self._fills))
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        return fills[:limit]

    async def get_products(self) -> List[ProductInfo]:
        # Delegate to real adapter if available
        if self._market_adapter:
            return await self._market_adapter.get_products()
        return []

    async def get_product(self, symbol: str) -> ProductInfo:
        if self._market_adapter:
            return await self._market_adapter.get_product(symbol)
        return ProductInfo(
            symbol=symbol, base_currency="BTC", quote_currency="USD",
            product_type="perpetual", is_tradable=True,
            min_order_size=0.001, max_order_size=999,
            size_increment=0.00001, price_increment=0.01,
            max_leverage=10.0, maker_fee_rate=0.0002, taker_fee_rate=0.0005,
            settlement_type=None, expiry_date=None,
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        if self._market_adapter:
            return await self._market_adapter.get_ticker(symbol)
        raise RuntimeError("No market adapter configured for paper exchange")

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        if self._market_adapter:
            return await self._market_adapter.get_order_book(symbol, depth)
        return OrderBook(symbol=symbol, bids=[], asks=[], timestamp=datetime.now(timezone.utc))

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 300,
    ) -> List[Candle]:
        if self._market_adapter:
            return await self._market_adapter.fetch_candles(symbol, timeframe, start, end, limit)
        return []

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
        current_price: float = 0.0,
        slippage_pct: float = 0.0005,
        fee_rate: float = 0.0005,
        **kwargs: Any,
    ) -> ExchangeOrder:
        """
        Simulate order placement with slippage and fees.
        For market orders, fill immediately at current_price + slippage.
        For limit orders, store as open and fill when price is touched.
        """
        order_id = client_order_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Determine fill price
        if order_type == "market":
            slip = slippage_pct if side.lower() == "buy" else -slippage_pct
            fill_price = current_price * (1 + slip) if current_price > 0 else (price or 0.0)
            fill_price = round(fill_price, 2)
            fee = fill_price * size * fee_rate
            fill = ExchangeFill(
                fill_id=str(uuid.uuid4()),
                order_id=order_id,
                symbol=symbol,
                side=side.lower(),
                fill_price=fill_price,
                fill_size=size,
                fee=fee,
                fee_currency="USD",
                fill_time=now,
            )
            self._fills.append(fill)
            order = ExchangeOrder(
                order_id=order_id,
                client_order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side.lower(),
                size=size,
                price=fill_price,
                stop_price=stop_price,
                filled_size=size,
                avg_fill_price=fill_price,
                status="filled",
                created_at=now,
                updated_at=now,
            )
        else:
            # Limit / stop - store as open
            order = ExchangeOrder(
                order_id=order_id,
                client_order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side.lower(),
                size=size,
                price=price,
                stop_price=stop_price,
                filled_size=0.0,
                avg_fill_price=None,
                status="open",
                created_at=now,
                updated_at=now,
            )

        self._orders[order_id] = order
        logger.debug("paper_order_placed", order_id=order_id, symbol=symbol,
                     order_type=order_type, side=side, size=size, status=order.status)
        return order

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False

    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> ExchangeOrder:
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found in paper exchange")
        return self._orders[order_id]

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        self._leverages[symbol] = leverage
        logger.debug("paper_leverage_set", symbol=symbol, leverage=leverage)
        return True

    def set_market_adapter(self, adapter: BaseExchangeAdapter) -> None:
        """Provide a real adapter for market data queries."""
        self._market_adapter = adapter
