"""
Coinbase Advanced Trade API adapter.

TODO: Verify exact endpoint paths, authentication method (JWT vs HMAC-SHA256),
      and product IDs for futures/perpetuals against your account type.
      Coinbase's derivatives API (International Exchange / Nano BTC-PERP etc.)
      may require different product IDs than shown here.
      Use this as the integration skeleton and test against your live account.

Reference:
  - Advanced Trade REST: https://docs.cdp.coinbase.com/advanced-trade/reference/
  - Futures: may require Coinbase Derivatives account
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx
import structlog
import websockets

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

# Mapping from internal timeframe strings to Coinbase granularity values
# Reference: https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_getcandles
TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "2h": "TWO_HOUR",
    "4h": "FOUR_HOUR", # NOTE: Check if Coinbase supports 4h directly; may need to aggregate
    "6h": "SIX_HOUR",
    "1d": "ONE_DAY",
}


class CoinbaseAdapter(BaseExchangeAdapter):
    """
    Coinbase Advanced Trade adapter.

    Authentication uses HMAC-SHA256 signing of the request.
    See: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
    """

    capabilities = ExchangeCapabilities(
        supports_ws_candles=True,
        supports_ws_ticker=True,
        supports_ws_orderbook=False,
        supports_set_leverage=False,  # TODO: Verify leverage setting capability
        supports_amend_order=False,
        supports_stop_orders=True,    # TODO: Verify stop order support for futures
        supports_tp_sl_bundled=False,
        supports_mark_price=True,
        supports_funding_rate=False,  # TODO: Verify for perpetuals
    )

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.coinbase_api_key
        self._api_secret = settings.coinbase_api_secret
        self._rest_url = settings.coinbase_rest_url.rstrip("/")
        self._ws_url = settings.coinbase_ws_url
        self._client: Optional[httpx.AsyncClient] = None
        self._ws_conn = None
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._connected = False

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Generate HMAC-SHA256 authentication headers.
        TODO: Coinbase may have migrated to JWT-based auth for the Advanced Trade API.
              Verify current auth method at: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
        """
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY": self._api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated REST request."""
        if self._client is None:
            raise RuntimeError("CoinbaseAdapter not connected. Call connect() first.")

        body_str = json.dumps(json_body) if json_body else ""
        headers = self._sign_request(method, path, body_str)

        url = self._rest_url + path
        try:
            resp = await self._client.request(
                method,
                url,
                params=params,
                content=body_str.encode() if body_str else None,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("coinbase_http_error", status=e.response.status_code,
                         url=url, body=e.response.text[:500])
            raise
        except Exception as e:
            logger.error("coinbase_request_error", url=url, error=str(e))
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        # Verify connectivity
        try:
            await self.get_account_balances()
            self._connected = True
            logger.info("coinbase_adapter_connected")
        except Exception as e:
            logger.error("coinbase_connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        if self._ws_conn:
            await self._ws_conn.close()
        if self._client:
            await self._client.aclose()
        self._connected = False
        logger.info("coinbase_adapter_disconnected")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account_balances(self) -> List[AccountBalance]:
        """
        TODO: Verify correct endpoint for futures account balances.
              Advanced Trade: GET /api/v3/brokerage/accounts
        """
        data = await self._request("GET", "/api/v3/brokerage/accounts")
        balances = []
        for acct in data.get("accounts", []):
            available = float(acct.get("available_balance", {}).get("value", 0))
            hold = float(acct.get("hold", {}).get("value", 0))
            balances.append(AccountBalance(
                currency=acct.get("currency", ""),
                total=available + hold,
                available=available,
                hold=hold,
            ))
        return balances

    async def get_positions(self) -> List[ExchangePosition]:
        """
        TODO: Verify endpoint for futures positions.
              Coinbase Derivatives may use: GET /api/v3/brokerage/positions
        """
        try:
            data = await self._request("GET", "/api/v3/brokerage/positions")
            positions = []
            for p in data.get("futures_positions", []):
                positions.append(ExchangePosition(
                    symbol=p.get("product_id", ""),
                    side="long" if float(p.get("number_of_contracts", 0)) > 0 else "short",
                    size=abs(float(p.get("number_of_contracts", 0))),
                    entry_price=float(p.get("avg_entry_price", 0)),
                    mark_price=float(p.get("mark_price", 0)) if p.get("mark_price") else None,
                    unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                    leverage=float(p.get("leverage", 1)),
                    margin_used=float(p.get("margin_used", 0)),
                    raw=p,
                ))
            return positions
        except Exception as e:
            logger.warning("get_positions_failed", error=str(e))
            return []

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """GET /api/v3/brokerage/orders/historical/batch"""
        params: Dict[str, Any] = {"order_status": "OPEN", "limit": 100}
        if symbol:
            params["product_id"] = symbol
        data = await self._request("GET", "/api/v3/brokerage/orders/historical/batch", params=params)
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def get_recent_fills(self, symbol: Optional[str] = None, limit: int = 50) -> List[ExchangeFill]:
        """GET /api/v3/brokerage/orders/historical/fills"""
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["product_id"] = symbol
        data = await self._request("GET", "/api/v3/brokerage/orders/historical/fills", params=params)
        fills = []
        for f in data.get("fills", []):
            fills.append(ExchangeFill(
                fill_id=f.get("entry_id", ""),
                order_id=f.get("order_id", ""),
                symbol=f.get("product_id", ""),
                side=f.get("side", "").lower(),
                fill_price=float(f.get("price", 0)),
                fill_size=float(f.get("size", 0)),
                fee=float(f.get("commission", 0)),
                fee_currency="USD",
                fill_time=_parse_ts(f.get("trade_time", "")),
                raw=f,
            ))
        return fills

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_products(self) -> List[ProductInfo]:
        """GET /api/v3/brokerage/products"""
        data = await self._request("GET", "/api/v3/brokerage/products")
        return [self._parse_product(p) for p in data.get("products", [])]

    async def get_product(self, symbol: str) -> ProductInfo:
        data = await self._request("GET", f"/api/v3/brokerage/products/{symbol}")
        return self._parse_product(data)

    async def get_ticker(self, symbol: str) -> Ticker:
        """GET /api/v3/brokerage/products/{product_id}/ticker"""
        data = await self._request("GET", f"/api/v3/brokerage/products/{symbol}/ticker")
        trades = data.get("trades", [{}])
        last_trade = trades[0] if trades else {}
        best_bid = data.get("best_bid", "0")
        best_ask = data.get("best_ask", "0")
        return Ticker(
            symbol=symbol,
            bid=float(best_bid),
            ask=float(best_ask),
            last=float(last_trade.get("price", best_bid)),
            mark_price=None,  # TODO: fetch mark price from futures endpoint
            index_price=None,
            volume_24h=0.0,   # TODO: parse volume from product data
            timestamp=datetime.now(timezone.utc),
        )

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """GET /api/v3/brokerage/product_book"""
        data = await self._request("GET", "/api/v3/brokerage/product_book",
                                   params={"product_id": symbol, "limit": depth})
        pricebook = data.get("pricebook", {})
        bids = [(float(b["price"]), float(b["size"])) for b in pricebook.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in pricebook.get("asks", [])]
        return OrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=datetime.now(timezone.utc))

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 300,
    ) -> List[Candle]:
        """
        GET /api/v3/brokerage/products/{product_id}/candles
        Max 350 candles per request; paginate if needed.
        TODO: Verify 4h granularity support; may need to aggregate from 1h.
        """
        gran = TIMEFRAME_MAP.get(timeframe)
        if not gran:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: Dict[str, Any] = {"granularity": gran, "limit": min(limit, 350)}
        if start:
            params["start"] = str(int(start.timestamp()))
        if end:
            params["end"] = str(int(end.timestamp()))

        data = await self._request("GET", f"/api/v3/brokerage/products/{symbol}/candles", params=params)
        candles = []
        tf_seconds = _timeframe_to_seconds(timeframe)
        for c in data.get("candles", []):
            open_ts = int(c.get("start", 0))
            open_dt = datetime.fromtimestamp(open_ts, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(open_ts + tf_seconds, tz=timezone.utc)
            candles.append(Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=open_dt,
                close_time=close_dt,
                open=float(c.get("open", 0)),
                high=float(c.get("high", 0)),
                low=float(c.get("low", 0)),
                close=float(c.get("close", 0)),
                volume=float(c.get("volume", 0)),
                is_closed=True,
            ))
        # Coinbase returns newest first; reverse for chronological order
        candles.sort(key=lambda x: x.open_time)
        return candles

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

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
        """
        POST /api/v3/brokerage/orders
        TODO: Verify order type payloads for futures.
              Coinbase uses nested config objects for order configuration.
        """
        client_oid = client_order_id or str(uuid.uuid4())
        body: Dict[str, Any] = {
            "client_order_id": client_oid,
            "product_id": symbol,
            "side": side.upper(),
        }

        if order_type == "market":
            body["order_configuration"] = {
                "market_market_ioc": {"base_size": str(size)}
            }
        elif order_type == "limit":
            if price is None:
                raise ValueError("price required for limit order")
            body["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(price),
                    "post_only": False,
                }
            }
        elif order_type in ("stop", "stop_limit"):
            # TODO: Verify stop order config for Coinbase futures
            if stop_price is None:
                raise ValueError("stop_price required for stop order")
            body["order_configuration"] = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(size),
                    "limit_price": str(stop_price),
                    "stop_price": str(stop_price),
                    "stop_direction": "STOP_DIRECTION_STOP_DOWN" if side.lower() == "sell" else "STOP_DIRECTION_STOP_UP",
                }
            }
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        data = await self._request("POST", "/api/v3/brokerage/orders", json_body=body)
        success = data.get("success", False)
        if not success:
            error_resp = data.get("error_response", {})
            raise RuntimeError(f"Order placement failed: {error_resp}")

        order_data = data.get("order", data.get("success_response", {}))
        # Immediately fetch to get full order details
        order_id = order_data.get("order_id", client_oid)
        return await self.get_order(order_id, symbol)

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """POST /api/v3/brokerage/orders/batch_cancel"""
        data = await self._request("POST", "/api/v3/brokerage/orders/batch_cancel",
                                   json_body={"order_ids": [order_id]})
        results = data.get("results", [])
        return any(r.get("order_id") == order_id and not r.get("failure_reason") for r in results)

    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> ExchangeOrder:
        """GET /api/v3/brokerage/orders/historical/{order_id}"""
        data = await self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        return self._parse_order(data.get("order", data))

    # ------------------------------------------------------------------
    # Websocket streaming
    # ------------------------------------------------------------------

    async def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable) -> None:
        """
        Subscribe to candle updates via Coinbase Advanced Trade WebSocket.
        TODO: Confirm candles channel availability and message format.
              Reference: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-channels
        """
        channel = "candles"
        key = f"{channel}:{symbol}:{timeframe}"
        if key not in self._ws_callbacks:
            self._ws_callbacks[key] = []
        self._ws_callbacks[key].append(callback)
        asyncio.create_task(self._ws_listener(symbol, channel))

    async def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """Subscribe to ticker/heartbeat channel."""
        channel = "ticker"
        key = f"{channel}:{symbol}"
        if key not in self._ws_callbacks:
            self._ws_callbacks[key] = []
        self._ws_callbacks[key].append(callback)
        asyncio.create_task(self._ws_listener(symbol, channel))

    async def _ws_listener(self, symbol: str, channel: str) -> None:
        """
        WebSocket listener for a single channel.
        TODO: Implement proper auth for WS (JWT or HMAC).
              Reference: https://docs.cdp.coinbase.com/advanced-trade/docs/ws-auth
        """
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [symbol],
            "channel": channel,
            "api_key": self._api_key,
            # TODO: Add proper WS auth signature
        }
        backoff = 1
        while True:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    backoff = 1
                    logger.info("ws_subscribed", symbol=symbol, channel=channel)
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        await self._dispatch_ws_message(msg, symbol, channel)
            except Exception as e:
                logger.error("ws_error", channel=channel, symbol=symbol, error=str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _dispatch_ws_message(self, msg: Dict, symbol: str, channel: str) -> None:
        """Route WebSocket messages to registered callbacks."""
        msg_type = msg.get("channel", "")
        events = msg.get("events", [])
        for event in events:
            # Candle events
            if channel == "candles":
                for candle_data in event.get("candles", []):
                    key = f"candles:{symbol}:{candle_data.get('granularity', '')}"
                    for cb in self._ws_callbacks.get(key, []):
                        try:
                            await cb(candle_data)
                        except Exception as e:
                            logger.error("ws_callback_error", error=str(e))

    async def unsubscribe_all(self) -> None:
        self._ws_callbacks.clear()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_order(self, data: Dict) -> ExchangeOrder:
        status_map = {
            "OPEN": "open", "FILLED": "filled", "CANCELLED": "cancelled",
            "EXPIRED": "expired", "PENDING": "pending", "UNKNOWN_ORDER_STATUS": "pending",
        }
        return ExchangeOrder(
            order_id=data.get("order_id", ""),
            client_order_id=data.get("client_order_id"),
            symbol=data.get("product_id", ""),
            order_type=data.get("order_type", "UNKNOWN").lower(),
            side=data.get("side", "").lower(),
            size=float(data.get("base_size", 0)),
            price=float(data["order_configuration"].get("limit_limit_gtc", {}).get("limit_price", 0))
                  if "order_configuration" in data else None,
            stop_price=None,
            filled_size=float(data.get("filled_size", 0)),
            avg_fill_price=float(data.get("average_filled_price", 0)) or None,
            status=status_map.get(data.get("status", ""), "unknown"),
            created_at=_parse_ts(data.get("created_time", "")),
            updated_at=_parse_ts(data.get("last_fill_time", "")) if data.get("last_fill_time") else None,
            raw=data,
        )

    def _parse_product(self, data: Dict) -> ProductInfo:
        return ProductInfo(
            symbol=data.get("product_id", ""),
            base_currency=data.get("base_currency_id", ""),
            quote_currency=data.get("quote_currency_id", ""),
            product_type=data.get("product_type", "SPOT").lower(),
            is_tradable=data.get("is_disabled", False) is False,
            min_order_size=float(data.get("base_min_size", 0.001)),
            max_order_size=float(data.get("base_max_size", 999999)),
            size_increment=float(data.get("base_increment", 0.00001)),
            price_increment=float(data.get("quote_increment", 0.01)),
            max_leverage=float(data.get("future_product_details", {}).get("contract_size", 1)),
            # TODO: Verify leverage field from futures product metadata
            maker_fee_rate=0.0,  # TODO: Fetch from fee schedule
            taker_fee_rate=0.0005,  # Default; TODO: fetch from /api/v3/brokerage/transaction_summary
            settlement_type=data.get("future_product_details", {}).get("contract_display_name"),
            expiry_date=None,
            raw=data,
        )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        # Handle trailing 'Z'
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def _timeframe_to_seconds(timeframe: str) -> int:
    mapping = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400,
    }
    return mapping.get(timeframe, 60)
