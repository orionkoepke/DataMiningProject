"""
Order and Fill types for the trading framework.
Broker-agnostic; no Alpaca or other broker types in core.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    """Side of an order or fill."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class Order:
    """A single order to be sent to a broker."""

    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price required for limit order")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("stop_price required for stop/stop_limit order")


@dataclass(frozen=True)
class Fill:
    """A fill (execution) of an order. Optional fee is deducted from cash when applied to a portfolio."""

    order_id: str
    symbol: str
    side: OrderSide
    price: float
    qty: int
    timestamp: datetime
    fee: float = 0.0

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.fee < 0:
            raise ValueError("fee cannot be negative")
