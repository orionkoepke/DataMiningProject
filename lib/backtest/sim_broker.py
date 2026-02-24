"""
Simulated broker: holds pending orders, fills at bar close using OHLC.
"""

import dataclasses
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import pandas as pd

from lib.backtest.fees import round_up_to_cent
from lib.framework.broker import Broker
from lib.framework.orders import Fill, Order, OrderSide, OrderType


class SimBroker(Broker):
    """
    Broker that fills market orders at the current bar's close price.
    Call set_current_bars(snapshot) before get_fills() so the broker has price and time.
    """

    def __init__(
        self,
        slippage_bps: float = 0,
        fee_model: Optional[Callable[[Fill], float]] = None,
    ) -> None:
        self._pending: List[Tuple[Order, str]] = []  # (order, assigned_order_id)
        self._next_id = 0
        self._slippage_bps = slippage_bps
        self._fee_model = fee_model
        self._current_bars: Optional[pd.DataFrame] = None
        self._current_time: Optional[datetime] = None

    def submit(self, order: Order) -> None:
        """Add order to pending; assign id if missing."""
        oid = order.id if order.id is not None else f"sim-{self._next_id}"
        self._next_id += 1
        self._pending.append((order, oid))

    def set_current_bars(self, snapshot: pd.DataFrame, current_time: Optional[datetime] = None) -> None:
        """
        Set the current bar snapshot and time for the next get_fills() call.
        Fills use the 'close' price and this time as fill timestamp.
        """
        self._current_bars = snapshot if not snapshot.empty else None
        if current_time is not None:
            self._current_time = current_time
        elif snapshot is not None and not snapshot.empty and hasattr(snapshot.index, "get_level_values"):
            try:
                ts = snapshot.index.get_level_values("timestamp")
                self._current_time = ts[0].to_pydatetime() if hasattr(ts[0], "to_pydatetime") else ts[0]
            except (KeyError, IndexError):
                self._current_time = None
        else:
            self._current_time = None

    def get_fills(self) -> List[Fill]:
        """
        Fill all pending market orders at current bar close; return fills and clear them.
        Call set_current_bars(snapshot, time) before this. Limit/stop orders are not filled (left pending).
        """
        if self._current_bars is None or self._current_bars.empty or self._current_time is None:
            return []
        fills: List[Fill] = []
        still_pending: List[Tuple[Order, str]] = []
        for order, oid in self._pending:
            if order.order_type != OrderType.MARKET:
                still_pending.append((order, oid))
                continue
            price = self._close_for_symbol(order.symbol)
            if price is None:
                still_pending.append((order, oid))
                continue
            if self._slippage_bps > 0:
                if order.side == OrderSide.BUY:
                    price = price * (1 + self._slippage_bps / 10_000)
                else:
                    price = price * (1 - self._slippage_bps / 10_000)
            fill = Fill(
                order_id=oid,
                symbol=order.symbol,
                side=order.side,
                price=price,
                qty=order.qty,
                timestamp=self._current_time,
                fee=0.0,
            )
            if self._fee_model is not None:
                fill = dataclasses.replace(fill, fee=round_up_to_cent(self._fee_model(fill)))
            fills.append(fill)
        self._pending = still_pending
        return fills

    def _close_for_symbol(self, symbol: str) -> Optional[float]:
        """Get close price for symbol from current bars; None if not found."""
        if self._current_bars is None or self._current_bars.empty:
            return None
        df = self._current_bars
        if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
            try:
                row = df.xs(symbol, level="symbol")
                if "close" in row.columns:
                    return float(row["close"].iloc[-1])
                return None
            except KeyError:
                return None
        if "close" in df.columns:
            return float(df["close"].iloc[-1])
        return None
