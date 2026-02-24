"""
Backtest engine: run a strategy over historical bars using framework types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Tuple

import pandas as pd

from lib.backtest.data_feed import DataFrameDataFeed
from lib.backtest.fees import alpaca_regulatory_fee
from lib.backtest.sim_broker import SimBroker
from lib.backtest.sim_clock import SimClock
from lib.framework.portfolio import Portfolio

if TYPE_CHECKING:
    from lib.framework.orders import Fill
    from lib.framework.strategy import Strategy


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    portfolio: Portfolio
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


def run(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 0.0,
    record_equity_curve: bool = True,
    slippage_bps: float = 0,
    fee_model: Callable[["Fill"], float] | None = None,
) -> BacktestResult:
    """
    Run a strategy over historical bars.

    Args:
        data: DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns (same shape as lib.stock).
        strategy: Strategy implementing next(current_time, market_snapshot, portfolio) -> list[Order].
        initial_cash: Starting cash for the portfolio.
        record_equity_curve: If True, record (time, equity) at each bar.
        slippage_bps: Basis points to worsen fill price (buy higher, sell lower). 0 = no slippage.
        fee_model: If None, uses Alpaca regulatory fees (TAF + CAT). Pass lambda fill: 0 for no fees.

    Returns:
        BacktestResult with final portfolio and optional equity_curve.
    """
    if fee_model is None:
        fee_model = alpaca_regulatory_fee
    feed = DataFrameDataFeed(data)
    if isinstance(data.index, pd.MultiIndex) and "timestamp" in data.index.names:
        timestamps = data.index.get_level_values("timestamp").unique()
    else:
        timestamps = data.index.tolist() if hasattr(data.index, "tolist") else []
    clock = SimClock(timestamps)
    broker = SimBroker(slippage_bps=slippage_bps, fee_model=fee_model)
    portfolio = Portfolio(cash=initial_cash)
    results = BacktestResult(portfolio=portfolio)

    while clock.advance():
        t = clock.current_time
        snapshot = feed.get_bars(t)
        orders = strategy.next(t, snapshot, portfolio)
        for order in orders:
            broker.submit(order)
        broker.set_current_bars(snapshot, t)
        for fill in broker.get_fills():
            portfolio.apply_fill(fill)
        if record_equity_curve:
            results.equity_curve.append((t, portfolio.equity(snapshot)))

    return results
