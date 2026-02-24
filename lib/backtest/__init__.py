"""
Backtest package: run strategies over historical bars using lib.framework.
"""

from lib.backtest.data_feed import DataFrameDataFeed
from lib.backtest.engine import BacktestResult, run
from lib.backtest.fees import alpaca_regulatory_fee, round_up_to_cent
from lib.backtest.sim_broker import SimBroker
from lib.backtest.sim_clock import SimClock

__all__ = [
    "BacktestResult",
    "DataFrameDataFeed",
    "SimBroker",
    "SimClock",
    "alpaca_regulatory_fee",
    "round_up_to_cent",
    "run",
]
