"""
Unit tests for Portfolio and Position.
"""

import unittest
from datetime import datetime, timezone

import pandas as pd

from lib.framework.orders import Fill, OrderSide
from lib.framework.portfolio import Portfolio, Position


def _ts(y: int, m: int, d: int, h: int = 14, mi: int = 30) -> datetime:
    return datetime(y, m, d, h, mi, 0, tzinfo=timezone.utc)


class TestPosition(unittest.TestCase):
    """Test Position dataclass."""

    def test_avg_price_zero_quantity(self):
        """avg_price is 0 when quantity is 0."""
        p = Position(symbol="AAPL", quantity=0, cost_basis=0.0)
        self.assertEqual(p.avg_price, 0.0)

    def test_avg_price_positive(self):
        """avg_price is cost_basis / quantity."""
        p = Position(symbol="AAPL", quantity=10, cost_basis=1500.0)
        self.assertEqual(p.avg_price, 150.0)


class TestPortfolio(unittest.TestCase):
    """Test Portfolio: apply_fill, equity, trade_history."""

    def test_cash_cannot_be_negative(self):
        """Portfolio rejects negative cash."""
        with self.assertRaises(ValueError) as ctx:
            Portfolio(cash=-100.0)
        self.assertIn("cash", str(ctx.exception))

    def test_position_returns_zero_position_for_unknown_symbol(self):
        """position(symbol) returns zero position when not held."""
        pf = Portfolio(cash=10_000.0)
        pos = pf.position("AAPL")
        self.assertEqual(pos.symbol, "AAPL")
        self.assertEqual(pos.quantity, 0)
        self.assertEqual(pos.cost_basis, 0.0)

    def test_apply_fill_buy_updates_cash_and_position(self):
        """Apply buy fill: cash decreases, position created."""
        pf = Portfolio(cash=10_000.0)
        fill = Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15))
        pf.apply_fill(fill)
        self.assertEqual(pf.cash, 9_000.0)
        pos = pf.position("AAPL")
        self.assertEqual(pos.quantity, 10)
        self.assertEqual(pos.cost_basis, 1000.0)
        self.assertEqual(len(pf.trade_history), 1)
        self.assertEqual(pf.trade_history[0], fill)

    def test_apply_fill_deducts_fee_from_cash(self):
        """Apply fill with fee: portfolio deducts fill.fee from cash."""
        pf = Portfolio(cash=10_000.0)
        fill = Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15), fee=0.50)
        pf.apply_fill(fill)
        self.assertEqual(pf.cash, 10_000.0 - 1000.0 - 0.50)

    def test_apply_fill_sell_updates_cash_and_position(self):
        """Apply sell fill: cash increases, position reduced."""
        pf = Portfolio(cash=10_000.0)
        pf.apply_fill(Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15)))
        pf.apply_fill(Fill(order_id="o2", symbol="AAPL", side="sell", price=110.0, qty=5, timestamp=_ts(2024, 1, 16)))
        self.assertEqual(pf.cash, 9_000.0 + 550.0)
        pos = pf.position("AAPL")
        self.assertEqual(pos.quantity, 5)
        self.assertEqual(len(pf.trade_history), 2)

    def test_apply_fill_sell_closes_position(self):
        """Sell full position removes symbol from positions."""
        pf = Portfolio(cash=10_000.0)
        pf.apply_fill(Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15)))
        pf.apply_fill(Fill(order_id="o2", symbol="AAPL", side="sell", price=110.0, qty=10, timestamp=_ts(2024, 1, 16)))
        self.assertNotIn("AAPL", pf.positions)
        self.assertEqual(pf.cash, 10_000.0 - 1000.0 + 1100.0)

    def test_apply_fill_sell_exceeds_position_raises(self):
        """Selling more than position raises ValueError."""
        pf = Portfolio(cash=10_000.0)
        pf.apply_fill(Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15)))
        with self.assertRaises(ValueError) as ctx:
            pf.apply_fill(Fill(order_id="o2", symbol="AAPL", side="sell", price=110.0, qty=11, timestamp=_ts(2024, 1, 16)))
        self.assertIn("Cannot sell", str(ctx.exception))

    def test_equity_no_prices_uses_cost_basis(self):
        """equity() with no prices uses cost basis (book value)."""
        pf = Portfolio(cash=1_000.0)
        pf.apply_fill(Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15)))
        # After buy: cash=0, position value at cost=1000
        self.assertEqual(pf.equity(), 1_000.0)

    def test_equity_with_snapshot(self):
        """equity(snapshot) uses close from snapshot for position value."""
        pf = Portfolio(cash=1_000.0)
        pf.apply_fill(Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15)))
        index = pd.MultiIndex.from_arrays(
            [["AAPL"], [pd.Timestamp("2024-01-15 14:30:00", tz="UTC")]],
            names=["symbol", "timestamp"],
        )
        snapshot = pd.DataFrame({"close": [120.0]}, index=index)
        # After buy: cash=0, position 10 @ 120 = 1200
        self.assertEqual(pf.equity(snapshot), 1_200.0)

    def test_trade_history_records_every_fill(self):
        """trade_history contains every fill in order."""
        pf = Portfolio(cash=10_000.0)
        f1 = Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=_ts(2024, 1, 15))
        f2 = Fill(order_id="o2", symbol="AAPL", side="sell", price=105.0, qty=3, timestamp=_ts(2024, 1, 16))
        pf.apply_fill(f1)
        pf.apply_fill(f2)
        self.assertEqual(pf.trade_history, [f1, f2])
