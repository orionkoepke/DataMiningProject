"""
Unit tests for Order and Fill.
"""

import unittest
from datetime import datetime, timezone

from lib.framework.orders import Fill, Order


class TestOrder(unittest.TestCase):
    """Test Order dataclass."""

    def test_market_order_minimal(self):
        """Market order with required fields only."""
        o = Order(symbol="AAPL", side="buy", qty=10)
        self.assertEqual(o.symbol, "AAPL")
        self.assertEqual(o.side, "buy")
        self.assertEqual(o.qty, 10)
        self.assertEqual(o.order_type, "market")
        self.assertIsNone(o.limit_price)
        self.assertIsNone(o.stop_price)
        self.assertIsNone(o.id)

    def test_order_with_id(self):
        """Order can have optional id."""
        o = Order(symbol="MSFT", side="sell", qty=5, id="ord-1")
        self.assertEqual(o.id, "ord-1")

    def test_limit_order_requires_limit_price(self):
        """Limit order must have limit_price."""
        with self.assertRaises(ValueError) as ctx:
            Order(symbol="AAPL", side="buy", qty=10, order_type="limit")
        self.assertIn("limit_price", str(ctx.exception))

    def test_limit_order_with_price(self):
        """Limit order with limit_price is valid."""
        o = Order(symbol="AAPL", side="buy", qty=10, order_type="limit", limit_price=150.0)
        self.assertEqual(o.limit_price, 150.0)

    def test_stop_order_requires_stop_price(self):
        """Stop order must have stop_price."""
        with self.assertRaises(ValueError) as ctx:
            Order(symbol="AAPL", side="sell", qty=10, order_type="stop")
        self.assertIn("stop_price", str(ctx.exception))

    def test_qty_must_be_positive(self):
        """Order qty must be positive."""
        with self.assertRaises(ValueError) as ctx:
            Order(symbol="AAPL", side="buy", qty=0)
        self.assertIn("qty", str(ctx.exception))
        with self.assertRaises(ValueError):
            Order(symbol="AAPL", side="buy", qty=-1)


class TestFill(unittest.TestCase):
    """Test Fill dataclass."""

    def test_fill_minimal(self):
        """Fill with required fields."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        f = Fill(order_id="o1", symbol="AAPL", side="buy", price=152.5, qty=10, timestamp=ts)
        self.assertEqual(f.order_id, "o1")
        self.assertEqual(f.symbol, "AAPL")
        self.assertEqual(f.side, "buy")
        self.assertEqual(f.price, 152.5)
        self.assertEqual(f.qty, 10)
        self.assertEqual(f.timestamp, ts)

    def test_fill_qty_must_be_positive(self):
        """Fill qty must be positive."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError) as ctx:
            Fill(order_id="o1", symbol="AAPL", side="buy", price=152.5, qty=0, timestamp=ts)
        self.assertIn("qty", str(ctx.exception))

    def test_fill_price_must_be_positive(self):
        """Fill price must be positive."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError) as ctx:
            Fill(order_id="o1", symbol="AAPL", side="buy", price=0, qty=10, timestamp=ts)
        self.assertIn("price", str(ctx.exception))

    def test_fill_fee_default_zero(self):
        """Fill fee defaults to 0."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        f = Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=ts)
        self.assertEqual(f.fee, 0.0)

    def test_fill_fee_cannot_be_negative(self):
        """Fill fee cannot be negative."""
        ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError) as ctx:
            Fill(order_id="o1", symbol="AAPL", side="buy", price=100.0, qty=10, timestamp=ts, fee=-0.01)
        self.assertIn("fee", str(ctx.exception))
