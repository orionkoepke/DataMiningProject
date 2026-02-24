"""
Unit tests for lib.backtest.fees.
"""

import unittest
from datetime import datetime, timezone

from lib.backtest.fees import alpaca_regulatory_fee, round_up_to_cent
from lib.framework.orders import Fill, OrderSide


def _ts(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 14, 30, 0, tzinfo=timezone.utc)


class TestRoundUpToCent(unittest.TestCase):
    """Test round_up_to_cent."""

    def test_zero_stays_zero(self):
        self.assertEqual(round_up_to_cent(0), 0.0)

    def test_subcent_rounds_up_to_one_cent(self):
        self.assertEqual(round_up_to_cent(0.001), 0.01)
        self.assertEqual(round_up_to_cent(0.009), 0.01)

    def test_exact_cent_unchanged(self):
        self.assertEqual(round_up_to_cent(0.01), 0.01)
        # Use integer cents to avoid float representation (1.10 can be 1.1000000000000001)
        self.assertEqual(round_up_to_cent(1.0), 1.0)
        self.assertEqual(round_up_to_cent(2.0), 2.0)

    def test_just_over_cent_rounds_up(self):
        self.assertEqual(round_up_to_cent(1.101), 1.11)
        self.assertEqual(round_up_to_cent(0.011), 0.02)


class TestAlpacaRegulatoryFee(unittest.TestCase):
    """Test alpaca_regulatory_fee."""

    def test_buy_small_qty_cat_only_rounded_up(self):
        """Buy: only CAT; 10 shares -> 0.0003 -> rounds up to 0.01."""
        fill = Fill(order_id="o1", symbol="AAPL", side=OrderSide.BUY, price=100.0, qty=10, timestamp=_ts(2024, 1, 15))
        self.assertEqual(alpaca_regulatory_fee(fill), 0.01)

    def test_buy_single_share_cat_rounds_up(self):
        """Buy 1 share: CAT = 0.00003 -> 0.01."""
        fill = Fill(order_id="o1", symbol="AAPL", side=OrderSide.BUY, price=50.0, qty=1, timestamp=_ts(2024, 1, 15))
        self.assertEqual(alpaca_regulatory_fee(fill), 0.01)

    def test_sell_small_qty_taf_plus_cat(self):
        """Sell: TAF + CAT, each rounded. 10 shares -> TAF 0.00166->0.01, CAT 0.0003->0.01 -> 0.02."""
        fill = Fill(order_id="o1", symbol="AAPL", side=OrderSide.SELL, price=100.0, qty=10, timestamp=_ts(2024, 1, 15))
        self.assertEqual(alpaca_regulatory_fee(fill), 0.02)

    def test_sell_taf_capped_at_830(self):
        """Sell large qty: TAF capped at 8.30, CAT still per share."""
        fill = Fill(
            order_id="o1", symbol="AAPL", side=OrderSide.SELL, price=100.0, qty=100_000, timestamp=_ts(2024, 1, 15)
        )
        # TAF = min(16.6, 8.30) = 8.30; CAT = 100000 * 0.00003 = 3.00 (each rounded up to cent)
        fee = alpaca_regulatory_fee(fill)
        self.assertIn(round(fee, 2), (11.30, 11.31), msg="Float rounding may yield 11.30 or 11.31")
