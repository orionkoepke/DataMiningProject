"""
Unit tests for conversion utilities.
"""

import unittest
from datetime import timedelta

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from lib.utils.conversions import timeframe_to_timedelta


class FakeTimeFrame:
    """Minimal timeframe-like object (used when Alpaca's TimeFrame restricts amount)."""

    def __init__(self, amount, unit):
        self.amount = amount
        self.unit = unit


class TestTimeframeToTimedelta(unittest.TestCase):
    """Test timeframe_to_timedelta."""

    def test_minute(self):
        """One minute."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(1, TimeFrameUnit.Minute)),
            timedelta(minutes=1),
        )

    def test_minutes_amount(self):
        """Multiple minutes."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(5, TimeFrameUnit.Minute)),
            timedelta(minutes=5),
        )

    def test_hour(self):
        """One hour."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(1, TimeFrameUnit.Hour)),
            timedelta(hours=1),
        )

    def test_hours_amount(self):
        """Multiple hours."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(2, TimeFrameUnit.Hour)),
            timedelta(hours=2),
        )

    def test_day(self):
        """One day."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(1, TimeFrameUnit.Day)),
            timedelta(days=1),
        )

    def test_days_amount(self):
        """Multiple days (Alpaca only allows amount=1 for Day; use fake to test conversion)."""
        self.assertEqual(
            timeframe_to_timedelta(FakeTimeFrame(3, TimeFrameUnit.Day)),
            timedelta(days=3),
        )

    def test_week(self):
        """One week."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(1, TimeFrameUnit.Week)),
            timedelta(weeks=1),
        )

    def test_weeks_amount(self):
        """Multiple weeks (Alpaca only allows amount=1 for Week; use fake to test conversion)."""
        self.assertEqual(
            timeframe_to_timedelta(FakeTimeFrame(2, TimeFrameUnit.Week)),
            timedelta(weeks=2),
        )

    def test_month(self):
        """One month -> 30 days."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(1, TimeFrameUnit.Month)),
            timedelta(days=30),
        )

    def test_months_amount(self):
        """Multiple months -> 30 * amount days."""
        self.assertEqual(
            timeframe_to_timedelta(TimeFrame(3, TimeFrameUnit.Month)),
            timedelta(days=90),
        )

    def test_unsupported_unit_raises(self):
        """Unsupported timeframe unit raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            timeframe_to_timedelta(FakeTimeFrame(1, "Unsupported"))
        self.assertIn("Unsupported timeframe unit", str(ctx.exception))
