"""
Integration tests for StockDataFetcher.

These tests require valid Alpaca API credentials and will make actual API calls.
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from alpaca.common.exceptions import APIError

from lib.stock.data_fetcher import StockDataFetcher
from lib.stock.data_checks import StockDataChecker
from lib.stock.data_cleaner import StockDataCleaner
from alpaca.data.timeframe import TimeFrame


class TestStockDataFetcherInitialization(unittest.TestCase):
    """Test StockDataFetcher initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default API key path."""
        fetcher = StockDataFetcher()
        self.assertIsNotNone(fetcher)
        self.assertIsNotNone(fetcher.client)

    def test_init_with_custom_path(self):
        """Test initialization with custom API key path."""
        # tests/integration/test_stock_data.py -> project root is parent.parent.parent
        key_path = (
            Path(__file__).parent.parent.parent
            / "etc"
            / "private"
            / "alpaca_key.json"
        )
        fetcher = StockDataFetcher(key_file_path=key_path)
        self.assertIsNotNone(fetcher)
        self.assertIsNotNone(fetcher.client)

    def test_init_with_invalid_path(self):
        """Test initialization fails with invalid API key path."""
        with self.assertRaises(FileNotFoundError):
            StockDataFetcher(key_file_path="/nonexistent/path/key.json")


class TestGetHistoricalBars(unittest.TestCase):
    """Test get_historical_bars method."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher()
        self.test_symbol = "AAPL"
        end_date = datetime.now()
        self.start_date = end_date - timedelta(days=30)
        self.end_date = end_date

    def test_get_bars_single_symbol_string_dates(self):
        """Test fetching bars for a single symbol with string dates."""
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')

        data = self.fetcher.get_historical_bars(
            symbol=self.test_symbol,
            start_date=start_str,
            end_date=end_str,
            timeframe=TimeFrame.Day
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertGreater(len(data), 0)

    def test_get_bars_single_symbol_datetime_dates(self):
        """Test fetching bars for a single symbol with datetime objects."""
        data = self.fetcher.get_historical_bars(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=TimeFrame.Day
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_get_bars_multiple_symbols(self):
        """Test fetching bars for multiple symbols."""
        symbols = ["AAPL", "MSFT"]

        data = self.fetcher.get_historical_bars(
            symbol=symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=TimeFrame.Day
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        # Check that we have data for multiple symbols
        if 'symbol' in data.columns:
            unique_symbols = data['symbol'].unique()
            self.assertGreaterEqual(len(unique_symbols), 1)

    def test_get_bars_different_timeframes(self):
        """Test fetching bars with different timeframes."""
        # Test Day timeframe
        daily_data = self.fetcher.get_historical_bars(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=TimeFrame.Day
        )
        self.assertIsInstance(daily_data, pd.DataFrame)

        # Test Hour timeframe (if date range is recent enough)
        if (self.end_date - self.start_date).days <= 7:
            hourly_data = self.fetcher.get_historical_bars(
                symbol=self.test_symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=TimeFrame.Hour
            )
            self.assertIsInstance(hourly_data, pd.DataFrame)

    def test_get_bars_data_structure(self):
        """Test that returned data has expected structure."""
        data = self.fetcher.get_historical_bars(
            symbol=self.test_symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframe=TimeFrame.Day
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

        # Check for common OHLCV columns (exact names may vary by API response)
        self.assertGreater(len(data.columns), 0)

    def test_get_bars_date_range_validation(self):
        """Test that date range validation raises APIError for invalid date range."""
        end_date = datetime.now() - timedelta(days=10)
        start_date = datetime.now()

        with self.assertRaises(APIError):
            self.fetcher.get_historical_bars(
                symbol=self.test_symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=TimeFrame.Day
            )

    def test_get_bars_invalid_symbol(self):
        """Test fetching bars with an invalid symbol."""
        with self.assertRaises(Exception):
            self.fetcher.get_historical_bars(
                symbol="INVALID_SYMBOL_XYZ123",
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=TimeFrame.Day
            )


class TestStockDataFetcherWithChecker(unittest.TestCase):
    """Integration tests for StockDataFetcher with StockDataChecker."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher()
        self.checker = StockDataChecker()

    def test_fetch_and_validate_data(self):
        """Test fetching data and validating it with StockDataChecker."""
        data = self.fetcher.get_historical_bars(
            symbol="AAPL",
            start_date=datetime(2026, 1, 9),
            end_date=datetime(2026, 1, 11),
            timeframe=TimeFrame.Day
        )

        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

        try:
            self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        except Exception as e:
            self.fail(f"Data validation failed: {e}")

        self.assertTrue(self.checker.check_data(data))


class TestStockDataFetcherWithCleaner(unittest.TestCase):
    """Integration tests for StockDataFetcher with StockDataCleaner."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher()
        self.cleaner = StockDataCleaner()
        self.checker = StockDataChecker()
        end_date = datetime.now()
        self.start_date = end_date - timedelta(days=5)

    def test_fetch_then_remove_closed_market_rows(self):
        """Test fetching minute data and removing closed-market rows."""
        data = self.fetcher.get_historical_bars(
            symbol="AAPL",
            start_date=self.start_date,
            end_date=datetime.now(),
            timeframe=TimeFrame.Minute,
        )
        self.assertFalse(data.empty, "Fetched data should not be empty")
        self.assertIsInstance(data.index, pd.MultiIndex)
        self.assertIn("timestamp", data.index.names)
        self.assertIn("symbol", data.index.names)

        cleaned = self.cleaner.remove_closed_market_rows(data)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertIsNot(cleaned, data)
        self.assertLessEqual(len(cleaned), len(data))
        self.assertEqual(list(cleaned.columns), list(data.columns))
        if not cleaned.empty:
            self.assertIsInstance(cleaned.index, pd.MultiIndex)
            self.assertEqual(cleaned.index.names, data.index.names)

    def test_fetch_remove_closed_market_then_validate(self):
        """Test fetch -> remove closed market -> data passes checker (session-only)."""
        data = self.fetcher.get_historical_bars(
            symbol="AAPL",
            start_date=self.start_date,
            end_date=datetime.now(),
            timeframe=TimeFrame.Minute,
        )
        self.assertFalse(data.empty)
        cleaned = self.cleaner.remove_closed_market_rows(data)
        if cleaned.empty:
            self.skipTest("No RTH rows in fetched range")
        self.checker.assert_data_clean(
            cleaned,
            timeframe=TimeFrame.Minute,
            contains_closed_market_data=False,
        )
        self.assertTrue(
            self.checker.check_data(
                cleaned,
                timeframe=TimeFrame.Minute,
                contains_closed_market_data=False,
            )
        )

    def test_fetch_forward_propagate_then_validate(self):
        """Test fetch -> remove closed -> forward_propagate -> passes completeness check."""
        data = self.fetcher.get_historical_bars(
            symbol="AAPL",
            start_date=self.start_date,
            end_date=datetime.now(),
            timeframe=TimeFrame.Minute,
        )
        self.assertFalse(data.empty)
        cleaned = self.cleaner.remove_closed_market_rows(data)
        if cleaned.empty:
            self.skipTest("No RTH rows in fetched range")
        filled = self.cleaner.forward_propagate(
            cleaned,
            TimeFrame.Minute,
            only_when_market_open=True,
        )
        self.assertFalse(filled.empty)
        self.assertGreaterEqual(len(filled), len(cleaned))
        self.checker.assert_data_clean(
            filled,
            timeframe=TimeFrame.Minute,
            contains_closed_market_data=False,
        )

    def test_fetch_day_bars_remove_closed_market_rows(self):
        """Test remove_closed_market_rows on daily bars (one bar per day, all typically RTH)."""
        data = self.fetcher.get_historical_bars(
            symbol="AAPL",
            start_date=self.start_date,
            end_date=datetime.now(),
            timeframe=TimeFrame.Day,
        )
        self.assertFalse(data.empty)
        cleaned = self.cleaner.remove_closed_market_rows(data)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertLessEqual(len(cleaned), len(data))
        if not cleaned.empty:
            self.checker.assert_data_clean(
                cleaned,
                timeframe=TimeFrame.Day,
                contains_closed_market_data=False,
            )


if __name__ == '__main__':
    unittest.main()
