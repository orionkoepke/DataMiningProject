"""
Unit tests for StockDataChecker.
"""

import unittest
import pandas as pd
import numpy as np

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_checks import StockDataChecker, InvalidDataException


class TestStockDataChecker(unittest.TestCase):
    """Test StockDataChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = StockDataChecker()

    def _create_valid_dataframe(self, symbols=None, timestamps=None, **column_overrides):
        """
        Helper to create a valid DataFrame with MultiIndex (symbol, timestamp) and required columns.

        Args:
            symbols: List of symbol strings (default: ['AAPL', 'MSFT', 'GOOGL'])
            timestamps: DatetimeIndex or list, UTC (default: 3 UTC datetimes)
            **column_overrides: Column names -> values to override (open, high, etc.)

        Returns:
            DataFrame with MultiIndex and columns open, high, low, close, volume, trade_count, vwap
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
        if timestamps is None:
            timestamps = pd.to_datetime(
                ['2024-01-01 14:30:00', '2024-01-02 14:30:00', '2024-01-03 14:30:00'],
                utc=True
            )
        timestamps = pd.to_datetime(timestamps, utc=True)
        n = min(len(symbols), len(timestamps))
        symbols, timestamps = symbols[:n], timestamps[:n]
        defaults = {
            'open': [150.0, 200.0, 250.0][:n],
            'high': [155.0, 205.0, 255.0][:n],
            'low': [148.0, 198.0, 248.0][:n],
            'close': [152.0, 202.0, 252.0][:n],
            'volume': [1000000, 2000000, 3000000][:n],
            'trade_count': [5000, 6000, 7000][:n],
            'vwap': [151.0, 201.0, 251.0][:n]
        }
        defaults.update(column_overrides)
        index = pd.MultiIndex.from_arrays([symbols, timestamps], names=['symbol', 'timestamp'])
        return pd.DataFrame(defaults, index=index)

    def test_assert_data_clean_with_clean_data(self):
        """Test assert_data_clean passes with clean data and check_data returns True."""
        data = self._create_valid_dataframe()
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_nan_in_one_column(self):
        """Test assert_data_clean raises exception and check_data returns False when one column has NaN."""
        data = self._create_valid_dataframe(open=[150.0, np.nan, 250.0])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('open', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_nan_in_multiple_columns(self):
        """Test assert_data_clean raises exception and check_data returns False when multiple columns have NaN."""
        data = self._create_valid_dataframe(
            open=[150.0, np.nan, 250.0],
            high=[155.0, 205.0, np.nan]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        exception_msg = str(context.exception)
        self.assertIn('open', exception_msg)
        self.assertIn('high', exception_msg)
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_empty_dataframe(self):
        """Test assert_data_clean handles empty DataFrame and check_data returns True."""
        data = pd.DataFrame()
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_all_nan_column(self):
        """Test assert_data_clean raises exception and check_data returns False when entire column is NaN."""
        data = self._create_valid_dataframe(high=[np.nan, np.nan, np.nan])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('high', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_timestamp_in_gmt(self):
        """Test assert_data_clean passes and check_data returns True when timestamp index is in GMT."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'MSFT'],
            timestamps=pd.to_datetime(['2024-06-01 12:00:00', '2024-06-02 12:00:00'], utc=True)
        )
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_timestamp_naive_fails(self):
        """Test assert_data_clean raises and check_data returns False when timestamp index is timezone-naive."""
        index = pd.MultiIndex.from_arrays(
            [
                ['AAPL', 'MSFT'],
                pd.to_datetime(['2024-06-01 12:00:00', '2024-06-02 12:00:00'])
            ],
            names=['symbol', 'timestamp']
        )
        data = pd.DataFrame(
            {
                'open': [150.0, 200.0],
                'high': [155.0, 205.0],
                'low': [148.0, 198.0],
                'close': [152.0, 202.0],
                'volume': [1000000, 2000000],
                'trade_count': [5000, 6000],
                'vwap': [151.0, 201.0]
            },
            index=index
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('not in GMT', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_timestamp_not_gmt_fails(self):
        """Test assert_data_clean raises and check_data returns False when timestamp index is in another timezone."""
        index = pd.MultiIndex.from_arrays(
            [
                ['AAPL', 'MSFT'],
                pd.to_datetime(['2024-06-01 12:00:00', '2024-06-02 12:00:00']).tz_localize('US/Eastern')
            ],
            names=['symbol', 'timestamp']
        )
        data = pd.DataFrame(
            {
                'open': [150.0, 200.0],
                'high': [155.0, 205.0],
                'low': [148.0, 198.0],
                'close': [152.0, 202.0],
                'volume': [1000000, 2000000],
                'trade_count': [5000, 6000],
                'vwap': [151.0, 201.0]
            },
            index=index
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('not in GMT', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_missing_timestamp_index_fails(self):
        """Test assert_data_clean raises and check_data returns False when MultiIndex or timestamp level is missing."""
        data = pd.DataFrame({
            'open': [150.0, 200.0],
            'high': [155.0, 205.0],
            'low': [148.0, 198.0],
            'close': [152.0, 202.0],
            'volume': [1000000, 2000000],
            'trade_count': [5000, 6000],
            'vwap': [151.0, 201.0]
        })
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('timestamp', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_missing_column_fails(self):
        """Test assert_data_clean raises and check_data returns False when a required column is missing."""
        data = self._create_valid_dataframe()
        data = data.drop(columns=['trade_count'])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('missing required columns', str(context.exception))
        self.assertIn('trade_count', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_multiple_missing_columns_fails(self):
        """Test assert_data_clean raises and check_data returns False when multiple columns are missing."""
        data = self._create_valid_dataframe()
        data = data.drop(columns=['trade_count', 'vwap'])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('missing required columns', str(context.exception))
        self.assertIn('trade_count', str(context.exception))
        self.assertIn('vwap', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_wrong_data_type_fails(self):
        """Test assert_data_clean raises and check_data returns False when a column has wrong data type."""
        data = self._create_valid_dataframe(open=['150.0', '200.0', '250.0'])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('Data type errors', str(context.exception))
        self.assertIn("'open' should be float", str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_wrong_int_type_fails(self):
        """Test assert_data_clean raises and check_data returns False when int column has wrong type."""
        data = self._create_valid_dataframe(volume=['1000000', '2000000', '3000000'])
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('Data type errors', str(context.exception))
        self.assertIn("'volume'", str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_wrong_timestamp_type_fails(self):
        """Test assert_data_clean raises and check_data returns False when timestamp index level is not datetime."""
        index = pd.MultiIndex.from_arrays(
            [
                ['AAPL', 'MSFT', 'GOOGL'],
                ['2024-01-01 14:30:00', '2024-01-02 14:30:00', '2024-01-03 14:30:00']
            ],
            names=['symbol', 'timestamp']
        )
        data = pd.DataFrame(
            {
                'open': [150.0, 200.0, 250.0],
                'high': [155.0, 205.0, 255.0],
                'low': [148.0, 198.0, 248.0],
                'close': [152.0, 202.0, 252.0],
                'volume': [1000000, 2000000, 3000000],
                'trade_count': [5000, 6000, 7000],
                'vwap': [151.0, 201.0, 251.0]
            },
            index=index
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('Data type errors', str(context.exception))
        self.assertIn("'timestamp' should be datetime", str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_multiple_type_errors_fails(self):
        """Test assert_data_clean raises and check_data returns False when multiple columns have wrong types."""
        data = self._create_valid_dataframe(
            open=['150.0', '200.0', '250.0'],
            volume=['1000000', '2000000', '3000000']
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('Data type errors', str(context.exception))
        self.assertIn("'open' should be float", str(context.exception))
        self.assertIn("'volume'", str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_valid_ohlc_passes(self):
        """Test assert_data_clean passes when OHLC satisfies low<=high, low<=open<=high, low<=close<=high."""
        data = self._create_valid_dataframe()
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_ohlc_boundary_passes(self):
        """Test assert_data_clean passes when open/close equal low or high (boundary)."""
        data = self._create_valid_dataframe(
            open=[148.0, 205.0, 250.0],
            high=[155.0, 205.0, 255.0],
            low=[148.0, 198.0, 248.0],
            close=[155.0, 198.0, 252.0]
        )
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_low_gt_high_fails(self):
        """Test assert_data_clean raises when low > high and check_data returns False."""
        data = self._create_valid_dataframe(
            high=[155.0, 205.0, 255.0],
            low=[160.0, 198.0, 248.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('OHLC validity', str(context.exception))
        self.assertIn('low must be <= high', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_open_lt_low_fails(self):
        """Test assert_data_clean raises when open < low and check_data returns False."""
        data = self._create_valid_dataframe(
            open=[140.0, 200.0, 250.0],
            low=[148.0, 198.0, 248.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('OHLC validity', str(context.exception))
        self.assertIn('open must be >= low', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_open_gt_high_fails(self):
        """Test assert_data_clean raises when open > high and check_data returns False."""
        data = self._create_valid_dataframe(
            open=[160.0, 200.0, 250.0],
            high=[155.0, 205.0, 255.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('OHLC validity', str(context.exception))
        self.assertIn('open must be <= high', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_close_lt_low_fails(self):
        """Test assert_data_clean raises when close < low and check_data returns False."""
        data = self._create_valid_dataframe(
            low=[148.0, 198.0, 248.0],
            close=[140.0, 202.0, 252.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('OHLC validity', str(context.exception))
        self.assertIn('close must be >= low', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_close_gt_high_fails(self):
        """Test assert_data_clean raises when close > high and check_data returns False."""
        data = self._create_valid_dataframe(
            high=[155.0, 205.0, 255.0],
            close=[160.0, 202.0, 252.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        self.assertIn('OHLC validity', str(context.exception))
        self.assertIn('close must be <= high', str(context.exception))
        self.assertFalse(self.checker.check_data(data))

    def test_assert_data_clean_with_timeframe_none_unchanged(self):
        """Test assert_data_clean without timeframe still passes (no timeframe check)."""
        data = self._create_valid_dataframe()
        self.checker.assert_data_clean(data)
        self.assertTrue(self.checker.check_data(data))

    def test_assert_data_clean_with_timeframe_day_complete_passes(self):
        """Test assert_data_clean passes when bars are 1 day apart and timeframe is Day."""
        data = self._create_valid_dataframe()
        self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertTrue(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_day_gap_missing_bars_fails(self):
        """Test assert_data_clean raises when gap is more than one day (missing bars)."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(['2024-01-01 14:30:00', '2024-01-04 14:30:00'], utc=True)
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertIn('Missing bar(s)', str(context.exception))
        self.assertIn('more than one', str(context.exception))
        self.assertFalse(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_day_gap_not_multiple_fails(self):
        """Test assert_data_clean raises when gap is not a multiple of 1 day."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(['2024-01-01 14:30:00', '2024-01-03 02:30:00'], utc=True)
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertIn('Missing bar(s)', str(context.exception))
        self.assertIn('more than one', str(context.exception))
        self.assertFalse(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_day_gap_too_small_fails(self):
        """Test assert_data_clean raises when consecutive bars are closer than 1 day."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 02:30:00'], utc=True)
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertIn('closer than', str(context.exception))
        self.assertFalse(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_empty_passes(self):
        """Test assert_data_clean with timeframe on empty DataFrame passes."""
        data = pd.DataFrame()
        self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertTrue(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_single_bar_passes(self):
        """Test assert_data_clean with timeframe and single bar (no consecutive pair) passes."""
        data = self._create_valid_dataframe(
            symbols=['AAPL'],
            timestamps=pd.to_datetime(['2024-01-01 14:30:00'], utc=True)
        )
        self.checker.assert_data_clean(data, timeframe=TimeFrame.Day)
        self.assertTrue(self.checker.check_data(data, timeframe=TimeFrame.Day))

    def test_assert_data_clean_with_timeframe_minute_complete_passes(self):
        """Test assert_data_clean passes when bars are 1 minute apart and timeframe is Minute."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-01 14:30:00', '2024-01-01 14:31:00', '2024-01-01 14:32:00'],
                utc=True
            )
        )
        self.checker.assert_data_clean(data, timeframe=TimeFrame.Minute)
        self.assertTrue(self.checker.check_data(data, timeframe=TimeFrame.Minute))

    def test_assert_data_clean_with_timeframe_minute_gap_fails(self):
        """Test assert_data_clean raises when minute bars have a non-integer-minute gap."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-01 14:30:00', '2024-01-01 14:31:30'],
                utc=True
            )
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data, timeframe=TimeFrame.Minute)
        self.assertIn('Missing bar(s)', str(context.exception))
        self.assertIn('more than one', str(context.exception))
        self.assertFalse(self.checker.check_data(data, timeframe=TimeFrame.Minute))

    def test_assert_complete_timeframe_session_only_consecutive_minutes_passes(self):
        """With contains_closed_market_data=False, consecutive RTH minute bars pass."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:31:00', '2024-01-02 14:32:00'],
                utc=True
            )
        )
        self.checker.assert_data_clean(
            data, timeframe=TimeFrame.Minute, contains_closed_market_data=False
        )
        self.assertTrue(
            self.checker.check_data(
                data, timeframe=TimeFrame.Minute, contains_closed_market_data=False
            )
        )

    def test_assert_complete_timeframe_session_only_missing_minute_fails(self):
        """With contains_closed_market_data=False, missing RTH minute bar still fails."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-02 14:32:00'],
                utc=True
            )
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(
                data, timeframe=TimeFrame.Minute, contains_closed_market_data=False
            )
        self.assertIn('Missing bar(s)', str(context.exception))
        self.assertFalse(
            self.checker.check_data(
                data, timeframe=TimeFrame.Minute, contains_closed_market_data=False
            )
        )

    def test_assert_complete_timeframe_session_only_two_trading_days_passes(self):
        """With contains_closed_market_data=False, two consecutive trading days pass for Day timeframe."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 14:30:00', '2024-01-03 14:30:00'],
                utc=True
            )
        )
        self.checker.assert_data_clean(
            data, timeframe=TimeFrame.Day, contains_closed_market_data=False
        )
        self.assertTrue(
            self.checker.check_data(
                data, timeframe=TimeFrame.Day, contains_closed_market_data=False
            )
        )

    def test_assert_complete_timeframe_session_only_overnight_gap_allowed_passes(self):
        """With contains_closed_market_data=False, overnight gap between sessions does not raise."""
        data = self._create_valid_dataframe(
            symbols=['AAPL', 'AAPL'],
            timestamps=pd.to_datetime(
                ['2024-01-02 21:00:00', '2024-01-03 14:30:00'],
                utc=True
            )
        )
        self.checker.assert_data_clean(
            data, timeframe=TimeFrame.Minute, contains_closed_market_data=False
        )
        self.assertTrue(
            self.checker.check_data(
                data,
                timeframe=TimeFrame.Minute,
                contains_closed_market_data=False,
            )
        )

    def test_assert_data_clean_with_multiple_ohlc_violations_fails(self):
        """Test assert_data_clean raises with multiple OHLC violations and check_data returns False."""
        data = self._create_valid_dataframe(
            open=[160.0, 190.0, 250.0],
            high=[155.0, 205.0, 255.0],
            low=[148.0, 198.0, 248.0],
            close=[140.0, 210.0, 252.0]
        )
        with self.assertRaises(InvalidDataException) as context:
            self.checker.assert_data_clean(data)
        msg = str(context.exception)
        self.assertIn('OHLC validity', msg)
        self.assertIn('open must be <= high', msg)
        self.assertIn('open must be >= low', msg)
        self.assertIn('close must be >= low', msg)
        self.assertIn('close must be <= high', msg)
        self.assertFalse(self.checker.check_data(data))


if __name__ == '__main__':
    unittest.main()
