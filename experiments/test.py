"""
Test script to fetch trade data for VOO using the stock_data library.
"""

from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame

from lib.stock.data_fetcher import StockDataFetcher
from lib.stock.data_checks import StockDataChecker
from lib.stock.data_cleaner import StockDataCleaner


def main():
    """Fetch the last year's worth of trade data for VOO."""
    # Initialize the stock data fetcher
    print("Initializing StockDataFetcher...")
    fetcher = StockDataFetcher()
    checker = StockDataChecker()
    cleaner = StockDataCleaner()
    
    # Calculate date range (last year to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    # Fetch trade data
    try:
        trade_data = fetcher.get_historical_bars(
            symbol="PMEC",
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Minute
        )

        print("Removing closed market rows...")
        trade_data = cleaner.remove_closed_market_rows(trade_data)

        print("Forward-propagating data...")
        trade_data = cleaner.forward_propagate(trade_data, TimeFrame.Minute, only_when_market_open=True)

        import os

        output_dir = "./etc/data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "trade_data.csv")

        print(f"Saving trade data to {output_path} ...")
        trade_data.to_csv(output_path)

        print("Checking data...")
        checker.assert_data_clean(trade_data, timeframe=TimeFrame.Minute, contains_closed_market_data=False)


            
    except Exception as e:
        print(f"Error fetching trade data: {e}")
        raise


if __name__ == "__main__":
    main()
