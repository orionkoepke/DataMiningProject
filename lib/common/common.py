"""Common dataframe feature-engineering helpers used by experiments."""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _index_position(data: pd.DataFrame, label) -> int:
    """Return integer position in data for the given index label."""
    pos = data.index.get_loc(label)
    if isinstance(pos, slice):
        return pos.start
    if isinstance(pos, np.ndarray):
        return int(pos.flat[0])
    return int(pos)


def _trade_date_series(data: pd.DataFrame) -> pd.Series:
    """Return trade date (America/New_York date) per row for grouping by day."""
    ts = data.index.get_level_values("timestamp")
    series = pd.Series(ts, index=data.index)
    if hasattr(ts, "tz") and ts.tz is not None:
        series = series.dt.tz_convert("America/New_York")
    return series.dt.date


def _targets_for_day_vectorized(
    highs: np.ndarray,
    lows: np.ndarray,
    n: int,
    take_profit: float,
    stop_loss: float,
    *,
    max_bars_after_entry: int | None = None,
) -> np.ndarray:
    """Fill target for each bar in the day (0 for last 3 bars). Returns array of length n."""
    out = np.zeros(n, dtype=np.int64)
    num_entries = max(0, n - 3)
    if num_entries == 0:
        return out
    entries = highs[1 : num_entries + 1]
    sl_prices = entries * (1 - stop_loss)
    tp_prices = entries * (1 + take_profit)
    # Mask invalid entries (<= 0)
    valid = entries > 0
    for i in range(num_entries):
        if not valid[i]:
            continue
        sl_price = sl_prices[i]
        tp_price = tp_prices[i]
        end = n
        if max_bars_after_entry is not None:
            end = min(n, i + 2 + max_bars_after_entry)
        lows_seg = lows[i + 2 : end]
        highs_seg = highs[i + 2 : end]
        hit_sl = lows_seg <= sl_price
        hit_tp = highs_seg >= tp_price
        first_sl = int(np.argmax(hit_sl)) if hit_sl.any() else len(lows_seg)
        first_tp = int(np.argmax(hit_tp)) if hit_tp.any() else len(highs_seg)
        # Same bar both hit -> 0; else 1 iff TP before SL
        if first_sl == first_tp and first_sl < len(lows_seg):
            out[i] = 0
        elif first_tp < first_sl:
            out[i] = 1
        else:
            out[i] = 0
    return out


def _iter_daily_ohlc(data: pd.DataFrame, trade_date: pd.Series, symbol: pd.Index):
    """Yield (locs, highs, lows, n) for each trading day in data."""
    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        locs = group.index
        highs = group["high"].values
        lows = group["low"].values
        yield locs, highs, lows, len(group)


def create_target_column(
    data: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    *,
    column_name: str = "target",
    max_bars_after_entry: int | None = None,
) -> pd.DataFrame:
    """
    Add a binary target column: 1 if a buy at the next bar's high would hit take_profit
    before stop_loss and before end of trading day, and (when set) within the next
    ``max_bars_after_entry`` bars after the entry bar, else 0.

    Modifies data in place and returns it. Entry is assumed to be the next bar's high
    (we buy at that price). Exit is checked from the bar after that onward. The last 3
    bars of each day always get target 0 (not enough time for a round trip).

    Assumes data has MultiIndex (symbol, timestamp) and columns open, high, low, close.
    take_profit and stop_loss are decimals (e.g. 0.01 = 1%).
    When both TP and SL are reachable in the same bar, we treat as stop_loss (conservative).

    Args:
        column_name: Name of the column to add (default "target").
        max_bars_after_entry: If set, only bars this many positions after the entry bar
            are considered for TP/SL (still capped by end of day). ``None`` means no extra
            limit beyond the session.
    """
    targets = np.zeros(len(data), dtype=np.int64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")

    for locs, highs, lows, n in _iter_daily_ohlc(data, trade_date, symbol):
        base = _index_position(data, locs[0])
        day_targets = _targets_for_day_vectorized(
            highs,
            lows,
            n,
            take_profit,
            stop_loss,
            max_bars_after_entry=max_bars_after_entry,
        )
        targets[base : base + n] = day_targets

    data[column_name] = targets
    return data


def add_feature_bars_until_close(
    data: pd.DataFrame,
    *,
    column_name: str = "bars_until_close",
) -> pd.DataFrame:
    """
    Add a column with the number of bars left in the trading day until close,
    excluding the current bar (0 for the last bar of the day).

    Modifies data in place and returns it. Assumes data has MultiIndex (symbol, timestamp).
    Uses America/New_York for date when tz-aware.

    Args:
        column_name: Name of the column to add (default "bars_until_close").
    """
    values = np.zeros(len(data), dtype=np.int64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")

    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(data, locs[0])
        values[base : base + n] = np.arange(n - 1, -1, -1)

    data[column_name] = values
    return data


def add_feature_bars_since_open(
    data: pd.DataFrame,
    *,
    column_name: str = "bars_since_open",
) -> pd.DataFrame:
    """
    Add a column with the number of bars in the trading day before this one
    (0 for the first bar of the day).

    Modifies data in place and returns it. Assumes data has MultiIndex (symbol, timestamp).
    Uses America/New_York for date when tz-aware.

    Args:
        column_name: Name of the column to add (default "bars_since_open").
    """
    values = np.zeros(len(data), dtype=np.int64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")

    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(data, locs[0])
        values[base : base + n] = np.arange(n)

    data[column_name] = values
    return data


def _pct_change_1d(closes: np.ndarray, bars_back: int) -> np.ndarray:
    """Percent change from bars_back ago within one day. Length len(closes) - bars_back."""
    prev = closes[:-bars_back]
    curr = closes[bars_back:]
    out = np.zeros_like(prev, dtype=np.float64)
    np.divide(curr - prev, prev, out=out, where=prev != 0)
    return out


def add_feature_pct_change(
    data: pd.DataFrame,
    bars_back: int,
    *,
    column_name: str | None = None,
) -> pd.DataFrame:
    """
    Add a column with the percent change in close from bars_back bars ago, within the same day.
    If there aren't enough bars back in the day, the value is 0.

    Modifies data in place and returns it. Assumes data has MultiIndex (symbol, timestamp)
    and a 'close' column. Uses America/New_York for date when tz-aware.

    Args:
        bars_back: Number of bars back for the percent change.
        column_name: Name of the column to add. If None, uses 'pct_change_{bars_back}'.
    """
    col = column_name if column_name is not None else f"pct_change_{bars_back}"
    values = np.zeros(len(data), dtype=np.float64)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    closes = data["close"].values

    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(data, locs[0])
        closes_day = closes[base : base + n]
        pct = _pct_change_1d(closes_day, bars_back)
        values[base + bars_back : base + n] = pct

    data[col] = values
    return data


def add_feature_pct_change_batch(
    data: pd.DataFrame,
    bars_back_list: list[int],
    *,
    column_name_fn: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """
    Add multiple pct_change columns in one pass. Avoids DataFrame fragmentation from
    many single-column inserts. Same semantics per column as add_feature_pct_change.

    Args:
        data: DataFrame with MultiIndex (symbol, timestamp) and 'close' column.
        bars_back_list: List of bars_back values (e.g. [1, 2, ..., 119]).
        column_name_fn: If given, called as column_name_fn(bars_back) for each column
            name; else uses f"pct_change_{bars_back}".

    Returns:
        data with new columns added (single concat, no fragmentation).
    """
    if not bars_back_list:
        return data
    name_fn = column_name_fn if column_name_fn is not None else (lambda b: f"pct_change_{b}")
    n_rows = len(data)
    closes = data["close"].values
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    # Preallocate one array per column
    columns = {name_fn(b): np.zeros(n_rows, dtype=np.float64) for b in bars_back_list}

    for (_sym, _date), group in data.groupby([symbol, trade_date]):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(data, locs[0])
        closes_day = closes[base : base + n]
        for bars_back in bars_back_list:
            pct = _pct_change_1d(closes_day, bars_back)
            col_name = name_fn(bars_back)
            columns[col_name][base + bars_back : base + n] = pct

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)

def evaluate_and_print(
    name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    *,
    take_profit: float = 0.04,
    stop_loss: float = 0.02,
    cost: float = 0.004,
) -> None:
    """
    Print binary-classification stats from a confusion matrix (positive class = 1).

    take_profit / stop_loss should match the target definition; cost is per-side fee
    (subtracted on wins, added on losses) in the printed PnL approximation.
    """
    print(f"\n--- {name} ---")
    cm = confusion_matrix(y_test, y_pred)
    tp_count = int(cm[1, 1])
    fp_count = int(cm[0, 1])
    fn_count = int(cm[1, 0])

    pos_denom = tp_count + fn_count
    pred_pos_denom = tp_count + fp_count
    recall_pct = 100 * tp_count / pos_denom if pos_denom else 0.0
    precision_pct = 100 * tp_count / pred_pos_denom if pred_pos_denom else 0.0
    loss_pct = 100 * fp_count / pred_pos_denom if pred_pos_denom else 0.0

    print(f"Profitable Trades Taken (Recall): {tp_count:,}/{pos_denom:,} ({recall_pct:,.2f}%)")
    print(f"Win Rate (Precision): {tp_count:,}/{pred_pos_denom:,} ({precision_pct:,.2f}%)")
    print(f"Loss Rate: {fp_count:,}/{pred_pos_denom:,} ({loss_pct:,.2f}%)")
    win_per = take_profit - cost
    loss_per = stop_loss + cost
    print(
        f"Profit: {100 * ((tp_count * win_per) - (fp_count * loss_per)):,.2f}% "
        f"of max possible {100 * (pos_denom * win_per):,.2f}%"
    )

def calculate_min_win_rate(take_profit: float, stop_loss: float, cost: float) -> float:
    """
    Calculate the minimum win rate required to break even.
    """
    return (stop_loss + cost) / (take_profit + stop_loss)
