"""Hypothetical gross returns if we enter on every minute bar with TP / SL / horizon exit.

No per-trade fees: returns are ``exit / entry - 1``. Uses cleaned minute bars from
``elib.pull_and_clean``. Per signal bar *i* (same indexing as
``create_target_column``): buy at **high** of bar *i+1*; then scan bars *i+2* … up to
``max_bars_after_entry`` bars or end of session. If a bar's **low** ≤ stop price, exit at
stop price; else if **high** ≥ take-profit price, exit at take-profit price (same bar: stop
wins, matching ``lib.common.common``). If neither triggers, exit at the **low** of the last
bar in that window.

Run from project root::

    python -m experiments.orion.returns
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from experiments.orion.elib import pull_and_clean
from lib.common.common import _index_position, _trade_date_series

# Exit labels aligned with ``gross_returns_entry_next_high_tp_sl_horizon`` second return value.
EXIT_NONE = -1
EXIT_TAKE_PROFIT = 0
EXIT_STOP_LOSS = 1
EXIT_HORIZON = 2


def gross_returns_entry_next_high_tp_sl_horizon(
    bars: pd.DataFrame,
    *,
    take_profit: float,
    stop_loss: float,
    max_bars_after_entry: int,
) -> tuple[pd.Series, pd.Series]:
    """Per-row gross simple return (fraction) for a long from next bar's high to rule-based exit.

    No fees: ``exit_price / entry_price - 1``. Return is stored on the **signal** bar row
    (bar *i*). Last 3 bars of each session are NaN (insufficient horizon, same as target
    construction).

    Returns:
        ``(gross_returns, exit_kind)`` where ``exit_kind`` is int8: ``EXIT_NONE`` (-1) if no trade,
        else ``EXIT_TAKE_PROFIT``, ``EXIT_STOP_LOSS``, or ``EXIT_HORIZON``.
    """
    if max_bars_after_entry < 1:
        raise ValueError("max_bars_after_entry must be >= 1")

    out = np.full(len(bars), np.nan, dtype=np.float64)
    kinds = np.full(len(bars), EXIT_NONE, dtype=np.int8)
    trade_date = _trade_date_series(bars)
    symbol = bars.index.get_level_values("symbol")

    tp = float(take_profit)
    sl = float(stop_loss)

    for (_sym, _date), group in bars.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        locs = group.index
        n = len(group)
        base = _index_position(bars, locs[0])
        highs = group["high"].to_numpy(dtype=np.float64, copy=False)
        lows = group["low"].to_numpy(dtype=np.float64, copy=False)

        num_signals = max(0, n - 3)
        for i in range(num_signals):
            entry_price = highs[i + 1]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            sl_price = entry_price * (1.0 - sl)
            tp_price = entry_price * (1.0 + tp)
            end_idx = min(n, i + 2 + max_bars_after_entry)

            exit_price: float | None = None
            exit_kind: int = EXIT_NONE
            for j in range(i + 2, end_idx):
                lo = lows[j]
                hi = highs[j]
                if lo <= sl_price:
                    exit_price = sl_price
                    exit_kind = EXIT_STOP_LOSS
                    break
                if hi >= tp_price:
                    exit_price = tp_price
                    exit_kind = EXIT_TAKE_PROFIT
                    break
            if exit_price is None:
                if end_idx - 1 < i + 2:
                    continue
                exit_price = float(lows[end_idx - 1])
                exit_kind = EXIT_HORIZON

            out[base + i] = exit_price / entry_price - 1.0
            kinds[base + i] = exit_kind

    ret_s = pd.Series(out, index=bars.index, name="gross_ret_tp_sl")
    kind_s = pd.Series(kinds, index=bars.index, name="exit_kind")
    return ret_s, kind_s


def _count_and_pct(count: int, total: int) -> str:
    """Format count and percent of total, e.g. ``12,345 (34.56%)``."""
    if total <= 0:
        return f"{count:,} (--%)"
    pct = 100.0 * count / total
    return f"{count:,} ({pct:.2f}%)"


def print_trade_every_minute_summary(
    per_bar_gross: pd.Series,
    exit_kind: pd.Series,
) -> None:
    valid = per_bar_gross.notna() & np.isfinite(
        per_bar_gross.to_numpy(dtype=np.float64, copy=False)
    )
    r = per_bar_gross[valid]
    k = exit_kind[valid].to_numpy(dtype=np.int8, copy=False)

    n = len(r)
    print("\n--- Every-minute trade (gross, no fees; next-bar-high entry, TP / SL / horizon) ---")
    print(f"Trades: {n:,}")
    if n == 0:
        return

    n_tp = int((k == EXIT_TAKE_PROFIT).sum())
    n_sl = int((k == EXIT_STOP_LOSS).sum())
    n_hz = int((k == EXIT_HORIZON).sum())
    print(f"Hit take profit: {_count_and_pct(n_tp, n)}")
    print(f"Hit stop loss: {_count_and_pct(n_sl, n)}")
    print(f"Neither (exit at horizon bar low): {_count_and_pct(n_hz, n)}")

    hz_mask = k == EXIT_HORIZON
    if hz_mask.any():
        hz_mean = float(r.to_numpy()[hz_mask].mean())
        print(f"Mean gross return when neither hit (%): {100.0 * hz_mean:.6f}")
    else:
        print("Mean gross return when neither hit (%): n/a (no such trades)")

    print(f"Mean gross return per trade (all) (%): {100.0 * float(r.mean()):.6f}")
    print(f"Std dev gross return per trade (all) (%): {100.0 * float(r.std(ddof=0)):.6f}")


def main() -> None:
    # Local run parameters (edit here; this script does not use experiments.orion.config).
    SYMBOL = "MARA"
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    TAKE_PROFIT = 0.03
    STOP_LOSS = 0.03
    MAX_BARS_AFTER_ENTRY = 90

    bars = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    bars = bars.sort_index()

    per_bar, exit_kind = gross_returns_entry_next_high_tp_sl_horizon(
        bars,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        max_bars_after_entry=MAX_BARS_AFTER_ENTRY,
    )
    print_trade_every_minute_summary(per_bar, exit_kind)


if __name__ == "__main__":
    main()
