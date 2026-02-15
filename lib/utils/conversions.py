"""
Conversion utilities (e.g. Alpaca TimeFrame to standard library types).
"""

from datetime import timedelta

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def timeframe_to_timedelta(timeframe: TimeFrame) -> timedelta:
    """Convert Alpaca TimeFrame to timedelta."""
    amount = timeframe.amount
    unit = timeframe.unit
    if unit == TimeFrameUnit.Minute:
        return timedelta(minutes=amount)
    if unit == TimeFrameUnit.Hour:
        return timedelta(hours=amount)
    if unit == TimeFrameUnit.Day:
        return timedelta(days=amount)
    if unit == TimeFrameUnit.Week:
        return timedelta(weeks=amount)
    if unit == TimeFrameUnit.Month:
        return timedelta(days=30 * amount)
    raise ValueError(f"Unsupported timeframe unit: {unit}")
