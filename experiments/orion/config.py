"""Shared defaults for orion experiment scripts (symbol, dates, costs, splits)."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OrionExperimentConfig:
    """Configuration for building the training table and evaluating models."""

    symbol: str = "MARA"
    reference_symbols: tuple[str, ...] = ("SPY", "IBIT", "WGMI", "CLSK", "MSTR")
    start_date: datetime = datetime(2022, 1, 1)
    end_date: datetime = datetime(2025, 12, 31)
    take_profit: float = 0.01
    stop_loss: float = 0.01
    target_max_bars_after_entry: int | None = 15
    trade_cost: float = 0.004
    validation_fraction: float = 0.15
    test_fraction: float = 0.2


DEFAULT_ORION_CONFIG = OrionExperimentConfig()
