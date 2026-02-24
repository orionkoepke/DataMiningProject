"""
Fee helpers for backtest: Alpaca regulatory fees and round-up to cent.
"""

import math

from lib.framework.orders import Fill, OrderSide

# Alpaca pass-through regulatory fees (equities)
TAF_RATE_PER_SHARE = 0.000166  # FINRA TAF, sells only
TAF_MAX_PER_TRADE = 8.30
CAT_RATE_PER_SHARE = 0.00003   # FINRA-CAT, buys and sells


def round_up_to_cent(amount: float) -> float:
    """Round up to the nearest $0.01 (Alpaca behavior for fee deduction)."""
    return math.ceil(amount * 100) / 100


def alpaca_regulatory_fee(fill: Fill) -> float:
    """
    Alpaca pass-through regulatory fees for equities.
    TAF: sells only, $0.000166/share, max $8.30/trade.
    CAT: buys and sells, $0.00003/share.
    Each component is rounded up to the nearest cent, then summed.
    """
    cat = round_up_to_cent(CAT_RATE_PER_SHARE * fill.qty)
    if fill.side == OrderSide.SELL:
        taf = round_up_to_cent(min(TAF_RATE_PER_SHARE * fill.qty, TAF_MAX_PER_TRADE))
        return taf + cat
    return cat
