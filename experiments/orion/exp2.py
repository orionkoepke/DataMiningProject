"""Experiment 2 — minute bars for one symbol: engineered table, split, z-scored features, classifiers.

Data is loaded via ``elib.pull_and_clean``. The training table keeps ``range_target``,
``target`` (TP before SL / horizon / EOD, same as ``lib.common.common.create_target_column``),
and engineered columns (no raw OHLCV), matching the slim-frame pattern in
``create_orion_training_data``.
Features are standardized with ``sklearn.preprocessing.StandardScaler`` fit on the train split only
(``zscore_exp2_tables``; binary targets are excluded). Volume columns are rolling within-day means from
``add_exp2_volume_features``; their global z-scoring happens only after the chronological split.

Trains the same model zoo as ``experiments.orion.experiment`` (fixed hyperparameters from a
representative grid search) and evaluates on the held-out test split via
``print_exp2_test_results`` (accuracy and confusion-matrix counts).

Parameters live in ``main``; this module does not use ``experiments.orion.config``.

Run from project root::

    python -m experiments.orion.exp2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from lib.models import (
    train_adaboost,
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_xgboost,
)

from experiments.orion.elib import (
    add_feature_bars_since_open,
    add_feature_bars_until_close,
    add_feature_close_sma_pct_diff,
    print_training_data_stats,
    pull_and_clean,
    split_training_data,
)
from lib.common.common import (
    _index_position,
    _trade_date_series,
    add_range_target_column,
    create_target_column,
)

# From grid search console (val F1); keys match each ``train_*`` keyword arguments —
# same defaults as ``experiments.orion.experiment``.
_OPTIMAL: dict[str, dict[str, Any]] = {
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 100},
    "naive_bayes": {"var_smoothing": 1e-9},
    "knn": {"n_neighbors": 5, "p": 2, "weights": "uniform"},
    "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 100},
    "adaboost": {"n_estimators": 50, "learning_rate": 0.5},
    "xgboost": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "subsample": 1.0,
    },
}

VOLUME_ROLL_WINDOWS = (1, 2, 5, 10, 20, 30, 60, 90, 180)
# Same periods as ``create_orion_training_data`` in ``elib`` (``close_sma_{n}_pct_diff``).
CLOSE_SMA_PERIODS = (1, 2, 5, 10, 20, 30, 60, 90, 180)
# EMA spans match SMA windows; ``alpha = 2/(span+1)`` (pandas ``ewm(span=..., adjust=False)``).
CLOSE_EMA_SPANS = CLOSE_SMA_PERIODS
EXP2_TARGET_COLUMN = "range_target"
# Same definition as ``create_orion_training_data`` / ``load_orion_training_frames`` (TP before SL).
EXP2_TP_SL_TARGET_COLUMN = "target"
EXP2_COLUMNS_EXCLUDED_FROM_ZSCORE: tuple[str, ...] = (
    EXP2_TARGET_COLUMN,
    EXP2_TP_SL_TARGET_COLUMN,
)


def exp2_training_column_names() -> list[str]:
    """Targets first, then features (same convention as ``create_orion_training_data``)."""
    return [
        EXP2_TARGET_COLUMN,
        EXP2_TP_SL_TARGET_COLUMN,
        "bars_until_close",
        "bars_since_open",
        *[f"volume_roll_mean_{w}" for w in VOLUME_ROLL_WINDOWS],
        *[f"close_sma_{p}_pct_diff" for p in CLOSE_SMA_PERIODS],
        *[f"close_ema_{p}_pct_diff" for p in CLOSE_EMA_SPANS],
    ]


def _rolling_volume_mean_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of ``values`` over ``window`` bars; leading incomplete windows are 0."""
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    if window < 1 or n == 0:
        return out
    s = pd.Series(values, dtype=np.float64, copy=False)
    rolled = s.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(rolled)
    out[valid] = rolled[valid]
    return out


def add_exp2_volume_features(
    data: pd.DataFrame,
    window_list: list[int],
) -> pd.DataFrame:
    """Add per-day rolling **mean volume** columns (not z-scored).

    For each window, within each (symbol, trade_date) session, ``volume_roll_mean_w`` is the mean
    of ``volume`` over the last ``w`` bars including the current bar. Rows before ``w`` bars
    exist in that session are 0. Global z-scoring is applied later via ``zscore_feature_columns``
    after train/validation/test split.
    """
    if not window_list:
        return data
    for w in window_list:
        if w < 1:
            raise ValueError("each volume window size must be >= 1")
    n_rows = len(data)
    trade_date = _trade_date_series(data)
    symbol = data.index.get_level_values("symbol")
    vol_all = data["volume"].to_numpy(dtype=np.float64, copy=False)
    columns = {
        f"volume_roll_mean_{w}": np.zeros(n_rows, dtype=np.float64) for w in window_list
    }

    for (_sym, _date), group in data.groupby([symbol, trade_date], sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        seg = vol_all[base : base + n]
        for w in window_list:
            columns[f"volume_roll_mean_{w}"][base : base + n] = _rolling_volume_mean_1d(seg, w)

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def _ema_1d(close: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average of ``close``; leading bars before ``span`` samples are 0."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if span < 1 or n == 0:
        return out
    s = pd.Series(close, dtype=np.float64, copy=False)
    ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
    arr = ema.to_numpy(dtype=np.float64, copy=False)
    out[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _pct_diff_close_vs_reference(close: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``(close - ref) / ref`` with 0 where ``ref`` is not finite or not positive."""
    out = np.zeros(len(close), dtype=np.float64)
    valid = np.isfinite(ref) & (ref > 0) & np.isfinite(close)
    out[valid] = (close[valid] - ref[valid]) / ref[valid]
    return out


def add_exp2_close_ema_pct_diff(
    data: pd.DataFrame,
    span_list: list[int],
) -> pd.DataFrame:
    """Add ``(close - EMA) / EMA`` per span, timestamp order per symbol (continuous across days).

    Same grouping as ``add_feature_close_sma_pct_diff`` in ``elib``. EMA uses pandas
    ``ewm(span=span, adjust=False, min_periods=span)``. Column names: ``close_ema_{span}_pct_diff``.
    """
    if not span_list:
        return data
    for sp in span_list:
        if sp < 1:
            raise ValueError("each EMA span must be >= 1")
    n_rows = len(data)
    symbol = data.index.get_level_values("symbol")
    columns = {
        f"close_ema_{sp}_pct_diff": np.zeros(n_rows, dtype=np.float64) for sp in span_list
    }

    for _sym, group in data.groupby(symbol, sort=False):
        group = group.sort_index(level="timestamp")
        base = _index_position(data, group.index[0])
        n = len(group)
        close = group["close"].to_numpy(dtype=np.float64, copy=False)
        for sp in span_list:
            ema = _ema_1d(close, sp)
            columns[f"close_ema_{sp}_pct_diff"][base : base + n] = _pct_diff_close_vs_reference(
                close,
                ema,
            )

    new_df = pd.DataFrame(columns, index=data.index)
    return pd.concat([data, new_df], axis=1)


def add_features(
    bars: pd.DataFrame,
    *,
    take_profit: float,
    stop_loss: float,
    max_bars_after_entry: int | None = None,
) -> pd.DataFrame:
    """Build target + features, then keep only those columns (MultiIndex ``symbol``, ``timestamp``).

    Raw OHLCV and other base columns are dropped; same idea as ``create_orion_training_data`` returning
    ``data[col_names]``.
    """
    out = bars.copy()
    add_range_target_column(
        out,
        take_profit,
        stop_loss,
        max_bars_after_entry=max_bars_after_entry,
    )
    create_target_column(
        out,
        take_profit,
        stop_loss,
        column_name=EXP2_TP_SL_TARGET_COLUMN,
        max_bars_after_entry=max_bars_after_entry,
    )
    add_feature_bars_until_close(out)
    add_feature_bars_since_open(out)
    out = add_exp2_volume_features(out, list(VOLUME_ROLL_WINDOWS))
    out = add_feature_close_sma_pct_diff(out, list(CLOSE_SMA_PERIODS))
    out = add_exp2_close_ema_pct_diff(out, list(CLOSE_EMA_SPANS))
    cols = exp2_training_column_names()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"add_features: expected columns missing: {missing}")
    return out[cols].copy()


def zscore_exp2_tables(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    non_feature_columns: tuple[str, ...] = EXP2_COLUMNS_EXCLUDED_FROM_ZSCORE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit ``StandardScaler`` on train features only; transform all three splits.

    Columns listed in ``non_feature_columns`` (binary targets) are not scaled.
    """
    exclude = set(non_feature_columns)
    feature_cols = [c for c in train_df.columns if c not in exclude]
    scaler = StandardScaler()
    train_out = train_df.copy()
    val_out = validation_df.copy()
    test_out = test_df.copy()
    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_out[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_out, val_out, test_out


def print_preamble(bars: pd.DataFrame) -> None:
    print(f"====================== Preamble ======================")
    print(f"Symbol: {bars.index.levels[0][0]}")
    print(f"Date Range: {bars.index.levels[1][0].date()} to {bars.index.levels[1][-1].date()}")
    print(f"Total Bars: {len(bars):,}")
    print("========================================================")


def print_exp2_test_results(
    name: str,
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print test accuracy and confusion-matrix cell counts (binary labels 0 and 1)."""
    y_true = np.asarray(y_test).ravel()
    y_hat = np.asarray(y_pred).ravel()
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_hat, labels=labels)
    acc = float(accuracy_score(y_true, y_hat))
    recall = float(recall_score(y_true, y_hat))
    precision = float(precision_score(y_true, y_hat))

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    n = tn + fp + fn + tp

    print(f"\n--- {name} ---")
    print(f"Accuracy: {(tn + tp):,} / {n:,} ({acc:.6f} ({100.0 * acc:.4f}%)")
    print(f"Oportunities Identified (Recall): {tp:,} / {tp + fn:,} ({recall:.6f} ({100.0 * recall:.4f}%)")
    print(f"Win Rate (Precision): {tp:,} / {tp + fp:,} ({precision:.6f} ({100.0 * precision:.4f}%)")
    print(f"F1 Score: {2 * tp / (2 * tp + fp + fn):.6f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"  True negatives (actual 0, pred 0):  {tn:,}")
    print(f"  False positives (actual 0, pred 1): {fp:,}")
    print(f"  False negatives (actual 1, pred 0): {fn:,}")
    print(f"  True positives (actual 1, pred 1):  {tp:,}")


def main() -> None:
    # Local run parameters (edit here; this script does not use experiments.orion.config).
    SYMBOL = "MARA"
    START_DATE = datetime(2022, 1, 1)
    END_DATE = datetime(2025, 12, 31)
    TAKE_PROFIT = 0.03
    STOP_LOSS = 0.03
    MAX_BARS_AFTER_ENTRY = 90
    VALIDATION_FRACTION = 0.15
    TEST_FRACTION = 0.2

    bars = pull_and_clean(SYMBOL, START_DATE, END_DATE)
    bars = bars.sort_index()

    training_table = add_features(
        bars,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        max_bars_after_entry=MAX_BARS_AFTER_ENTRY,
    )

    print_preamble(training_table)
    train_df, val_df, test_df = split_training_data(
        training_table,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    train_df, val_df, test_df = zscore_exp2_tables(train_df, val_df, test_df)

    x_test = test_df.drop(columns=[EXP2_TARGET_COLUMN, EXP2_TP_SL_TARGET_COLUMN])
    y_test = test_df[EXP2_TARGET_COLUMN]
    p = _OPTIMAL

    print_training_data_stats(
        train_df,
        val_df,
        test_df,
        target_column=EXP2_TARGET_COLUMN,
    )
    print_training_data_stats(
        train_df,
        val_df,
        test_df,
        target_column=EXP2_TP_SL_TARGET_COLUMN,
    )

    print("\n====================== Experiment 2 — test metrics ======================")

    tree_clf = train_decision_tree(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["decision_tree"],
    )
    print_exp2_test_results("Decision Tree", y_test, tree_clf.predict(x_test))

    nb_clf = train_naive_bayes(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["naive_bayes"],
    )
    print_exp2_test_results("Naive Bayes", y_test, nb_clf.predict(x_test))

    knn_clf = train_knn(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["knn"],
    )
    print_exp2_test_results("K-Nearest Neighbors", y_test, knn_clf.predict(x_test))

    forest_clf = train_forest(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["random_forest"],
    )
    print_exp2_test_results("Random Forest", y_test, forest_clf.predict(x_test))

    adaboost_clf = train_adaboost(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["adaboost"],
    )
    print_exp2_test_results("AdaBoost", y_test, adaboost_clf.predict(x_test))

    xgb_clf = train_xgboost(
        train_df,
        val_df,
        target_column=EXP2_TARGET_COLUMN,
        param_grid=None,
        verbose=False,
        **p["xgboost"],
    )
    print_exp2_test_results("XGBoost", y_test, xgb_clf.predict(x_test))


if __name__ == "__main__":
    main()
