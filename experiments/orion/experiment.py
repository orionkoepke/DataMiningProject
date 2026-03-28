"""Load orion data, train classifiers with fixed hyperparameters, evaluate on the test split.

Best params below match a representative grid-search run (MARA, config in ``config.py``).
Update the dict after re-tuning if you want this script to mirror new optima.

Run from project root::

    python -m experiments.orion.experiment
"""

from typing import Any

from lib.common.common import evaluate_and_print
from lib.models import (
    train_adaboost,
    train_decision_tree,
    train_forest,
    train_knn,
    train_naive_bayes,
    train_xgboost,
)

from experiments.orion.config import DEFAULT_ORION_CONFIG
from experiments.orion.elib import load_orion_training_frames, print_orion_run_preamble

# From grid search console (val F1); keys match each ``train_*`` keyword arguments.
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


def main() -> None:
    cfg = DEFAULT_ORION_CONFIG
    trade_cost = cfg.trade_cost

    print_orion_run_preamble(cfg, trade_cost=trade_cost, run_title="Experiment (fixed params)")

    train_df, val_df, test_df = load_orion_training_frames(
        symbol=cfg.symbol,
        reference_symbols=cfg.reference_symbols,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
        target_max_bars_after_entry=cfg.target_max_bars_after_entry,
    )

    x_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    p = _OPTIMAL

    tree_clf = train_decision_tree(
        train_df, val_df, param_grid=None, verbose=False, **p["decision_tree"]
    )
    evaluate_and_print(
        "Decision Tree",
        y_test,
        tree_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    nb_clf = train_naive_bayes(train_df, val_df, param_grid=None, verbose=False, **p["naive_bayes"])
    evaluate_and_print(
        "Naive Bayes",
        y_test,
        nb_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    knn_clf = train_knn(train_df, val_df, param_grid=None, verbose=False, **p["knn"])
    evaluate_and_print(
        "K-Nearest Neighbors",
        y_test,
        knn_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    forest_clf = train_forest(train_df, val_df, param_grid=None, verbose=False, **p["random_forest"])
    evaluate_and_print(
        "Random Forest",
        y_test,
        forest_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    adaboost_clf = train_adaboost(
        train_df, val_df, param_grid=None, verbose=False, **p["adaboost"]
    )
    evaluate_and_print(
        "AdaBoost",
        y_test,
        adaboost_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )

    xgb_clf = train_xgboost(train_df, val_df, param_grid=None, verbose=False, **p["xgboost"])
    evaluate_and_print(
        "XGBoost",
        y_test,
        xgb_clf.predict(x_test),
        take_profit=cfg.take_profit,
        stop_loss=cfg.stop_loss,
        cost=trade_cost,
    )


if __name__ == "__main__":
    main()
