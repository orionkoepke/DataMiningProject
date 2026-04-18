# Zhongjie Results

This attempt started from the same Orion pipeline and reused the existing data fetching, feature construction, target definition, chronological split, and z-score normalization. On top of that setup, I added a new linear / SVM baseline line, including Logistic Regression, LinearSVC, and RBF SVM. I also update the evaluation function which only predicted positive trades contribute to realized profit.

## Evaluation function

I updated the evaluation function to evaluate_and_print_trade_zhongjie(). The main change was to make the profit calculation match a trade-only interpretation. The original evaluate_and_print() was closer to a classification-style profit approximation, because it also gave weight to correct negative predictions. For my experiments, that was not the behavior I wanted. I only wanted rows predicted as positive to affect realized profit, since those are the trades that would actually be taken. True positives are treated as winning trades and false positives as losing trades, while true negatives and false negatives are not counted in realized profit. 

## Result

I focused on model types that were not part of the original Orion experiments, namely Logistic Regression and SVM-based models. The goal was to see whether a linear or margin-based classifier could produce a cleaner trading signal under the same Orion pipeline, including the same data source, feature construction, target definition, and chronological train / validation / test split.

Three models are tested:

- `LogisticRegression`
- `LinearSVC`
- `RBF SVM`

For each model, I tested multiple values of `C` and decision thresholds. For `RBF SVM`, I also varied `gamma`. This made it possible to compare not only the final test result, but also how sensitive each model was to parameter choice and trade-selection threshold.

| Model | C | Threshold | Gamma | Validation Trades | Validation Precision | Validation Profit | Test Trades | Test Precision | Test Profit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.1 | 0.7 | — | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |
| LinearSVC | 1.0 | 0.5 | — | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |
| RBF SVM | 0.1 | -0.5 | scale | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |

## Conclusion
Overall, the experiments suggest a clear pattern. Under the current Orion setup, the linear / SVM baselines did not all perform equally well. Logistic Regression was too aggressive, and RBF SVM was not practically useful, while the processing speed is also very slow. LinearSVC was the most promising of the three, especially in a conservative parameter region around C≈1 and threshold≈0.5. In that region, the model achieved a test precision slightly above the estimated 55% break-even win rate, although the validation signal remained unstable. This suggests that the current feature set may contain some usable directional information, but the selection process is still sensitive and would likely benefit from a more robust validation design.