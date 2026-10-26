# Model Comparison Report

The notebook compares five classifiers using Stratified 5-Fold Cross-Validation and a held-out 20% test set.

| Model | CV Accuracy | Test Accuracy | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.9725 ± 0.0139 | 0.9595 | 0.9595 |
| Extra Trees | 0.9760 ± 0.0085 | 0.9595 | 0.9595 |
| XGBoost | 0.9656 ± 0.0189 | 0.9189 | 0.9186 |
| LightGBM | 0.9656 ± 0.0189 | 0.9189 | 0.9157 |
| SVM | 0.9760 ± 0.0085 | **0.9730** | **0.9730** |

## Recommendation

Based strictly on the recorded held-out test results, SVM is the strongest predictive model in this notebook.

Extra Trees is retained for detailed interpretability because the notebook uses it for the confusion matrix and Gini feature-importance analysis.

This is an educational result and should not be interpreted as clinical validation.
