# AMEX Default Prediction

Client behavior modeling project based on Kaggle's **American Express - Default Prediction** challenge. The project is designed to be interview-ready with a clear ML workflow, leakage-aware validation, interpretable feature strategy, and business-focused model evaluation.

## Project Structure

- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`: Main end-to-end notebook
- `/Users/claire/PycharmProjects/amex-default-prediction/docs/project_summary.md`: Problem framing, modeling strategy, and recommendations
- `/Users/claire/PycharmProjects/amex-default-prediction/data/raw/amex-default-prediction/`: Place Kaggle dataset files here
- `/Users/claire/PycharmProjects/amex-default-prediction/requirements.txt`: Dependencies

## Setup

1. Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download Kaggle AMEX dataset and place files in:

`/Users/claire/PycharmProjects/amex-default-prediction/data/raw/amex-default-prediction/`

Expected files:
- `train_data.csv` or `train_data.parquet`
- `train_labels.csv`

3. Launch notebook:

```bash
jupyter notebook
```

Then open `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`.

## Workflow Implemented

1. Data loading (with optional downsampling for quick iteration)
2. EDA (class balance, missingness, sequence length profile)
3. Leakage-safe, customer-level feature engineering:
   - recency/history/activity
   - aggregate stats (`last`, `mean`, `std`, `min`, `max`)
   - trend features (per-customer slope)
   - missingness behavior features
   - categorical last/nunique signals
4. Time-aware train/validation split by each customer's latest statement date
5. Baseline models:
   - Logistic Regression (interpretable baseline)
   - LightGBM (preferred) or XGBoost fallback
6. Evaluation:
   - ROC-AUC
   - PR-AUC
   - decile lift table/curve
7. Explainability:
   - global feature importance
   - SHAP summary for top risk drivers
8. Segment insights by risk decile

## Key Modeling Guardrails

- No direct use of future observations across train/validation boundary.
- Split strategy is time-aware at the customer level.
- Feature construction is sequence-based and supports behavior interpretation.
- Baseline vs improved model comparison is explicit.

## Next Iteration Ideas

- Out-of-fold target encoding for high-cardinality variables.
- Add custom AMEX metric implementation.
- Add calibrated probabilities and threshold policy analysis.
- Modularize notebook logic into `/Users/claire/PycharmProjects/amex-default-prediction/src/` scripts for production pipelines.
