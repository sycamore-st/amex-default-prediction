# AMEX Default Prediction

Interview-focused Kaggle project for **American Express - Default Prediction** with a clean single-notebook workflow.

## Repo Layout
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`: single end-to-end notebook
- `/Users/claire/PycharmProjects/amex-default-prediction/data/raw/amex-default-prediction/`: local raw data (optional for local runs)
- `/Users/claire/PycharmProjects/amex-default-prediction/requirements.txt`: dependencies

## Modeling Versions
- `v1`: baseline incremental LightGBM
- `v2`: advanced production ensemble (this is the former `v3` pipeline)
- `v3`: additional online-inspired meta-ensemble layer on top of `v2` part predictions

## Run Notes
- In notebook config cell, choose one: `TRAIN_VERSION = 'v1'` or `'v2'` or `'v3'`.
- Recommended order for leaderboard push:
  1. Run `v2` to produce `adv_v2_part_*.csv` and `advanced_submission_v2.csv`
  2. Run `v3` to produce `advanced_submission_v3.csv`
- Full-run cache sanity checks are enforced:
  - min label customers: `400000`
  - min feature customers: `350000`
  - min coverage ratio: `0.85`

## Data Path (Colab)
Notebook default data root:
`/content/drive/MyDrive/amex_data_parquet`

Required files:
- `train_data.parquet`
- `test_data.parquet`
- `train_labels.csv`
- `sample_submission.csv`
