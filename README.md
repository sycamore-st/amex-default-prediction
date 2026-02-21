# AMEX Default Prediction

Kaggle project for **American Express - Default Prediction**.
Google Colab: https://colab.research.google.com/drive/1esDn3W4uVg1T8FkYZyUHhU5YFc-lg3rx?usp=sharing

## Primary Notebook
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`
  - `v1`: baseline incremental LightGBM
  - `v2`: advanced production ensemble
  - `v3`: two-stage advanced model (statement meta -> customer model)
  - `v4`: sequence embedding model (bi-GRU denoising autoencoder + LightGBM)
  - `blend`: rank blend of `v2 + v3 + v4`

## Data Path (Colab)
Default root:
`/content/drive/MyDrive/amex_data_parquet`

Required files:
- `train_data.parquet` or `train_data.csv`
- `test_data.parquet` or `test_data.csv`
- `train_labels.csv`
- `sample_submission.csv`

## Outputs
- `advanced_submission_v2.csv`
- `advanced_submission_v3.csv`
- `v4_submission.csv`
- `blend_v234_submission.csv`

## Notes
- Choose one run target with `TRAIN_VERSION` in setup cell: `v1` / `v2` / `v3` / `v4`.
- Run blend section after `v2`, `v3`, and `v4` submissions are generated.
