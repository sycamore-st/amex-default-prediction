# AMEX Default Prediction

Interview-focused Kaggle project for **American Express - Default Prediction**.

## Notebooks
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`
  - `v1`: baseline incremental LightGBM
  - `v2`: advanced production ensemble
  - `v3`: two-stage advanced model (statement meta -> customer model)
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_v4_sequence_embeddings.ipynb`
  - sequence embedding representation learning (GRU autoencoder)
  - merge embeddings with tabular features
  - train LightGBM on hybrid features

## Data Path (Colab)
Default root:
`/content/drive/MyDrive/amex_data_parquet`

Required files:
- `train_data.parquet`
- `test_data.parquet` (needed for submission pipelines)
- `train_labels.csv`
- `sample_submission.csv` (needed for submission pipelines)

## Notes
- Use `v2`/`v3` for leaderboard submission flow.
- Use `v4` to experiment with sequence representation learning while keeping Colab memory stable via subsampling.
