# AMEX Default Prediction

Interview-focused Kaggle project for **American Express - Default Prediction**.

## Notebooks
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_default_prediction.ipynb`
  - `v1`: baseline incremental LightGBM
  - `v2`: advanced production ensemble
  - `v3`: two-stage advanced model (statement meta -> customer model)
  - `v2+v3+v4` rank blend cell saves `blend_v234_submission.csv`
- `/Users/claire/PycharmProjects/amex-default-prediction/notebooks/amex_v4_sequence_embeddings.ipynb`
  - upgraded sequence representation learning (bi-GRU denoising autoencoder)
  - delta-sequence features + early stopping + larger embeddings
  - merge embeddings with tabular aggregates and train LightGBM
  - test inference saves `v4_submission.csv`

## Data Path (Colab)
Default root:
`/content/drive/MyDrive/amex_data_parquet`

Required files:
- `train_data.parquet` or `train_data.csv`
- `test_data.parquet` / `test_data.csv` (needed for submission pipelines)
- `train_labels.csv`
- `sample_submission.csv` (needed for submission pipelines)

## Notes
- Use `v2`/`v3` for leaderboard submission flow.
- Use `v4` for deeper sequence modeling.
- If runtime is tight in Colab, reduce in `v4` config:
  - `train_customer_sample`
  - `embed_dim`, `hidden_dim`
  - `epochs`
