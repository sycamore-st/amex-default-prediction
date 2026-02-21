### Project Title
AMEX Default Prediction: Temporal Behavior Modeling with Hybrid Tabular + Sequence Ensembling

### Problem Framing
The business problem is to estimate **default risk per customer** from monthly statement histories and produce a **risk ranking** that can support credit-risk decisions (line management, collections prioritization, proactive outreach).

This is not just binary classification. In production, teams typically act on ranked queues (top-risk deciles) rather than a single global threshold. That is why ranking quality and lift concentration matter as much as raw classification accuracy.

### Why This Problem Is Technically Hard
1. The dataset is **longitudinal**: each customer has a variable-length sequence of statements.
2. The feature space is mixed-type and noisy with high missingness.
3. Naive random splits can leak temporal information and overstate performance.
4. Large file sizes make iterative experimentation difficult in constrained environments (Colab).

### Modeling Strategy Overview
The project uses a staged strategy:
1. Build reliable tabular baselines from customer-level aggregates.
2. Add stronger ensemble variants that preserve operational reliability.
3. Add sequence representation learning (v4) for additional signal diversity.
4. Blend independent models with rank-based weighting for robust leaderboard and portfolio performance.

### Data Design Choices (What + Why)
#### Customer-level target
Target is defined per customer, so final training table is one row per `customer_ID` for core models.

Why:
- Aligns directly with prediction objective (default risk at customer level).
- Makes feature governance and explainability easier for technical and business stakeholders.

#### Time-aware split
Validation is based on customer timeline (`last_statement_date`), not pure random row split.

Why:
- Reduces temporal leakage risk.
- Better approximates real deployment where predictions are made on future customers/statements.

#### Chunked feature store
Feature generation and model training use chunked parquet paths in Drive.

Why:
- Keeps memory bounded in Colab.
- Enables restart-safe long-running jobs.
- Supports full-data runs without requiring large RAM machines.

### Feature Engineering: What Was Built and Why It Works
The feature strategy intentionally captures behavioral dynamics, not just static values.

#### 1) Distributional summaries per variable
- `last`, `mean`, `std`, `min`, `max`

Why:
- `last` captures current state.
- `mean` captures persistent behavior level.
- `std` captures stability vs volatility (often predictive in credit behavior).
- `min/max` capture stress/extremes that average can hide.

#### 2) Temporal behavior features
- `recency_days`, `history_days`, `n_statements`
- trend slopes (`*_trend`) for selected numeric signals

Why:
- Recency and activity often proxy engagement/behavior shifts.
- Trend direction is critical when deterioration is gradual.

#### 3) Missingness behavior
- per-feature missing rates (`*_missing_rate`)

Why:
- In risk data, missingness is often informative and non-random.
- Behavioral missingness can act as a latent process signal.

#### 4) Categorical behavior persistence/diversity
- `*_last_cat`, `*_nunique_cat`

Why:
- Last category can encode current state.
- In-history category diversity can reflect instability/transition dynamics.

#### 5) v3 two-stage meta features
- statement-level model predictions aggregated to customer-level stats (`mean/max/min/std/last/trend`)

Why:
- This captures sequence-level risk trajectory in a compact meta representation.
- It transforms row-level temporal signal into customer-level features that boosting handles well.

#### 6) v4 sequence embedding features
- bi-GRU denoising autoencoder embeddings
- optional delta-sequence channels

Why:
- Embeddings can preserve temporal shape information beyond handcrafted aggregates.
- Denoising objective can regularize noisy statement sequences.

### Why LightGBM Is the Core Learner
LightGBM is chosen as the primary model family because it is a strong fit for this dataset shape.

#### How LightGBM works (short technical view)
1. Gradient boosting builds trees sequentially to minimize residual error from previous trees.
2. LightGBM uses histogram-based split finding (bucketized values), making training faster and memory efficient.
3. Leaf-wise tree growth (with depth/leaf controls) often improves tabular performance.
4. Native handling of missing values and categorical representations is practical for mixed credit data.

#### Why this is a strong match for AMEX-like data
- High-dimensional, mixed-type tabular signals
- Nonlinear feature interactions are important
- Missingness and sparse patterns are common
- Strong ranking performance with relatively simple deployment path

### Why Not Use Only Deep Sequence Models
Pure deep sequence models can work, but for this project they are not used as the sole solution because:
1. Tabular GBMs remain very competitive on structured credit datasets.
2. Deep models are more sensitive to optimization/runtime and infrastructure constraints.
3. Portfolio and production communication both benefit from transparent baselines + incremental complexity.

Hence the project uses deep sequence modeling as **complementary signal** (`v4`) rather than replacing robust GBM pipelines.

### Model Versions and Their Role
#### v1 (Baseline)
- Single strong LightGBM baseline on engineered customer features.

Purpose:
- Establish a reliable benchmark.
- Validate data pipeline and leakage controls.

#### v2 (Advanced Ensemble)
- Production-focused advanced ensemble path.

Purpose:
- Improve ranking quality over baseline while staying operationally stable.

#### v3 (Two-stage Meta Model)
- Statement-level signal extraction + customer-level booster.

Purpose:
- Capture additional temporal signal not fully represented by simple aggregates.

#### v4 (Sequence Embedding + LightGBM)
- Bi-GRU denoising embedding model merged into tabular booster.

Purpose:
- Add representation-learning diversity to ensemble family.

#### Final blend (v2 + v3 + v4)
- Rank-based weighted blend.

Why rank blending:
- AMEX objective is ranking-sensitive.
- Rank blending reduces calibration mismatch across heterogeneous models.

### Evaluation and Diagnostics
#### Core metrics
- ROC-AUC
- PR-AUC
- Decile lift
- AMEX-oriented ranking behavior (via OOF/leaderboard outcomes)

#### Why decile lift is emphasized
Decile lift is operationally meaningful:
- “How much bad-rate concentration do we get in top predicted buckets?”
- This maps directly to prioritized interventions in risk operations.

### Explainability Strategy
#### Global importance
- Booster feature importances to identify dominant drivers.

#### SHAP
- Local/global directional explanations for high-impact features.

#### Segment summaries
- Risk decile behavior profiles (trend/volatility/missingness slices).

Why this matters:
- Converts model outputs into actionable risk narratives instead of just scores.

### Leakage and Validation Guardrails
1. Time-aware split by customer timeline.
2. Group-aware folds where applicable.
3. Cache health checks for full-data runs (prevents accidental tiny-sample artifacts).
4. Consistent schema alignment between train/test in sequence pipelines.

### Infrastructure and Runtime Engineering
Because Colab can crash or reset:
1. Chunked preprocessing/training to bound memory.
2. Drive-backed artifacts for restart resilience.
3. Safe fallbacks for missing in-memory objects.
4. CPU/GPU tradeoff awareness (data prep often CPU-bound even on A100).

### Results
Observed progression (from project runs):
- `baseline_submission.csv`: Private `0.77292`, Public `0.76131`
- `advanced_submission_v3.csv` (pre-renaming run): Private `0.79270`, Public `0.78229`
- `blend_v234_submission.csv`: Private `0.79596`, Public `0.78592` (best reported)

Interpretation:
- Most lift came from ensemble diversification + rank blending, not from a single model swap.

### Key Lessons Learned
1. Strong tabular feature engineering + robust validation still dominates many credit tasks.
2. Sequence models add value when used as complementary signals, not as isolated replacements.
3. Restart-safe, chunked pipelines are critical for practical iteration at AMEX scale.
4. Blending independent model families is often the final step that unlocks additional ranking gains.

### Portfolio Talking Points
1. “I designed a leakage-aware temporal pipeline with chunked feature storage for full-scale training in constrained environments.”
2. “I used behavior-centric features (recency, trend, volatility, missingness) before moving to deeper sequence representations.”
3. “I implemented multi-version model progression and unified them with rank-based ensemble blending for ranking objective gains.”
4. “I treated explainability and segment interpretation as first-class outputs, not afterthoughts.”

### Known Limitations and Next Steps
1. Add OOF blend-weight optimization directly against AMEX metric.
2. Add DART branch for extra ensemble diversity.
3. Add calibrated risk-threshold policy simulation for business deployment scenarios.
4. Modularize final notebook logic into scripts/pipelines for production handoff.

### References
- AMEX Competition: [Kaggle AMEX Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
- LightGBM docs: [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- LightGBM parameters (`boosting_type`, etc.): [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- SHAP docs: [SHAP Documentation](https://shap.readthedocs.io/)
- CatBoost docs: [CatBoost Documentation](https://catboost.ai/docs/)
- PyTorch docs: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Ensemble-direction inspiration: [LGBM AMEX Solution Repo](https://github.com/Nicolas-Bolouri/LGBM-Solution-Amex-Kaggle)
