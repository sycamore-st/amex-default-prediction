# Project Summary: AMEX Default Prediction

## Problem Framing

American Express needs to identify customers with elevated default risk as early as possible from longitudinal behavior data. This project frames the task as a supervised binary classification problem at the **customer level**, where each customer has a sequence of monthly statements.

Core objective: rank customers by likelihood of default to support better risk policy decisions (line management, proactive collections, and portfolio monitoring).

## Feature Strategy

Feature engineering follows a behavior-modeling lens with leakage prevention.

1. Sequence aggregation per customer:
- central tendency: `mean`, `last`
- volatility proxy: `std`
- bounds: `min`, `max`

2. Temporal behavior:
- recency (`days since last statement`)
- history length (`span between first and last statement`)
- activity (`number of statements`)
- trend proxy (`slope` for selected numeric variables)

3. Data quality / information patterns:
- feature-level missingness rate per customer

4. Categorical behavior:
- latest observed category value (`*_last_cat`)
- within-history diversity (`*_nunique_cat`)

This captures recency, trend, volatility, utilization-like changes, and missingness dynamics requested for interview depth.

## Validation and Leakage Control

- Split customers by **latest statement date** (time-aware holdout).
- Train uses earlier customers; validation uses later customers.
- Prevents information bleed from future behavior into training.

## Model Comparison

Baseline model stack in notebook:

1. Logistic Regression
- Pros: transparent coefficients, straightforward story for interviews
- Cons: limited non-linear interaction capture

2. Gradient Boosting (LightGBM preferred, XGBoost fallback)
- Pros: strong tabular performance, captures interactions/nonlinearity
- Cons: additional complexity and tuning overhead

Metrics reported:
- ROC-AUC
- PR-AUC
- Decile lift (ranking quality in top-risk segments)

## Explainability

- Global importance from boosting model.
- SHAP summary plot for dominant risk drivers.
- Segment-level summaries by predicted risk decile to connect model behavior to business actionability.

## Business Recommendations

1. Use decile-based risk tiers for treatment policy.
2. Focus operational interventions on top predicted deciles (higher expected bad rate concentration).
3. Track key behavior drivers over time (recency shifts, volatility spikes, missingness pattern changes).
4. Calibrate decision thresholds by business constraints (cost of intervention vs. expected loss reduction).

## Interview Readiness Talking Points

- Why time-aware validation is mandatory in temporal credit datasets.
- How behavior features (trend/volatility/missingness) outperform naive static snapshots.
- Why linear and boosted baselines are both needed (interpretability + performance).
- How ranking metrics (lift by decile) map directly to risk operations.
