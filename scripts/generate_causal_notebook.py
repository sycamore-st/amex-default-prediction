from pathlib import Path
from textwrap import dedent

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(
    """
    # Causal Methods with Synthetic Data

    This notebook is for interview prep. It uses small, readable Python examples to build intuition for four common causal inference situations:

    1. Interrupted Time Series (ITS)
    2. Difference-in-Differences (DiD)
    3. Propensity Score Weighting / Matching
    4. Instrumental Variables (IV)

    Everything below uses synthetic data only, sets a random seed, and is designed to run top to bottom.
    """
)

md(
    """
    ## Environment and imports

    We keep the stack simple: `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, and a small amount of `scikit-learn` for nearest-neighbor matching.
    """
)

code(
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from IPython.display import display
    from sklearn.neighbors import NearestNeighbors

    sns.set_theme(style="whitegrid", context="talk")
    np.random.seed(42)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    """
)

md(
    """
    ## Helper functions

    These stay intentionally lightweight so the notebook remains educational rather than overly abstract.
    """
)

code(
    """
    def smd(df, treatment_col, covariate, weights=None):
        treated = df[treatment_col] == 1
        control = df[treatment_col] == 0

        if weights is None:
            mt = df.loc[treated, covariate].mean()
            mc = df.loc[control, covariate].mean()
            vt = df.loc[treated, covariate].var(ddof=1)
            vc = df.loc[control, covariate].var(ddof=1)
        else:
            wt = np.asarray(weights[treated])
            wc = np.asarray(weights[control])
            xt = np.asarray(df.loc[treated, covariate])
            xc = np.asarray(df.loc[control, covariate])
            mt = np.average(xt, weights=wt)
            mc = np.average(xc, weights=wc)
            vt = np.average((xt - mt) ** 2, weights=wt)
            vc = np.average((xc - mc) ** 2, weights=wc)

        pooled_sd = np.sqrt((vt + vc) / 2)
        return (mt - mc) / pooled_sd


    def weighted_mean(values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        return np.sum(values * weights) / np.sum(weights)


    def coef_table(model):
        return model.summary2().tables[1].round(3)
    """
)

md(
    """
    # 1. Interrupted Time Series (ITS)

    ## A. Intuition

    **Business scenario:** a global risk rule launches on one date for everyone, so there is no untreated comparison group.

    **Why simpler methods fail:** a naive before-vs-after average compares two periods that may differ because of trend, seasonality, or unrelated shocks. ITS instead asks: *relative to the pre-existing trajectory, what changed at the intervention date?*
    """
)

md(
    """
    ## B. Data generation

    We simulate one monthly series with:

    - a baseline level
    - a linear pre-trend
    - annual seasonality
    - random noise
    - an intervention at month 72
    - both an immediate level shift and a slope change after the intervention

    Data-generating process:

    $$
    y_t = \\beta_0 + \\beta_1 t + \\beta_2 \\sin(2\\pi t/12) + \\beta_3 \\cos(2\\pi t/12) + \\tau_0 D_t + \\tau_1 (t - T_0) D_t + \\varepsilon_t
    $$

    where $D_t$ is 1 after the launch month and 0 before.
    """
)

code(
    """
    n_months = 120
    intervention_month = 72

    its = pd.DataFrame({"month": np.arange(n_months)})
    its["post"] = (its["month"] >= intervention_month).astype(int)
    its["time_after"] = np.clip(its["month"] - intervention_month, 0, None)
    its["sin12"] = np.sin(2 * np.pi * its["month"] / 12)
    its["cos12"] = np.cos(2 * np.pi * its["month"] / 12)

    noise = np.random.normal(0, 1.6, n_months)
    its["y"] = (
        55
        + 0.28 * its["month"]
        + 3.5 * its["sin12"]
        + 1.7 * its["cos12"]
        + 6.0 * its["post"]
        + 0.22 * its["time_after"]
        + noise
    )

    its.head()
    """
)

md(
    """
    ## C. Naive approach

    A naive estimate is just the average post-period outcome minus the average pre-period outcome.

    That is wrong here because post-period months are later in time and sit on a different seasonal mix.
    """
)

code(
    """
    its_naive = its.loc[its["post"] == 1, "y"].mean() - its.loc[its["post"] == 0, "y"].mean()
    print(f"Naive post-minus-pre difference: {its_naive:.3f}")
    """
)

md(
    """
    ## D. Correct approach

    ITS uses segmented regression.

    Core specification:

    $$
    y_t = \\beta_0 + \\beta_1 t + \\tau_0 D_t + \\tau_1 (t - T_0) D_t + f(\\text{seasonality}) + \\varepsilon_t
    $$

    - `month` captures the baseline slope before launch.
    - `post` is the immediate level change at launch.
    - `time_after` is the change in slope after launch.
    - `sin12` and `cos12` absorb recurring seasonality.
    """
)

code(
    """
    its_model = smf.ols("y ~ month + post + time_after + sin12 + cos12", data=its).fit()
    display(coef_table(its_model))

    its["fitted"] = its_model.predict(its)
    its_counterfactual = its.copy()
    its_counterfactual["post"] = 0
    its_counterfactual["time_after"] = 0
    its["counterfactual"] = its_model.predict(its_counterfactual)

    level_change = its_model.params["post"]
    slope_change = its_model.params["time_after"]
    print(f"Estimated immediate level change: {level_change:.3f}")
    print(f"Estimated slope change per month: {slope_change:.3f}")
    print()
    print(its_model.summary())
    """
)

md(
    """
    ## E. Visualization

    The key visual is the observed post-period series versus the counterfactual path implied by the pre-launch trend and seasonality.
    """
)

code(
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(its["month"], its["y"], label="Observed outcome", color="tab:blue")
    ax.plot(its["month"], its["fitted"], label="Segmented regression fit", color="tab:green", linewidth=2)
    ax.plot(its["month"], its["counterfactual"], label="No-intervention counterfactual", color="tab:red", linestyle="--", linewidth=2)
    ax.axvline(intervention_month, color="black", linestyle=":", linewidth=2, label="Launch month")
    ax.set_title("Interrupted Time Series: observed series and counterfactual")
    ax.set_xlabel("Month")
    ax.set_ylabel("Risk metric")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## F. Assumptions and failure modes

    **Identifying assumptions**

    - In the absence of treatment, the pre-period trend would have continued.
    - No other simultaneous shock changed the outcome at the same launch date.
    - The functional form for trend and seasonality is good enough.

    **Failure case:** now we create a second intervention that starts on the exact same month. ITS cannot separate the policy effect from the confounding shock.
    """
)

code(
    """
    its_failure = its[["month", "post", "time_after", "sin12", "cos12"]].copy()
    confounder_shock = 4.5 * its_failure["post"]
    noise_failure = np.random.normal(0, 1.6, len(its_failure))
    its_failure["y"] = (
        55
        + 0.28 * its_failure["month"]
        + 3.5 * its_failure["sin12"]
        + 1.7 * its_failure["cos12"]
        + 6.0 * its_failure["post"]
        + 0.22 * its_failure["time_after"]
        + confounder_shock
        + noise_failure
    )

    fail_model = smf.ols("y ~ month + post + time_after + sin12 + cos12", data=its_failure).fit()
    wrong_model = smf.ols("y ~ month + post + time_after", data=its).fit()

    print(f"Level change estimate with simultaneous hidden shock: {fail_model.params['post']:.3f}")
    print(f"Level change estimate if you ignore seasonality entirely: {wrong_model.params['post']:.3f}")
    """
)

code(
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(its_failure["month"], its_failure["y"], color="tab:orange")
    axes[0].axvline(intervention_month, color="black", linestyle=":")
    axes[0].set_title("Failure case: simultaneous hidden shock")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Outcome")

    axes[1].plot(its["month"], its["y"], label="Observed", color="tab:blue")
    axes[1].plot(its["month"], wrong_model.predict(its), label="Fit without seasonality", color="tab:red")
    axes[1].axvline(intervention_month, color="black", linestyle=":")
    axes[1].set_title("Failure case: misspecified seasonality")
    axes[1].set_xlabel("Month")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## G. Interview takeaway

    > **How I would explain this in an interview:** If a policy launches globally and there is no clean control group, I would use interrupted time series. I would model the pre-period trajectory, estimate any level and slope break at launch, and spend most of my credibility discussing whether another simultaneous shock could explain the break.
    """
)

md(
    """
    # 2. Difference-in-Differences (DiD)

    ## A. Intuition

    **Business scenario:** a pricing or policy change rolls out by region in phases.

    **Why simpler methods fail:** treated regions may have different baseline levels from untreated ones, and every region also experiences common macro shocks. DiD removes time-invariant region differences and common time shocks, then attributes the remaining break to treatment under parallel trends.
    """
)

md(
    """
    ## B. Data generation

    We simulate 8 regions over 24 months.

    - regions have different baseline levels
    - all regions share common month shocks
    - 4 regions adopt in different months
    - treatment adds a constant effect after adoption

    The identifying design is staggered rollout: some regions are untreated while others have already switched.
    """
)

code(
    """
    rng = np.random.default_rng(123)
    regions = [f"Region_{i}" for i in range(8)]
    months = np.arange(24)
    treat_start = {
        "Region_0": 12,
        "Region_1": 14,
        "Region_2": 16,
        "Region_3": 18,
        "Region_4": 999,
        "Region_5": 999,
        "Region_6": 999,
        "Region_7": 999,
    }

    region_fe = {r: rng.normal(0, 3) for r in regions}
    time_fe = {m: 0.7 * np.sin(m / 2.7) + 0.12 * m for m in months}

    did_rows = []
    for region in regions:
        for month in months:
            treated = int(month >= treat_start[region])
            effect = 3.0 * treated
            y = 40 + region_fe[region] + time_fe[month] + effect + rng.normal(0, 1.1)
            did_rows.append({
                "region": region,
                "month": month,
                "treated": treated,
                "ever_treated": int(treat_start[region] < 999),
                "adopt_month": treat_start[region],
                "y": y,
            })

    did = pd.DataFrame(did_rows)
    did.head()
    """
)

md(
    """
    ## C. Naive approach

    A naive comparison would take the mean outcome in treated observations minus the mean outcome in untreated observations.

    That bundles together the treatment effect, region baseline differences, and the fact that treated observations tend to occur later in calendar time.
    """
)

code(
    """
    did_naive = did.loc[did["treated"] == 1, "y"].mean() - did.loc[did["treated"] == 0, "y"].mean()
    print(f"Naive treated-minus-untreated difference: {did_naive:.3f}")
    """
)

md(
    """
    ## D. Correct approach

    ### 2x2 intuition

    Start with one early-treated group and one never-treated group, collapse time into pre vs post, and compute:

    $$
    \\text{DiD} = (\\bar{Y}_{T,post} - \\bar{Y}_{T,pre}) - (\\bar{Y}_{C,post} - \\bar{Y}_{C,pre})
    $$

    ### Fixed-effects style regression

    With multiple regions and months, we can estimate:

    $$
    y_{rt} = \\alpha_r + \\gamma_t + \\tau \\cdot \\text{treated}_{rt} + \\varepsilon_{rt}
    $$

    - `C(region)` creates region fixed effects.
    - `C(month)` creates month fixed effects.
    - `treated` is the DiD treatment indicator.
    """
)

code(
    """
    two_by_two = did[did["region"].isin(["Region_0", "Region_4"])].copy()
    two_by_two["post_period"] = (two_by_two["month"] >= 12).astype(int)

    means = two_by_two.groupby(["region", "post_period"])["y"].mean().unstack()
    did_2x2 = (means.loc["Region_0", 1] - means.loc["Region_0", 0]) - (means.loc["Region_4", 1] - means.loc["Region_4", 0])
    print("2x2 means")
    display(means)
    print(f"Simple 2x2 DiD estimate: {did_2x2:.3f}")
    """
)

code(
    """
    did_model = smf.ols("y ~ treated + C(region) + C(month)", data=did).fit()
    display(coef_table(did_model).loc[["treated"]])
    print(did_model.summary())
    did_tau = did_model.params["treated"]
    """
)

md(
    """
    ### Event-study style check

    An event-study plot helps assess parallel trends by plotting coefficients for leads and lags relative to adoption. Pre-treatment coefficients should be close to zero if the design is behaving well.
    """
)

code(
    """
    event = did[did["ever_treated"] == 1].copy()
    event["event_time"] = event["month"] - event["adopt_month"]

    window = list(range(-5, 6))
    window.remove(-1)

    event_terms = []
    for k in window:
        if k < 0:
            name = f"event_m{abs(k)}"
        else:
            name = f"event_p{k}"
        event[name] = (event["event_time"] == k).astype(int)
        event_terms.append((k, name))

    event_formula = "y ~ " + " + ".join(name for _, name in event_terms) + " + C(region) + C(month)"
    event_model = smf.ols(event_formula, data=event).fit()
    event_coefs = pd.Series({k: event_model.params[name] for k, name in event_terms}).sort_index()
    display(coef_table(event_model).loc[[name for _, name in event_terms]])
    """
)

md(
    """
    ## E. Visualization

    We first plot average trends for ever-treated vs never-treated regions, then the event-study coefficients.
    """
)

code(
    """
    avg_trends = did.groupby(["month", "ever_treated"])["y"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=avg_trends, x="month", y="y", hue="ever_treated", marker="o", ax=axes[0])
    axes[0].set_title("Average trends: ever-treated vs never-treated regions")
    axes[0].set_ylabel("Outcome")
    axes[0].legend(title="Ever treated", labels=["Never", "Ever"])

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(-1, color="black", linestyle=":", linewidth=1)
    axes[1].plot(event_coefs.index, event_coefs.values, marker="o")
    axes[1].set_title("Event-study style coefficients")
    axes[1].set_xlabel("Months relative to adoption")
    axes[1].set_ylabel("Coefficient vs month -1")

    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## F. Assumptions and failure modes

    **Identifying assumptions**

    - In the absence of treatment, treated and control regions would have followed parallel trends.
    - No spillovers across regions.
    - Treatment timing is not itself driven by imminent outcome jumps in a way the model does not capture.

    **Failure case:** treated regions now have a steeper upward trend even before adoption, violating parallel trends.
    """
)

code(
    """
    did_fail_rows = []
    for region in regions:
        pretrend_bonus = 0.18 if treat_start[region] < 999 else 0.0
        for month in months:
            treated = int(month >= treat_start[region])
            effect = 3.0 * treated
            y = (
                40
                + region_fe[region]
                + time_fe[month]
                + pretrend_bonus * month
                + effect
                + rng.normal(0, 1.1)
            )
            did_fail_rows.append({
                "region": region,
                "month": month,
                "treated": treated,
                "ever_treated": int(treat_start[region] < 999),
                "adopt_month": treat_start[region],
                "y": y,
            })

    did_fail = pd.DataFrame(did_fail_rows)
    fail_model = smf.ols("y ~ treated + C(region) + C(month)", data=did_fail).fit()
    print(f"DiD estimate when parallel trends is violated: {fail_model.params['treated']:.3f}")
    """
)

code(
    """
    fail_avg = did_fail.groupby(["month", "ever_treated"])["y"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=fail_avg, x="month", y="y", hue="ever_treated", marker="o")
    plt.title("Failure case: treated regions already trend upward before treatment")
    plt.ylabel("Outcome")
    plt.legend(title="Ever treated", labels=["Never", "Ever"])
    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## G. Interview takeaway

    > **How I would explain this in an interview:** If rollout happens by geography or cohort over time, I would start with DiD because it differences out stable group differences and common shocks. The real question is whether I believe parallel trends, so I would always show pre-trends or an event study rather than just one coefficient.
    """
)

md(
    """
    # 3. Propensity Score Weighting / Matching

    ## A. Intuition

    **Business scenario:** a new payout method is mostly adopted by high-value creators, so treatment is not randomized.

    **Why simpler methods fail:** treated creators look systematically different from untreated creators even before treatment. A raw treated-vs-control comparison confuses treatment effect with selection on observed covariates.
    """
)

md(
    """
    ## B. Data generation

    We simulate creator-level data where adoption depends on observed creator quality, audience size, tenure, and prior engagement.

    The outcome also depends on those same variables, so confounding is built into the synthetic design.
    """
)

code(
    """
    n = 4000
    quality = np.random.normal(0, 1, n)
    log_followers = np.random.normal(10.5, 0.8, n)
    tenure_months = np.random.gamma(shape=4, scale=6, size=n)
    prior_ctr = np.clip(np.random.normal(0.08, 0.02, n), 0.01, 0.20)

    adoption_score = (
        -6.0
        + 1.3 * quality
        + 0.35 * log_followers
        + 0.025 * tenure_months
        + 10.0 * prior_ctr
    )
    adoption_prob = 1 / (1 + np.exp(-adoption_score))
    treatment = np.random.binomial(1, adoption_prob)

    true_effect = 2.5
    outcome = (
        15
        + true_effect * treatment
        + 2.0 * quality
        + 0.7 * log_followers
        + 0.03 * tenure_months
        + 20.0 * prior_ctr
        + np.random.normal(0, 1.5, n)
    )

    ps = pd.DataFrame({
        "treatment": treatment,
        "quality": quality,
        "log_followers": log_followers,
        "tenure_months": tenure_months,
        "prior_ctr": prior_ctr,
        "outcome": outcome,
    })
    ps.head()
    """
)

md(
    """
    ## C. Naive approach

    A naive estimate compares the average outcome for adopters and non-adopters.

    That is biased because adopters have higher quality and larger audiences even before treatment.
    """
)

code(
    """
    ps_naive = ps.loc[ps["treatment"] == 1, "outcome"].mean() - ps.loc[ps["treatment"] == 0, "outcome"].mean()
    print(f"True treatment effect built into the data: {true_effect:.3f}")
    print(f"Naive treated-minus-control difference: {ps_naive:.3f}")
    """
)

md(
    """
    ## D. Correct approach

    We estimate the propensity score:

    $$
    e(X_i) = P(T_i = 1 \\mid X_i)
    $$

    Then use inverse probability weighting (IPW) to reweight the sample so treated and control creators look more comparable on observed covariates.

    ATE weights:

    $$
    w_i = \\frac{T_i}{e(X_i)} + \\frac{1-T_i}{1-e(X_i)}
    $$

    We also show nearest-neighbor matching as a simple alternative.
    """
)

code(
    """
    covariates = ["quality", "log_followers", "tenure_months", "prior_ctr"]
    X_ps = sm.add_constant(ps[covariates], has_constant="add")
    ps_model = sm.Logit(ps["treatment"], X_ps).fit(disp=False)
    display(coef_table(ps_model))

    ps["propensity_score"] = ps_model.predict(X_ps).clip(0.02, 0.98)
    ps["ate_weight"] = np.where(
        ps["treatment"] == 1,
        1 / ps["propensity_score"],
        1 / (1 - ps["propensity_score"]),
    )

    weighted_effect = (
        weighted_mean(ps.loc[ps["treatment"] == 1, "outcome"], ps.loc[ps["treatment"] == 1, "ate_weight"])
        - weighted_mean(ps.loc[ps["treatment"] == 0, "outcome"], ps.loc[ps["treatment"] == 0, "ate_weight"])
    )

    treated = ps[ps["treatment"] == 1].copy()
    control = ps[ps["treatment"] == 0].copy()
    matcher = NearestNeighbors(n_neighbors=1)
    matcher.fit(control[["propensity_score"]])
    distances, indices = matcher.kneighbors(treated[["propensity_score"]])
    matched_controls = control.iloc[indices.flatten()].copy()
    matching_effect = (treated["outcome"].reset_index(drop=True) - matched_controls["outcome"].reset_index(drop=True)).mean()

    print(f"IPW estimate: {weighted_effect:.3f}")
    print(f"Nearest-neighbor matching estimate: {matching_effect:.3f}")
    print()
    print(ps_model.summary())
    """
)

md(
    """
    Balance checks matter as much as the effect estimate. If weighting does not improve balance, the design is still not credible.
    """
)

code(
    """
    balance = []
    for cov in covariates:
        balance.append({
            "covariate": cov,
            "before": smd(ps, "treatment", cov),
            "after_ipw": smd(ps, "treatment", cov, weights=ps["ate_weight"]),
        })
    balance_df = pd.DataFrame(balance)
    balance_df
    """
)

md(
    """
    ## E. Visualization

    Two core plots are especially useful:

    - propensity score overlap between treated and control units
    - covariate balance before and after weighting
    """
)

code(
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(data=ps, x="propensity_score", hue="treatment", stat="density", common_norm=False, bins=30, alpha=0.45, ax=axes[0])
    axes[0].set_title("Propensity score overlap")
    axes[0].set_xlabel("Estimated propensity score")

    plot_balance = balance_df.melt(id_vars="covariate", value_vars=["before", "after_ipw"], var_name="stage", value_name="smd")
    sns.barplot(data=plot_balance, x="smd", y="covariate", hue="stage", ax=axes[1])
    axes[1].axvline(0.1, color="black", linestyle=":", linewidth=1)
    axes[1].axvline(-0.1, color="black", linestyle=":", linewidth=1)
    axes[1].set_title("Standardized mean differences before and after IPW")
    axes[1].set_xlabel("Standardized mean difference")

    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## F. Assumptions and failure modes

    **Identifying assumptions**

    - No unobserved confounders after conditioning on included covariates.
    - Positivity / overlap: every covariate profile has some chance of treatment and control.
    - Propensity score model is good enough to balance the sample.

    **Failure case:** poor overlap. If treatment is almost deterministic for high-quality creators, weights explode and the estimate becomes unstable.
    """
)

code(
    """
    poor_overlap_score = -8.5 + 3.2 * quality + 0.55 * log_followers + 0.04 * tenure_months + 12.0 * prior_ctr
    poor_overlap_prob = 1 / (1 + np.exp(-poor_overlap_score))
    poor_overlap_t = np.random.binomial(1, poor_overlap_prob)
    poor_overlap_df = ps[covariates].copy()
    poor_overlap_df["treatment"] = poor_overlap_t
    poor_overlap_df["outcome"] = (
        15
        + true_effect * poor_overlap_t
        + 2.0 * quality
        + 0.7 * log_followers
        + 0.03 * tenure_months
        + 20.0 * prior_ctr
        + np.random.normal(0, 1.5, n)
    )

    poor_X = sm.add_constant(poor_overlap_df[covariates], has_constant="add")
    poor_model = sm.Logit(poor_overlap_df["treatment"], poor_X).fit(disp=False)
    poor_overlap_df["ps"] = poor_model.predict(poor_X).clip(0.001, 0.999)
    poor_overlap_df["w"] = np.where(poor_overlap_df["treatment"] == 1, 1 / poor_overlap_df["ps"], 1 / (1 - poor_overlap_df["ps"]))

    print(poor_overlap_df["w"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    """
)

code(
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data=poor_overlap_df, x="ps", hue="treatment", stat="density", common_norm=False, bins=30, alpha=0.45)
    plt.title("Failure case: poor overlap creates extreme IPW weights")
    plt.xlabel("Estimated propensity score")
    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## G. Interview takeaway

    > **How I would explain this in an interview:** If treatment adoption is clearly selective but I believe the confounders are observed, I would estimate a propensity score, check overlap, and show covariate balance before trusting the weighted or matched effect. I would explicitly say that this does not solve hidden confounding.
    """
)

md(
    """
    # 4. Instrumental Variables (IV)

    ## A. Intuition

    **Business scenario:** strong concern about unobserved confounding means even rich observational controls may be biased.

    **Why simpler methods fail:** if an unobserved factor affects both treatment and outcome, naive OLS still attributes part of that confounding pathway to treatment. IV looks for a source of variation that shifts treatment but is otherwise unrelated to the outcome.
    """
)

md(
    """
    ## B. Data generation

    We simulate:

    - an observed covariate `x`
    - an unobserved confounder `u`
    - an instrument `z` that shifts treatment strongly
    - treatment `d`
    - outcome `y`

    Data-generating process:

    $$
    d_i = \\pi_0 + \\pi_1 z_i + \\pi_2 x_i + \\pi_3 u_i + \\nu_i
    $$

    $$
    y_i = \\beta_0 + \\tau d_i + \\beta_1 x_i + \\beta_2 u_i + \\epsilon_i
    $$

    Because `u` is omitted from estimation but affects both `d` and `y`, OLS is biased.
    """
)

code(
    """
    n_iv = 5000
    x = np.random.normal(0, 1, n_iv)
    u = np.random.normal(0, 1, n_iv)
    z = np.random.binomial(1, 0.5, n_iv)

    d = 0.9 * z + 0.6 * x + 1.0 * u + np.random.normal(0, 1, n_iv)
    true_tau = 2.0
    y = true_tau * d + 0.8 * x + 1.4 * u + np.random.normal(0, 1, n_iv)

    iv = pd.DataFrame({"y": y, "d": d, "z": z, "x": x})
    iv.head()
    """
)

md(
    """
    ## C. Naive approach

    OLS estimates:

    $$
    y_i = \\alpha + \\tau_{OLS} d_i + \\gamma x_i + \\epsilon_i
    $$

    But $d$ is correlated with omitted $u$, so $\\tau_{OLS}$ is biased.
    """
)

code(
    """
    ols_model = smf.ols("y ~ d + x", data=iv).fit()
    display(coef_table(ols_model).loc[["d"]])
    print(f"True effect: {true_tau:.3f}")
    print(f"Naive OLS estimate: {ols_model.params['d']:.3f}")
    print()
    print(ols_model.summary())
    """
)

md(
    """
    ## D. Correct approach

    Two-stage least squares (2SLS):

    1. **First stage**

    $$
    d_i = \\alpha_1 + \\pi z_i + \\gamma x_i + \\nu_i
    $$

    2. **Second stage**

    $$
    y_i = \\alpha_2 + \\tau \\hat{d}_i + \\delta x_i + \\epsilon_i
    $$

    - `pi` measures instrument relevance.
    - `tau` is the IV estimate of the causal effect for compliers under the IV assumptions.
    """
)

code(
    """
    first_stage = smf.ols("d ~ z + x", data=iv).fit()
    iv["d_hat"] = first_stage.predict(iv)

    second_stage = smf.ols("y ~ d_hat + x", data=iv).fit()
    display(coef_table(first_stage).loc[["z"]])
    display(coef_table(second_stage).loc[["d_hat"]])
    print(first_stage.summary())
    print()
    print(second_stage.summary())

    iv_tau = second_stage.params["d_hat"]
    first_stage_pi = first_stage.params["z"]
    """
)

md(
    """
    ## E. Visualization

    We compare the first-stage relationship and then place the OLS and IV estimates side by side.
    """
)

code(
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    first_stage_plot = iv.groupby("z")["d"].mean().reset_index()
    sns.barplot(data=first_stage_plot, x="z", y="d", ax=axes[0])
    axes[0].set_title("First stage: instrument shifts treatment")
    axes[0].set_xlabel("Instrument z")
    axes[0].set_ylabel("Average treatment intensity")

    compare_df = pd.DataFrame({
        "Estimator": ["OLS", "IV / 2SLS", "True effect"],
        "Estimate": [ols_model.params["d"], iv_tau, true_tau],
    })
    sns.barplot(data=compare_df, x="Estimator", y="Estimate", ax=axes[1])
    axes[1].set_title("Naive OLS versus IV estimate")
    axes[1].set_ylabel("Estimated effect of treatment on outcome")

    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## F. Assumptions and failure modes

    **Identifying assumptions**

    - **Relevance:** the instrument changes treatment.
    - **Exclusion restriction:** the instrument affects the outcome only through treatment.
    - **Independence:** the instrument is as-good-as-random with respect to unobserved determinants of the outcome.

    **Failure case:** weak instrument. If $z$ barely moves treatment, IV becomes noisy and unstable.
    """
)

code(
    """
    z_weak = np.random.binomial(1, 0.5, n_iv)
    d_weak = 0.08 * z_weak + 0.6 * x + 1.0 * u + np.random.normal(0, 1, n_iv)
    y_weak = true_tau * d_weak + 0.8 * x + 1.4 * u + np.random.normal(0, 1, n_iv)

    weak_df = pd.DataFrame({"y": y_weak, "d": d_weak, "z": z_weak, "x": x})
    weak_first = smf.ols("d ~ z + x", data=weak_df).fit()
    weak_df["d_hat"] = weak_first.predict(weak_df)
    weak_second = smf.ols("y ~ d_hat + x", data=weak_df).fit()
    weak_ols = smf.ols("y ~ d + x", data=weak_df).fit()

    print(f"Weak first-stage coefficient: {weak_first.params['z']:.3f}")
    print(f"Weak-instrument OLS estimate: {weak_ols.params['d']:.3f}")
    print(f"Weak-instrument IV estimate: {weak_second.params['d_hat']:.3f}")
    """
)

code(
    """
    weak_plot = weak_df.groupby("z")["d"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=weak_plot, x="z", y="d")
    plt.title("Failure case: weak instrument barely moves treatment")
    plt.xlabel("Instrument z")
    plt.ylabel("Average treatment intensity")
    plt.tight_layout()
    plt.show()
    """
)

md(
    """
    ## G. Interview takeaway

    > **How I would explain this in an interview:** If I think even rich controls miss important confounding, I would look for an instrument that moves treatment but has no direct path to the outcome. Then I would defend relevance and exclusion much harder than the coefficient itself, because IV is only as credible as that identification story.
    """
)

md(
    """
    # Final Summary Table

    This is the compact interview-prep version of the notebook.
    """
)

code(
    """
    summary = pd.DataFrame([
        {
            "Scenario": "Interrupted Time Series",
            "Naive method that fails": "Simple pre/post mean comparison",
            "Recommended method": "Segmented regression / ITS",
            "Core identifying assumption": "Pre-trend would have continued absent treatment",
            "Main diagnostic": "Plot observed post-period against counterfactual pre-trend",
            "Common failure mode": "Simultaneous shock at intervention date",
            "Interview one-liner": "Use ITS when rollout is global and no control group exists.",
        },
        {
            "Scenario": "Difference-in-Differences",
            "Naive method that fails": "Treated vs untreated mean comparison",
            "Recommended method": "DiD with region and time fixed effects",
            "Core identifying assumption": "Parallel trends",
            "Main diagnostic": "Pre-trend checks / event-study leads",
            "Common failure mode": "Treated units already trending differently",
            "Interview one-liner": "Use DiD for phased rollout when untreated groups provide a counterfactual trend.",
        },
        {
            "Scenario": "Propensity Score Weighting / Matching",
            "Naive method that fails": "Raw treated vs control outcome difference",
            "Recommended method": "IPW and/or matching on propensity score",
            "Core identifying assumption": "No unobserved confounding given covariates + overlap",
            "Main diagnostic": "Covariate balance and propensity overlap",
            "Common failure mode": "Poor overlap and extreme weights",
            "Interview one-liner": "Use propensity methods when selection is observed and you can defend ignorability.",
        },
        {
            "Scenario": "Instrumental Variables",
            "Naive method that fails": "OLS with confounded treatment",
            "Recommended method": "2SLS / IV",
            "Core identifying assumption": "Relevant and valid instrument with exclusion restriction",
            "Main diagnostic": "Strong first stage",
            "Common failure mode": "Weak or invalid instrument",
            "Interview one-liner": "Use IV when unobserved confounding is the main threat and a credible instrument exists.",
        },
    ])

    summary
    """
)

md(
    """
    ## Closing note

    For interviews, the strongest answer is usually not the most advanced estimator. It is the method that matches the rollout design, the main source of bias, and the diagnostics you would use to challenge your own estimate.
    """
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}

output_path = Path("notebooks/causal_methods_synthetic_demo.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output_path)
print(f"Wrote {output_path}")
