"""
model/evaluate.py

Diagnostics on the fitted regression:
1. R² and Adjusted R²
2. VIF for continuous variables (flag VIF > 10)
3. Top 10 most undervalued players
4. Top 10 most overvalued players
5. Coefficient table with significance flags

Run: python model/evaluate.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "results.csv")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "model_summary.txt")

DEPENDENT_VAR = "log_market_value"

INDEPENDENT_VARS = [
    "age", "age_squared", "minutes_played", "goals_per_90", "assists_per_90",
    "contract_months_remaining", "team_league_position",
    "is_striker", "is_winger", "is_attacking_mid", "is_central_mid",
    "is_fullback", "is_goalkeeper",
    "is_brazilian", "is_french", "is_english", "is_spanish", "is_german",
    "is_argentinian", "is_portuguese", "is_african", "is_asian",
    "is_south_american_other",
]

CONTINUOUS_VARS = [
    "age", "age_squared", "minutes_played",
    "goals_per_90", "assists_per_90",
    "contract_months_remaining", "team_league_position",
]


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for each predictor column in X (which must include no constant)."""
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception as e:
            vif = float("nan")
            print(f"  VIF error for {col}: {e}")
        vif_data.append({"variable": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    print_section("Model Evaluation & Diagnostics")

    # --- Load data and re-fit model ---
    print(f"\nLoading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8")

    missing = [v for v in [DEPENDENT_VAR] + INDEPENDENT_VARS if v not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    y = df[DEPENDENT_VAR].astype(float)
    X = df[INDEPENDENT_VARS].astype(float)

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    X_const = sm.add_constant(X)
    results = sm.OLS(y, X_const).fit()

    # ------------------------------------------------------------------
    # 1. R² and Adjusted R²
    # ------------------------------------------------------------------
    print_section("1. Model Fit Statistics")
    print(f"  R²:                   {results.rsquared:.4f}")
    print(f"  Adjusted R²:          {results.rsquared_adj:.4f}")
    print(f"  F-statistic:          {results.fvalue:.2f}")
    print(f"  F p-value:            {results.f_pvalue:.4e}")
    print(f"  Observations (N):     {int(results.nobs)}")
    print(f"  AIC:                  {results.aic:.2f}")
    print(f"  BIC:                  {results.bic:.2f}")

    # ------------------------------------------------------------------
    # 2. VIF for continuous variables
    # ------------------------------------------------------------------
    print_section("2. Variance Inflation Factors (continuous predictors)")
    cont_cols = [c for c in CONTINUOUS_VARS if c in X.columns]
    X_cont = X[cont_cols]
    vif_df = compute_vif(X_cont)

    flag_col = []
    for _, row in vif_df.iterrows():
        flag = " *** MULTICOLLINEARITY WARNING" if row["VIF"] > 10 else ""
        flag_col.append(flag)
    vif_df["flag"] = flag_col

    print(vif_df.to_string(index=False))

    high_vif = vif_df[vif_df["VIF"] > 10]
    if high_vif.empty:
        print("\n  No multicollinearity issues detected (all VIF <= 10).")
    else:
        print(f"\n  *** {len(high_vif)} variables have VIF > 10 — consider removing or transforming them.")

    # ------------------------------------------------------------------
    # 3 & 4. Top 10 undervalued and overvalued players
    # ------------------------------------------------------------------
    print(f"\nLoading results from {RESULTS_PATH}...")
    results_df = pd.read_csv(RESULTS_PATH, encoding="utf-8")

    results_df = results_df.sort_values("residual", ascending=False).reset_index(drop=True)

    display_cols = ["player_name", "club", "market_value_eur",
                    "predicted_market_value_eur", "residual"]
    available_display = [c for c in display_cols if c in results_df.columns]

    print_section("3. Top 10 Most UNDERVALUED Players (highest positive residual)")
    top_under = results_df.head(10)[available_display].copy()
    top_under["market_value_eur"] = top_under["market_value_eur"].apply(lambda x: f"€{x:,.0f}")
    top_under["predicted_market_value_eur"] = top_under["predicted_market_value_eur"].apply(lambda x: f"€{x:,.0f}")
    top_under["residual"] = top_under["residual"].round(4)
    print(top_under.to_string(index=False))

    print_section("4. Top 10 Most OVERVALUED Players (highest negative residual)")
    top_over = results_df.tail(10).sort_values("residual")[available_display].copy()
    top_over["market_value_eur"] = top_over["market_value_eur"].apply(lambda x: f"€{x:,.0f}")
    top_over["predicted_market_value_eur"] = top_over["predicted_market_value_eur"].apply(lambda x: f"€{x:,.0f}")
    top_over["residual"] = top_over["residual"].round(4)
    print(top_over.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Coefficient table
    # ------------------------------------------------------------------
    print_section("5. Coefficient Table")

    coef_df = pd.DataFrame({
        "variable": results.params.index,
        "coefficient": results.params.values,
        "std_err": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values,
    })

    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.10:
            return "."
        return ""

    coef_df["significant"] = coef_df["p_value"].apply(significance_stars)
    coef_df["coefficient"] = coef_df["coefficient"].round(4)
    coef_df["p_value"] = coef_df["p_value"].round(4)

    print(coef_df[["variable", "coefficient", "p_value", "significant"]].to_string(index=False))
    print("\nSignificance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

    print(f"\n{'=' * 60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
