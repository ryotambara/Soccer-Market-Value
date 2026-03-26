"""
model/regression.py

OLS regression using statsmodels.
Dependent variable: log_market_value
Saves model summary, residuals, and full results CSV.

Run: python model/regression.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_PL_2526  = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26")
FEATURES_PATH  = os.path.join(_PROC_PL_2526, "features.csv")
RESULTS_PATH   = os.path.join(_PROC_PL_2526, "results.csv")
SUMMARY_PATH   = os.path.join(_PROC_PL_2526, "model_summary.txt")
COEF_PATH      = os.path.join(_PROC_PL_2526, "model_coefficients.json")
AGE_MEAN_PATH  = os.path.join(_PROC_PL_2526, "age_mean.json")

DEPENDENT_VAR = "log_market_value"

_WS_STATS = [
    "tackles", "interceptions", "clearances", "blocks", "aerials_won",
    "key_passes", "dribbles", "crosses", "long_balls", "through_balls",
    "avg_passes", "shots", "xg", "xg_p90", "fouled",
]
_POSITIONS = ["striker", "winger", "attmid", "cm", "cdm", "fullback", "cb", "gk"]

_AGE_POSITIONS = ["striker", "winger", "attmid", "cm", "cdm", "fullback", "cb", "gk"]

INDEPENDENT_VARS = [
    "minutes_played",
    # Team tier dummies (baseline = mid table, positions 7-14)
    "is_top4",
    "is_top6",
    "is_bottom6",
    # Historic Big 6 prestige dummy (independent of current league position)
    "is_historic_top6",
    # Newly promoted clubs
    "is_promoted",
    # Global WhoScored stats
    "rating",
    "xg_diff",
    "pass_success_pct",
    "fouls_per_game",
    "dribbled_past_per_game",
    "yellow_cards",
    # Position dummies (baseline = centre_back)
    "is_striker",
    "is_winger",
    "is_attacking_mid",
    "is_central_mid",
    "is_cdm",
    "is_fullback",
    "is_goalkeeper",
    # Nationality dummies (baseline = other_europe)
    "is_brazilian",
    "is_french",
    "is_english",
    "is_spanish",
    "is_german",
    "is_argentinian",
    "is_portuguese",
    "is_african",
    "is_asian",
    "is_south_american_other",
    # Position-specific age interactions (replaces global age / age_squared)
] + [
    f"{pos}_age"    for pos in _AGE_POSITIONS
] + [
    f"{pos}_age_sq" for pos in _AGE_POSITIONS
] + [
    # Goals & assists × position (outfield only — GK version replaced below)
    "striker_goals_per_90",
    "winger_goals_per_90",
    "attmid_goals_per_90",
    "cm_goals_per_90",
    "cdm_goals_per_90",
    "fullback_goals_per_90",
    "cb_goals_per_90",
    "striker_assists_per_90",
    "winger_assists_per_90",
    "attmid_assists_per_90",
    "cm_assists_per_90",
    "cdm_assists_per_90",
    "fullback_assists_per_90",
    "cb_assists_per_90",
    # GK-specific keeper stats (is_goalkeeper × stat)
    "gk_save_pct",
    "gk_cs_per_90",
    "gk_ga_per_90",
    "gk_sota_per_90",
    "gk_pk_save_pct",
] + [
    # WhoScored stat × position interactions (15 stats × 8 positions = 120)
    f"{stat}_{pos}"
    for stat in _WS_STATS
    for pos in _POSITIONS
]

INTERACTION_VARS = [
    "striker_goals_per_90", "winger_goals_per_90", "attmid_goals_per_90",
    "cm_goals_per_90", "cdm_goals_per_90", "fullback_goals_per_90",
    "cb_goals_per_90", "gk_goals_per_90",
    "striker_assists_per_90", "winger_assists_per_90", "attmid_assists_per_90",
    "cm_assists_per_90", "cdm_assists_per_90", "fullback_assists_per_90",
    "cb_assists_per_90", "gk_assists_per_90",
] + [f"{stat}_{pos}" for stat in _WS_STATS for pos in _POSITIONS]


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("=" * 60)
    print("Model — OLS Regression")
    print("=" * 60)

    print(f"\nLoading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8")
    print(f"  Rows: {len(df)}")

    # --- Verify all columns exist ---
    missing = [v for v in [DEPENDENT_VAR] + INDEPENDENT_VARS if v not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return

    # --- Prepare X and y ---
    y = df[DEPENDENT_VAR].astype(float)
    X = df[INDEPENDENT_VARS].astype(float)

    # Drop rows with any NaN in X or y
    mask = X.notna().all(axis=1) & y.notna()
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with NaN in features or target.")
    X = X[mask]
    y = y[mask]
    df_model = df[mask].copy().reset_index(drop=True)

    print(f"  Modelling on {len(y)} observations, {len(INDEPENDENT_VARS)} predictors.")

    # --- Add constant for statsmodels ---
    X_const = sm.add_constant(X)

    # --- Fit OLS ---
    print("\nFitting OLS model...")
    model = sm.OLS(y, X_const)
    results = model.fit()

    # --- Print summary ---
    summary_text = results.summary().as_text()
    print("\n" + summary_text)

    # --- Save summary to file ---
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nModel summary saved to {SUMMARY_PATH}")

    # --- Compute residuals and predictions ---
    predicted_log = results.fittedvalues
    residuals = y - predicted_log

    df_model["predicted_log_value"] = predicted_log.values
    df_model["residual"] = residuals.values
    df_model["predicted_market_value_eur"] = np.exp(predicted_log.values)

    # residual = log_market_value - predicted_log_value
    # Positive → actual > predicted → OVERVALUED
    # Negative → actual < predicted → UNDERVALUED

    def valuation_label(r: float) -> str:
        if r > 0.15:
            return "overvalued"
        elif r < -0.15:
            return "undervalued"
        return "fairly valued"

    df_model["valuation_label"] = df_model["residual"].apply(valuation_label)

    # --- Sort by residual descending (most overvalued at top) ---
    df_model = df_model.sort_values("residual", ascending=False).reset_index(drop=True)

    # --- Save results ---
    df_model.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nFull results saved to {RESULTS_PATH}")

    # --- Save model coefficients JSON for Streamlit what-if tool ---
    coef_dict = {k: float(v) for k, v in results.params.items()}
    pval_dict = {k: float(v) for k, v in results.pvalues.items()}
    coef_json = {
        "intercept": float(results.params.get("const", 0.0)),
        "coefficients": {k: v for k, v in coef_dict.items() if k != "const"},
        "pvalues": {k: v for k, v in pval_dict.items() if k != "const"},
        "residual_std": float(np.std(residuals.values)),
        "mean_log_value": float(y.mean()),
    }
    with open(COEF_PATH, "w", encoding="utf-8") as f:
        json.dump(coef_json, f, indent=2)
    print(f"\nModel coefficients saved to {COEF_PATH}")

    cols = ["player_name", "club", "market_value_eur", "predicted_market_value_eur", "residual"]

    # ── 1. R² summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"R²:          {results.rsquared:.4f}")
    print(f"Adjusted R²: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.2f}  (p={results.f_pvalue:.4e})")
    print(f"N:           {int(results.nobs)}")

    # ── 2. Team tier coefficients ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Team Tier Coefficients  (baseline = mid table, positions 7–14)")
    print(f"  {'Variable':<14} {'Coef':>8}  {'P-value':>9}  {'Mkt Premium':>12}")
    print("  " + "-" * 48)
    for tier_var in ("is_top4", "is_top6", "is_bottom6"):
        if tier_var in results.params.index:
            c = float(results.params[tier_var])
            p = float(results.pvalues[tier_var])
            prem = (np.exp(c) - 1.0) * 100
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {tier_var:<20} {c:>8.4f}  {p:>9.4f}  {prem:>+11.1f}% {stars}")
        else:
            print(f"  {tier_var:<20}  (dropped — collinearity)")

    # ── 2b. Historic Big 6 prestige coefficient ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("Historic Big 6 Prestige Premium  (independent of current standing)")
    if "is_historic_top6" in results.params.index:
        c    = float(results.params["is_historic_top6"])
        p    = float(results.pvalues["is_historic_top6"])
        prem = (np.exp(c) - 1.0) * 100
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  is_historic_top6: coef={c:+.4f}  p={p:.4f}  premium={prem:+.1f}% {stars}")
    else:
        print("  is_historic_top6: (dropped — not in model)")

    # ── 2c. Promoted vs bottom 6 comparison ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Promoted vs Bottom 6 effect:")
    for var in ("is_promoted", "is_bottom6"):
        if var in results.params.index:
            c    = float(results.params[var])
            p    = float(results.pvalues[var])
            eff  = (np.exp(c) - 1.0) * 100
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {var:<12}  coef={c:+.4f}  p={p:.4f}  effect={eff:+.1f}% {stars}")
        else:
            print(f"  {var:<12}  (dropped — not in model)")

    # ── 3. Implied peak age per position ──────────────────────────────────────
    age_mean = 0.0
    if os.path.exists(AGE_MEAN_PATH):
        with open(AGE_MEAN_PATH, "r", encoding="utf-8") as _f:
            age_mean = float(json.load(_f).get("age_mean", 0.0))
    print(f"\n{'=' * 60}")
    print(f"Implied Peak Age by Position  (centered: peak = age_mean − coef_age / (2 × coef_age_sq))")
    print(f"  age_mean used for de-centering: {age_mean:.4f}")
    print(f"  {'Position':<12} {'Age coef':>10}  {'Age² coef':>10}  {'Peak age':>10}")
    print("  " + "-" * 48)
    for pos in _AGE_POSITIONS:
        age_var = f"{pos}_age"
        agesq_var = f"{pos}_age_sq"
        c_age   = float(results.params.get(age_var,   float("nan")))
        c_agesq = float(results.params.get(agesq_var, float("nan")))
        if np.isnan(c_age) or np.isnan(c_agesq) or c_agesq >= 0:
            peak_str = "could not estimate"
        else:
            peak_centered = -c_age / (2.0 * c_agesq)
            peak = age_mean + peak_centered
            peak_str = f"{peak:.1f}" if 18.0 <= peak <= 45.0 else "could not estimate"
        print(f"  {pos:<12} {c_age:>10.4f}  {c_agesq:>10.4f}  {peak_str:>10}")

    # ── 4. Top 20 standardised coefficients ───────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Top 20 Standardised Coefficients  |β × σ(x) / σ(y)|")
    std_y = float(y.std())
    std_rows = []
    for var in results.params.index:
        if var == "const":
            continue
        if var not in X.columns:
            continue
        std_x = float(X[var].std())
        c = float(results.params[var])
        p = float(results.pvalues[var])
        std_c = abs(c * std_x / std_y) if std_y > 0 else 0.0
        std_rows.append({"variable": var, "coef": c, "std_coef": std_c, "pvalue": p})
    std_rows.sort(key=lambda x: x["std_coef"], reverse=True)
    print(f"  {'Variable':<40} {'Std Coef':>10}  {'Raw Coef':>9}  {'P-value':>9}")
    print("  " + "-" * 72)
    for r in std_rows[:20]:
        stars = "***" if r["pvalue"] < 0.01 else "**" if r["pvalue"] < 0.05 else "*" if r["pvalue"] < 0.1 else ""
        print(f"  {r['variable']:<40} {r['std_coef']:>10.4f}  {r['coef']:>9.4f}  {r['pvalue']:>9.4f} {stars}")

    # ── 5. Top 10 under / overvalued ──────────────────────────────────────────
    problem_vars = [
        v for v in INDEPENDENT_VARS
        if v in results.params.index
        and (pd.isna(results.params[v]) or results.params[v] == 0.0)
    ]
    missing_vars = [v for v in INDEPENDENT_VARS if v not in results.params.index]
    if problem_vars or missing_vars:
        print("\nWARNING — variables with NaN or zero coefficient (consider dropping):")
        for v in problem_vars:
            print(f"  {v}  coef={results.params[v]}")
        for v in missing_vars:
            print(f"  {v}  (not in model — likely dropped due to perfect collinearity)")
    else:
        print("\nAll variables have non-zero, non-NaN coefficients.")

    print("\nTop 10 most UNDERVALUED (most negative residual — actual < predicted):")
    top_under = df_model.tail(10)[cols].iloc[::-1]
    print(top_under.to_string(index=False))

    print("\nTop 10 most OVERVALUED (most positive residual — actual > predicted):")
    top_over = df_model.head(10)[cols]
    print(top_over.to_string(index=False))


if __name__ == "__main__":
    main()
