"""
model/regression_2024.py

OLS regression for the 2024-25 season.
Reads from data/processed/2024-25/features.csv.
Saves results, coefficients, and summary to data/processed/2024-25/.

Identical structure to regression.py but uses the 2024-25 data paths.

Run: python model/regression_2024.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import json

BASE = os.path.dirname(__file__)
SEASON_DIR    = os.path.join(BASE, "..", "data", "processed", "2024-25")
FEATURES_PATH = os.path.join(SEASON_DIR, "features.csv")
RESULTS_PATH  = os.path.join(SEASON_DIR, "results.csv")
SUMMARY_PATH  = os.path.join(SEASON_DIR, "model_summary.txt")
COEF_PATH     = os.path.join(SEASON_DIR, "model_coefficients.json")
AGE_MEAN_PATH = os.path.join(SEASON_DIR, "age_mean.json")

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
    "contract_months_remaining",
    # Team tier (baseline = mid table, positions 7-14)
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
    # Position-specific age interactions
] + [
    f"{pos}_age"    for pos in _AGE_POSITIONS
] + [
    f"{pos}_age_sq" for pos in _AGE_POSITIONS
] + [
    # Goals & assists × position (outfield only — GK version replaced below)
    "striker_goals_per_90", "winger_goals_per_90", "attmid_goals_per_90",
    "cm_goals_per_90", "cdm_goals_per_90", "fullback_goals_per_90",
    "cb_goals_per_90",
    "striker_assists_per_90", "winger_assists_per_90", "attmid_assists_per_90",
    "cm_assists_per_90", "cdm_assists_per_90", "fullback_assists_per_90",
    "cb_assists_per_90",
    # GK-specific keeper stats (is_goalkeeper × stat)
    "gk_save_pct",
    "gk_cs_per_90",
    "gk_ga_per_90",
    "gk_sota_per_90",
    "gk_pk_save_pct",
] + [
    # WhoScored stat × position (15 stats × 8 positions = 120)
    f"{stat}_{pos}"
    for stat in _WS_STATS
    for pos in _POSITIONS
]


def main():
    os.makedirs(SEASON_DIR, exist_ok=True)

    print("=" * 60)
    print("Model 2024-25 — OLS Regression")
    print("=" * 60)

    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Missing {FEATURES_PATH}")
        print("  Run the pipeline in order:")
        print("    python scraper/parse_football_data_2024.py")
        print("    python scraper/whoscored_parse_2024.py")
        print("    python pipeline/merge_2024.py")
        print("    python pipeline/clean_2024.py")
        print("    python pipeline/features_2024.py")
        raise SystemExit(1)

    print(f"\nLoading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8")
    print(f"  Rows: {len(df)}")

    # --- Check which vars are available ---
    missing = [v for v in [DEPENDENT_VAR] + INDEPENDENT_VARS if v not in df.columns]
    if missing:
        print(f"\n  Missing columns ({len(missing)}):")
        for m in missing:
            print(f"    {m}")
        # Drop missing vars and continue
        INDEPENDENT_VARS_USED = [v for v in INDEPENDENT_VARS if v in df.columns]
        print(f"\n  Proceeding with {len(INDEPENDENT_VARS_USED)} available vars.")
    else:
        INDEPENDENT_VARS_USED = list(INDEPENDENT_VARS)
        print(f"  All {len(INDEPENDENT_VARS_USED)} independent variables present.")

    # --- Prepare X and y ---
    y = df[DEPENDENT_VAR].astype(float)
    X = df[INDEPENDENT_VARS_USED].astype(float)

    mask = X.notna().all(axis=1) & y.notna()
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with NaN in features or target.")
    X = X[mask]
    y = y[mask]
    df_model = df[mask].copy().reset_index(drop=True)

    print(f"  Modelling on {len(y)} observations, {len(INDEPENDENT_VARS_USED)} predictors.")

    # --- Fit OLS ---
    X_const = sm.add_constant(X)
    print("\nFitting OLS model...")
    model   = sm.OLS(y, X_const)
    results = model.fit()

    summary_text = results.summary().as_text()
    print("\n" + summary_text)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nModel summary saved to {SUMMARY_PATH}")

    # --- Residuals & predictions ---
    predicted_log = results.fittedvalues
    residuals     = y - predicted_log

    df_model["predicted_log_value"]       = predicted_log.values
    df_model["residual"]                  = residuals.values
    df_model["predicted_market_value_eur"] = np.exp(predicted_log.values)

    def valuation_label(r: float) -> str:
        if r > 0.15:
            return "overvalued"
        elif r < -0.15:
            return "undervalued"
        return "fairly valued"

    df_model["valuation_label"] = df_model["residual"].apply(valuation_label)
    df_model = df_model.sort_values("residual", ascending=False).reset_index(drop=True)

    df_model.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nFull results saved to {RESULTS_PATH}")

    # --- Save coefficients JSON ---
    coef_dict = {k: float(v) for k, v in results.params.items()}
    pval_dict = {k: float(v) for k, v in results.pvalues.items()}
    coef_json = {
        "intercept":      float(results.params.get("const", 0.0)),
        "coefficients":   {k: v for k, v in coef_dict.items() if k != "const"},
        "pvalues":        {k: v for k, v in pval_dict.items() if k != "const"},
        "residual_std":   float(np.std(residuals.values)),
        "mean_log_value": float(y.mean()),
        "season":         "2024-25",
    }
    with open(COEF_PATH, "w", encoding="utf-8") as f:
        json.dump(coef_json, f, indent=2)
    print(f"Model coefficients saved to {COEF_PATH}")

    cols = ["player_name", "club", "market_value_eur",
            "predicted_market_value_eur", "residual"]

    # ── 1. R² summary ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"R²:          {results.rsquared:.4f}")
    print(f"Adjusted R²: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.2f}  (p={results.f_pvalue:.4e})")
    print(f"N:           {int(results.nobs)}")

    # ── 2. Team tier coefficients ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Team Tier Coefficients  (baseline = mid table, positions 7–14)")
    print("  2024-25: top4=Liverpool/Arsenal/Chelsea/Man City")
    print("           top6=Newcastle/Aston Villa")
    print("           bottom6=Crystal Palace/Wolves/Everton/Leicester/Ipswich/Southampton")
    print(f"  {'Variable':<14} {'Coef':>8}  {'P-value':>9}  {'Mkt Premium':>12}")
    print("  " + "-" * 48)
    for tier_var in ("is_top4", "is_top6", "is_bottom6"):
        if tier_var in results.params.index:
            c    = float(results.params[tier_var])
            p    = float(results.pvalues[tier_var])
            prem = (np.exp(c) - 1.0) * 100
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {tier_var:<20} {c:>8.4f}  {p:>9.4f}  {prem:>+11.1f}% {stars}")
        else:
            print(f"  {tier_var:<20}  (dropped — collinearity)")

    # ── 2b. Historic Big 6 prestige coefficient ───────────────────
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

    # ── 2c. Promoted vs bottom 6 comparison ──────────────────────
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

    # ── 3. Implied peak age per position ──────────────────────────
    age_mean = 0.0
    if os.path.exists(AGE_MEAN_PATH):
        with open(AGE_MEAN_PATH, "r", encoding="utf-8") as _f:
            age_mean = float(json.load(_f).get("age_mean", 0.0))
    print(f"\n{'=' * 60}")
    print(f"Implied Peak Age by Position  "
          f"(peak = age_mean + (−coef_age / (2 × coef_age_sq)))")
    print(f"  age_mean: {age_mean:.4f}")
    print(f"  {'Position':<12} {'Age coef':>10}  {'Age² coef':>10}  {'Peak age':>10}")
    print("  " + "-" * 48)
    for pos in _AGE_POSITIONS:
        age_var   = f"{pos}_age"
        agesq_var = f"{pos}_age_sq"
        c_age     = float(results.params.get(age_var,   float("nan")))
        c_agesq   = float(results.params.get(agesq_var, float("nan")))
        if np.isnan(c_age) or np.isnan(c_agesq) or c_agesq >= 0:
            peak_str = "could not estimate"
        else:
            peak_centered = -c_age / (2.0 * c_agesq)
            peak = age_mean + peak_centered
            peak_str = f"{peak:.1f}" if 18.0 <= peak <= 45.0 else "could not estimate"
        print(f"  {pos:<12} {c_age:>10.4f}  {c_agesq:>10.4f}  {peak_str:>10}")

    # ── 4. Top 20 standardised coefficients ───────────────────────
    print(f"\n{'=' * 60}")
    print("Top 20 Standardised Coefficients  |β × σ(x) / σ(y)|")
    std_y    = float(y.std())
    std_rows = []
    for var in results.params.index:
        if var == "const" or var not in X.columns:
            continue
        std_x = float(X[var].std())
        c     = float(results.params[var])
        p     = float(results.pvalues[var])
        std_c = abs(c * std_x / std_y) if std_y > 0 else 0.0
        std_rows.append({"variable": var, "coef": c, "std_coef": std_c, "pvalue": p})
    std_rows.sort(key=lambda x: x["std_coef"], reverse=True)
    print(f"  {'Variable':<40} {'Std Coef':>10}  {'Raw Coef':>9}  {'P-value':>9}")
    print("  " + "-" * 72)
    for r in std_rows[:20]:
        stars = ("***" if r["pvalue"] < 0.01 else
                 "**"  if r["pvalue"] < 0.05 else
                 "*"   if r["pvalue"] < 0.1  else "")
        print(f"  {r['variable']:<40} {r['std_coef']:>10.4f}  "
              f"{r['coef']:>9.4f}  {r['pvalue']:>9.4f} {stars}")

    # ── 5. Collinearity check ─────────────────────────────────────
    problem_vars = [
        v for v in INDEPENDENT_VARS_USED
        if v in results.params.index
        and (pd.isna(results.params[v]) or results.params[v] == 0.0)
    ]
    missing_vars = [v for v in INDEPENDENT_VARS_USED if v not in results.params.index]
    if problem_vars or missing_vars:
        print("\nWARNING — variables with NaN/zero coef (dropped by OLS):")
        for v in problem_vars:
            print(f"  {v}  coef={results.params[v]}")
        for v in missing_vars:
            print(f"  {v}  (not in model)")
    else:
        print("\nAll variables have non-zero, non-NaN coefficients.")

    # ── 6. Top 10 under/overvalued ────────────────────────────────
    print("\nTop 10 most UNDERVALUED (actual < model prediction):")
    top_under = df_model.tail(10)[cols].iloc[::-1]
    print(top_under.to_string(index=False))

    print("\nTop 10 most OVERVALUED (actual > model prediction):")
    top_over = df_model.head(10)[cols]
    print(top_over.to_string(index=False))


if __name__ == "__main__":
    main()
