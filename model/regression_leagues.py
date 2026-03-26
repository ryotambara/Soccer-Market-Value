"""
model/regression_leagues.py

Runs OLS regression for each of the 5 new-pipeline leagues.
Reads from  data/processed/{league}/2024-25/features.csv
Outputs to  data/processed/{league}/2024-25/results.csv
                                             model_coefficients.json
                                             model_summary.txt

Run: python model/regression_leagues.py
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

LEAGUE_DIRS = {
    "Premier League":  os.path.join(PROC_DIR, "premier_league", "2024-25"),
    "Bundesliga":      os.path.join(PROC_DIR, "bundesliga",     "2024-25"),
    "La Liga":         os.path.join(PROC_DIR, "la_liga",        "2024-25"),
    "Serie A":         os.path.join(PROC_DIR, "serie_a",        "2024-25"),
    "Liga Portugal":   os.path.join(PROC_DIR, "liga_portugal",  "2024-25"),
}

DEPENDENT_VAR = "log_market_value"

_POSITIONS = ["striker", "winger", "attmid", "cm", "cdm", "fullback", "cb", "gk"]

# WhoScored stats (per-game, as named in parse_leagues.py output)
_WS_STATS = [
    "tackles_per_game", "interceptions_per_game", "clearances_per_game",
    "blocks_per_game", "aerials_won", "key_passes_per_game", "dribbles_per_game",
    "avg_passes_per_game", "shots_per_game", "xg", "fouled_per_game",
]

# Global stats (not position-interacted)
_GLOBAL_STATS = [
    "minutes_played",
    "is_top4", "is_top6", "is_bottom6",
    "is_historic_top6", "is_promoted",
    "rating", "xg_diff", "pass_success_pct",
    "fouls_per_game", "dribbled_past_per_game", "yellow_cards",
]

# Position dummies (baseline = cb)
_POS_DUMMIES = [
    "is_striker", "is_winger", "is_attmid", "is_cm",
    "is_cdm", "is_fullback", "is_gk",
]

# Nationality dummies (baseline = other European)
_NAT_DUMMIES = [
    "is_english", "is_spanish", "is_french", "is_german", "is_portuguese",
    "is_brazilian", "is_argentinian", "is_dutch", "is_african",
    "is_south_american", "is_scandinavian",
]

# Age × position interactions
_AGE_TERMS = [
    f"{pos}_age"    for pos in _POSITIONS
] + [
    f"{pos}_age_sq" for pos in _POSITIONS
]

# Goals/assists × position
_GOAL_ASSIST_TERMS = [
    f"{pos}_goals_per_90"   for pos in _POSITIONS if pos != "gk"
] + [
    f"{pos}_assists_per_90" for pos in _POSITIONS if pos != "gk"
]

# GK-specific stats (no position interaction — only GKs have these)
_GK_STATS = [
    "gk_save_pct", "gk_cs_per_90", "gk_ga_per_90",
    "gk_sota_per_90", "gk_pk_save_pct",
]

# WS stat × position interactions
_WS_INTERACTIONS = [
    f"{stat}_{pos}"
    for stat in _WS_STATS
    for pos in _POSITIONS
]

ALL_CANDIDATE_VARS = (
    _GLOBAL_STATS + _POS_DUMMIES + _NAT_DUMMIES
    + _AGE_TERMS + _GOAL_ASSIST_TERMS + _GK_STATS + _WS_INTERACTIONS
)


def valuation_label(r: float) -> str:
    if r > 0.15:
        return "overvalued"
    elif r < -0.15:
        return "undervalued"
    return "fairly valued"


def run_regression(name: str, league_dir: str) -> None:
    features_path = os.path.join(league_dir, "features.csv")
    results_path  = os.path.join(league_dir, "results.csv")
    summary_path  = os.path.join(league_dir, "model_summary.txt")
    coef_path     = os.path.join(league_dir, "model_coefficients.json")

    if not os.path.exists(features_path):
        print(f"  SKIP {name}: {features_path} not found.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Regression: {name} (2024-25)")
    print(f"{'=' * 60}")

    df = pd.read_csv(features_path)
    print(f"  Rows loaded: {len(df)}")

    # Require dependent variable
    if DEPENDENT_VAR not in df.columns:
        print(f"  ERROR: '{DEPENDENT_VAR}' column missing.")
        return

    # Build variable list from candidates that exist in this dataset
    ind_vars = [v for v in ALL_CANDIDATE_VARS if v in df.columns]

    # Drop any var that is entirely NaN or zero-variance
    def is_usable(col):
        s = df[col].astype(float)
        if s.isna().all():
            return False
        if s.dropna().std() < 1e-10:
            return False
        return True

    ind_vars = [v for v in ind_vars if is_usable(v)]
    print(f"  Usable predictors: {len(ind_vars)}")

    y  = df[DEPENDENT_VAR].astype(float)
    X  = df[ind_vars].astype(float)

    # Drop rows where target is missing
    y_mask = y.notna()
    X, y = X[y_mask], y[y_mask]
    df_model = df[y_mask].copy().reset_index(drop=True)

    # Impute missing feature values with column means (avoids massive row loss)
    X = X.fillna(X.mean())

    # Drop any row still with NaN (shouldn't happen, but safety check)
    final_mask = X.notna().all(axis=1)
    dropped = (~final_mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows still with NaN after imputation.")
    X, y = X[final_mask], y[final_mask]
    df_model = df_model[final_mask].reset_index(drop=True)
    print(f"  Modelling on {len(y)} observations.")

    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const)
    res     = model.fit()

    summary_text = res.summary().as_text()
    print(f"\n  R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}  "
          f"F={res.fvalue:.2f}  N={int(res.nobs)}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Compute predictions and residuals
    pred_log = res.fittedvalues
    residuals = y - pred_log

    df_model["predicted_log_value"]        = pred_log.values
    df_model["residual"]                   = residuals.values
    df_model["predicted_market_value_eur"] = np.exp(pred_log.values)
    df_model["valuation_label"]            = df_model["residual"].apply(valuation_label)
    df_model["percentile"] = (
        df_model["predicted_market_value_eur"]
        .rank(pct=True)
        .mul(100)
        .round(0)
        .astype(int)
    )

    df_model = df_model.sort_values("residual", ascending=False).reset_index(drop=True)
    df_model.to_csv(results_path, index=False, encoding="utf-8")
    print(f"  Results → {results_path}")

    coef_json = {
        "intercept":    float(res.params.get("const", 0.0)),
        "coefficients": {k: float(v) for k, v in res.params.items() if k != "const"},
        "pvalues":      {k: float(v) for k, v in res.pvalues.items() if k != "const"},
        "residual_std": float(np.std(residuals.values)),
        "mean_log_value": float(y.mean()),
    }
    with open(coef_path, "w", encoding="utf-8") as f:
        json.dump(coef_json, f, indent=2)
    print(f"  Coefficients → {coef_path}")

    # Key coefficient summary
    print("\n  Top predictors by |coef|:")
    params = {k: float(v) for k, v in res.params.items() if k != "const"}
    pvals  = {k: float(v) for k, v in res.pvalues.items() if k != "const"}
    top = sorted(params.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for var, c in top:
        p = pvals.get(var, 1.0)
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"    {var:<40} {c:>+8.4f}  p={p:.4f} {stars}")


if __name__ == "__main__":
    for league_name, league_dir in LEAGUE_DIRS.items():
        try:
            run_regression(league_name, league_dir)
        except Exception as e:
            import traceback
            print(f"\nERROR in {league_name}: {e}")
            traceback.print_exc()

    print(f"\n\nAll regressions complete.")
