"""
regression.py — OLS regression on master.csv per league/season.

Usage:
    python3 model/regression.py --league premier_league --season 2024-25
    python3 model/regression.py --league all --season 2024-25
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

LEAGUE_NAMES = {
    "premier_league": "Premier League",
    "bundesliga": "Bundesliga",
    "la_liga": "La Liga",
    "serie_a": "Serie A",
    "liga_portugal": "Liga Portugal",
}

GLOBAL_CONTROLS = [
    "minutes_played",
    "is_top4", "is_top6", "is_bottom6",
    "is_historic_top", "is_promoted",
    "yellow_cards",
    "pass_success_pct",
    "fouls_per_game",
    "dribbled_past_per_game",
    "xg_diff",
]

NATIONALITY_DUMMIES = [
    "is_brazilian", "is_french", "is_english", "is_spanish",
    "is_german", "is_argentine", "is_portuguese", "is_african",
    "is_asian", "is_south_american_other",
]

POSITION_DUMMIES = [
    "is_goalkeeper", "is_left_back", "is_right_back",
    "is_cdm", "is_central_mid", "is_attacking_mid",
    "is_left_winger", "is_right_winger", "is_striker",
]

POSITIONS = ["gk", "cb", "lb", "rb", "cdm", "cm", "am", "lw", "rw", "st"]

STAT_COLS = [
    "goals_per_90", "assists_per_90",
    "tackles_per_game", "interceptions_per_game",
    "clearances_per_game", "blocks_per_game",
    "aerials_won", "key_passes_per_game",
    "dribbles_per_game", "crosses_per_game",
    "long_balls_per_game", "through_balls_per_game",
    "avg_passes_per_game", "shots_per_game",
    "xg", "xg_per_shot", "fouled_per_game",
]

GK_COLS = ["gk_save_pct", "gk_ga_per_90", "gk_cs_per_90", "gk_pk_save_pct"]


def build_feature_list(df_cols):
    cols = list(GLOBAL_CONTROLS)
    cols += list(NATIONALITY_DUMMIES)
    cols += list(POSITION_DUMMIES)
    for pos in POSITIONS:
        cols.append("age_" + pos)
        cols.append("age_sq_" + pos)
    for pos in POSITIONS:
        for stat in STAT_COLS:
            cols.append(stat + "_" + pos)
    cols += list(GK_COLS)
    return [c for c in cols if c in df_cols]


def drop_zero_variance(X):
    keep = [c for c in X.columns if X[c].std() > 0]
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print("   Dropped zero-variance columns (" + str(len(dropped)) + "): " +
              ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else ""))
    return X[keep]


def drop_collinear(X):
    kept = list(X.columns)
    while True:
        mat = X[kept].values
        r = np.linalg.matrix_rank(mat)
        if r == len(kept):
            break
        removed = None
        for i in range(len(kept) - 1, -1, -1):
            candidate = [c for j, c in enumerate(kept) if j != i]
            if np.linalg.matrix_rank(X[candidate].values) == r:
                removed = kept[i]
                kept = candidate
                break
        if removed is None:
            break
        print("   Dropped collinear column: " + removed)
    return X[kept]


def fit_ols(y, X):
    X_const = sm.add_constant(X, has_constant="add")
    try:
        return sm.OLS(y, X_const).fit()
    except np.linalg.LinAlgError as e:
        print("   Singular matrix — dropping columns iteratively...")
        cols = list(X.columns)
        while cols:
            try:
                Xc = sm.add_constant(X[cols], has_constant="add")
                res = sm.OLS(y, Xc).fit()
                print("   Fit succeeded with " + str(len(cols)) + " columns")
                return res
            except np.linalg.LinAlgError:
                dropped = cols.pop()
                print("   Dropped: " + dropped)
        raise RuntimeError("Could not fit OLS after dropping all columns") from e


def run_league(league_key, season):
    league_name = LEAGUE_NAMES.get(league_key, league_key)
    print("\n" + "=" * 60)
    print("Regression: " + league_name + " " + season)
    print("=" * 60)

    data_dir = PROC_DIR / league_key / season
    master_path = data_dir / "master.csv"

    if not master_path.exists():
        print("   ERROR: " + str(master_path) + " not found")
        return None

    df = pd.read_csv(master_path, low_memory=False)
    print("   Loaded " + str(len(df)) + " rows")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["log_market_value"])
    print("   " + str(len(df)) + " rows after dropping NaN/inf log_market_value")

    y = df["log_market_value"].values

    feature_cols = build_feature_list(df.columns.tolist())
    X = df[feature_cols].copy().fillna(0)

    print("   Feature columns before cleaning: " + str(len(X.columns)))
    X = drop_zero_variance(X)
    X = drop_collinear(X)
    print("   Feature columns after cleaning: " + str(len(X.columns)))

    res = fit_ols(y, X)

    # ── results.csv ───────────────────────────────────────────────
    out = df.copy()
    fitted = res.fittedvalues
    # fittedvalues index may not align with df index after dropna reset
    out["predicted_log_value"] = fitted.values if hasattr(fitted, "values") else fitted
    out["residual"] = y - out["predicted_log_value"].values
    out["predicted_market_value_eur"] = np.exp(out["predicted_log_value"])

    def label(r):
        if r > 0.25:
            return "Overvalued"
        elif r < -0.25:
            return "Undervalued"
        return "Fair Value"

    out["valuation_label"] = out["residual"].apply(label)
    out.to_csv(data_dir / "results.csv", index=False)
    print("   Saved results.csv")

    # ── age_mean ──────────────────────────────────────────────────
    age_mean = 0.0
    age_mean_path = data_dir / "age_mean.json"
    if age_mean_path.exists():
        with open(age_mean_path) as f:
            age_mean = json.load(f).get("age_mean", 0.0)

    # ── model_coefficients.json ───────────────────────────────────
    coef_dict = {
        "league": league_name,
        "season": season,
        "n_obs": int(res.nobs),
        "r_squared": float(round(res.rsquared, 6)),
        "adj_r_squared": float(round(res.rsquared_adj, 6)),
        "f_statistic": float(round(res.fvalue, 4)),
        "age_mean": float(age_mean),
        "coefficients": {k: float(v) for k, v in res.params.items()},
        "pvalues": {k: float(v) for k, v in res.pvalues.items()},
        "std_errors": {k: float(v) for k, v in res.bse.items()},
    }
    with open(data_dir / "model_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=2)
    print("   Saved model_coefficients.json")

    # ── model_summary.txt ─────────────────────────────────────────
    with open(data_dir / "model_summary.txt", "w") as f:
        f.write(str(res.summary()))
    print("   Saved model_summary.txt")

    # ── sanity check ──────────────────────────────────────────────
    print("\nR²: " + str(round(res.rsquared, 3)) +
          "  Adj R²: " + str(round(res.rsquared_adj, 3)) +
          "  N: " + str(int(res.nobs)) +
          "  F: " + str(round(res.fvalue, 1)))

    out_sorted = out.sort_values("residual", ascending=False).reset_index(drop=True)

    print("\nTop 5 overvalued:")
    for _, row in out_sorted.head(5).iterrows():
        print("  {:<25} | {:<20} | Actual €{:.1f}M | Predicted €{:.1f}M | Residual {:.3f}".format(
            str(row.get("player_name", ""))[:25],
            str(row.get("club", ""))[:20],
            row["market_value_eur"] / 1e6,
            row["predicted_market_value_eur"] / 1e6,
            row["residual"],
        ))

    print("\nTop 5 undervalued:")
    for _, row in out_sorted.tail(5).iterrows():
        print("  {:<25} | {:<20} | Actual €{:.1f}M | Predicted €{:.1f}M | Residual {:.3f}".format(
            str(row.get("player_name", ""))[:25],
            str(row.get("club", ""))[:20],
            row["market_value_eur"] / 1e6,
            row["predicted_market_value_eur"] / 1e6,
            row["residual"],
        ))

    sig = {
        k: (float(res.params[k]), float(res.pvalues[k]))
        for k in res.params.index
        if res.pvalues[k] < 0.05 and k != "const"
    }
    sig_sorted = sorted(sig.items(), key=lambda x: abs(x[1][0]), reverse=True)

    print("\nSignificant coefficients (p < 0.05), sorted by |coef|:")
    for var, (coef, pval) in sig_sorted[:20]:
        print("  {:<45} | coef {:>8.4f} | p {:.4f}".format(var, coef, pval))

    return {
        "league": league_name,
        "n": int(res.nobs),
        "r2": round(res.rsquared, 4),
        "adj_r2": round(res.rsquared_adj, 4),
        "is_historic_top": float(res.params.get("is_historic_top", float("nan"))),
        "is_promoted": float(res.params.get("is_promoted", float("nan"))),
        "is_bottom6": float(res.params.get("is_bottom6", float("nan"))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league",
        choices=list(LEAGUE_NAMES.keys()) + ["all"], default="premier_league")
    parser.add_argument("--season", default="2024-25")
    args = parser.parse_args()

    if args.league == "all":
        summaries = []
        for lk in LEAGUE_NAMES:
            try:
                s = run_league(lk, args.season)
                if s:
                    summaries.append(s)
            except Exception as e:
                print("\nERROR in " + lk + ": " + str(e))
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 95)
        print("CROSS-LEAGUE SUMMARY")
        print("=" * 95)
        print("{:<18} {:>5} {:>7} {:>8} {:>18} {:>14} {:>12}".format(
            "League", "N", "R²", "Adj R²",
            "is_historic_top", "is_promoted", "is_bottom6"))
        print("-" * 95)
        for s in summaries:
            print("{:<18} {:>5} {:>7.4f} {:>8.4f} {:>18.4f} {:>14.4f} {:>12.4f}".format(
                s["league"], s["n"], s["r2"], s["adj_r2"],
                s["is_historic_top"], s["is_promoted"], s["is_bottom6"]))
    else:
        run_league(args.league, args.season)


if __name__ == "__main__":
    main()
