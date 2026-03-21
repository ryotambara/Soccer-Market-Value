"""
pipeline/features.py

Adds feature engineering columns:
- log_market_value (natural log)
- age_squared
- team_league_position (hardcoded 2024-25 PL table)
- All position/nationality dummies already created in clean.py

Saves final feature matrix to data/processed/features.csv

Run: python pipeline/features.py
"""

import json
import subprocess
import sys
import pandas as pd
import numpy as np
import os

_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_PL_2526 = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26")
CLEANED_PATH  = os.path.join(_PROC_PL_2526, "cleaned.csv")
OUT_PATH      = os.path.join(_PROC_PL_2526, "features.csv")
AGE_MEAN_PATH = os.path.join(_PROC_PL_2526, "age_mean.json")

# ---------------------------------------------------------------------------
# Historic Big 6 clubs — all known name variants from TM + WhoScored scrapes
# ---------------------------------------------------------------------------
BIG_6_CLUBS = {
    "Manchester City", "Man City",
    "Arsenal", "Arsenal FC",
    "Liverpool", "Liverpool FC",
    "Chelsea", "Chelsea FC",
    "Manchester United", "Man Utd", "Manchester United FC",
    "Tottenham Hotspur", "Tottenham",
}

# ---------------------------------------------------------------------------
# 2025-26 Premier League standings
# ---------------------------------------------------------------------------
LEAGUE_TABLE = {
    "Arsenal":                    1,
    "Arsenal FC":                 1,
    "Manchester City":            2,
    "Man City":                   2,
    "Manchester United":          3,
    "Man Utd":                    3,
    "Manchester United FC":       3,
    "Aston Villa":                4,
    "Liverpool":                  5,
    "Liverpool FC":               5,
    "Chelsea":                    6,
    "Chelsea FC":                 6,
    "Brentford":                  7,
    "Brentford FC":               7,
    "Everton":                    8,
    "Everton FC":                 8,
    "Newcastle United":           9,
    "Newcastle United FC":        9,
    "AFC Bournemouth":            10,
    "Bournemouth":                10,
    "Fulham":                     11,
    "Fulham FC":                  11,
    "Brighton & Hove Albion":     12,
    "Brighton":                   12,
    "Brighton and Hove Albion":   12,
    "Sunderland":                 13,
    "Sunderland AFC":             13,
    "Crystal Palace":             14,
    "Crystal Palace FC":          14,
    "Leeds United":               15,
    "Leeds":                      15,
    "Tottenham Hotspur":          16,
    "Tottenham":                  16,
    "Nottingham Forest":          17,
    "Nottm Forest":               17,
    "West Ham United":            18,
    "West Ham":                   18,
    "West Ham United FC":         18,
    "Burnley":                    19,
    "Burnley FC":                 19,
    "Wolverhampton Wanderers":    20,
    "Wolves":                     20,
}

_LEAGUE_TABLE_LOWER = {k.lower(): v for k, v in LEAGUE_TABLE.items()}


def get_league_position(club: str) -> int:
    """Return the league table position for a club. Raises ValueError if not found."""
    if not isinstance(club, str):
        raise ValueError(f"Club value is not a string: {club!r} — please check the data.")
    pos = LEAGUE_TABLE.get(club)
    if pos is not None:
        return pos
    pos = _LEAGUE_TABLE_LOWER.get(club.lower().strip())
    if pos is not None:
        return pos
    raise ValueError(f"Club '{club}' not found in LEAGUE_TABLE — please add it.")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 60)
    print("Pipeline — Feature Engineering")
    print("=" * 60)

    print(f"\nLoading {CLEANED_PATH}...")
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8")
    print(f"  Input rows: {len(df)}")

    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce")
    df = df[df["minutes_played"] >= 500].copy()
    print(f"  After 500 min filter: {len(df)} players")

    # --- Log transform of market value ---
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
    df = df[df["market_value_eur"] > 0].copy()
    df["log_market_value"] = np.log(df["market_value_eur"])
    print(f"\n  log_market_value computed. Range: {df['log_market_value'].min():.2f} – {df['log_market_value'].max():.2f}")

    # --- Age squared ---
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_squared"] = df["age"] ** 2
    print(f"  age_squared computed.")

    # --- Team league position ---
    df["team_league_position"] = df["club"].apply(get_league_position)
    print(f"\n  team_league_position distribution:")
    print(df["team_league_position"].value_counts().sort_index().to_string())

    # --- Team tier dummies (baseline = mid table, positions 7-14) ---
    # is_top4: Arsenal, Man City, Man Utd, Aston Villa
    # is_top6: Liverpool, Chelsea
    # is_bottom6: Leeds, Spurs, Forest, West Ham, Burnley, Wolves
    df["is_top4"]    = (df["team_league_position"] <= 4).astype(int)
    df["is_top6"]    = (df["team_league_position"].isin([5, 6])).astype(int)
    df["is_bottom6"] = (df["team_league_position"] >= 15).astype(int)
    mid_mask = ~df["team_league_position"].isin(list(range(1, 7)) + list(range(15, 21)))
    print(f"\n  Team tier counts:")
    print(f"    is_top4    = {df['is_top4'].sum()} players (pos 1-4)")
    print(f"    is_top6    = {df['is_top6'].sum()} players (pos 5-6)")
    print(f"    mid table  = {mid_mask.sum()} players (pos 7-14, baseline)")
    print(f"    is_bottom6 = {df['is_bottom6'].sum()} players (pos 15-20)")

    # --- Historic Big 6 dummy ---
    df["is_historic_top6"] = df["club"].isin(BIG_6_CLUBS).astype(int)
    print(f"  is_historic_top6: {df['is_historic_top6'].sum()} players")

    # --- Promoted clubs dummy (newly promoted to PL this season) ---
    PROMOTED_2025_26 = [
        "Sunderland AFC", "Sunderland",
        "Leeds United", "Leeds",
        "Burnley FC", "Burnley",
    ]
    df["is_promoted"] = df["club"].isin(PROMOTED_2025_26).astype(int)
    print(f"  is_promoted: {df['is_promoted'].sum()} players")

    # --- Verify all expected feature columns exist ---
    expected_position_dummies = [
        "is_striker", "is_winger", "is_attacking_mid",
        "is_central_mid", "is_cdm", "is_fullback", "is_goalkeeper",
    ]
    expected_nationality_dummies = [
        "is_brazilian", "is_french", "is_english", "is_spanish",
        "is_german", "is_argentinian", "is_portuguese",
        "is_african", "is_asian", "is_south_american_other",
    ]

    missing_cols = []
    for col in expected_position_dummies + expected_nationality_dummies:
        if col not in df.columns:
            missing_cols.append(col)
            df[col] = 0  # Fill missing dummies with 0

    if missing_cols:
        print(f"\n  WARNING: Created {len(missing_cols)} missing dummy columns as zeros: {missing_cols}")

    # --- is_centre_back (explicit dummy for interaction terms; NOT used in regression) ---
    pos_sum = (
        df.get("is_striker", 0) + df.get("is_winger", 0) +
        df.get("is_attacking_mid", 0) + df.get("is_central_mid", 0) +
        df.get("is_cdm", 0) + df.get("is_fullback", 0) +
        df.get("is_goalkeeper", 0)
    )
    df["is_centre_back"] = (1 - pos_sum).clip(lower=0)

    # --- Center age (reduces multicollinearity with age_sq interactions) ---
    age_mean = float(df["age"].mean())
    df["age_centered"]    = df["age"] - age_mean
    df["age_centered_sq"] = df["age_centered"] ** 2
    print(f"\n  Age centering: mean age = {age_mean:.4f}")
    print(f"  age_centered range: {df['age_centered'].min():.1f} to {df['age_centered'].max():.1f}")

    with open(AGE_MEAN_PATH, "w", encoding="utf-8") as _f:
        json.dump({"age_mean": age_mean}, _f)
    print(f"  Age mean saved to {AGE_MEAN_PATH}")

    # --- Position-specific age interaction terms (using centered age) ---
    age_pos_pairs = [
        ("striker",  "is_striker"),
        ("winger",   "is_winger"),
        ("attmid",   "is_attacking_mid"),
        ("cm",       "is_central_mid"),
        ("cdm",      "is_cdm"),
        ("fullback", "is_fullback"),
        ("cb",       "is_centre_back"),
        ("gk",       "is_goalkeeper"),
    ]
    age_interaction_cols = []
    for prefix, dummy in age_pos_pairs:
        col_age = f"{prefix}_age"
        col_age_sq = f"{prefix}_age_sq"
        df[col_age]    = df[dummy] * df["age_centered"]
        df[col_age_sq] = df[dummy] * df["age_centered_sq"]
        age_interaction_cols.extend([col_age, col_age_sq])
    print(f"\n  Position-specific age interactions created (centered): {age_interaction_cols}")

    # --- Position-specific performance interaction terms ---
    position_prefixes = [
        ("striker",    "is_striker"),
        ("winger",     "is_winger"),
        ("attmid",     "is_attacking_mid"),
        ("cm",         "is_central_mid"),
        ("cdm",        "is_cdm"),
        ("fullback",   "is_fullback"),
        ("cb",         "is_centre_back"),
        ("gk",         "is_goalkeeper"),
    ]
    interaction_cols = list(age_interaction_cols)
    for prefix, dummy in position_prefixes:
        for stat in ("goals_per_90", "assists_per_90"):
            col = f"{prefix}_{stat}"
            df[col] = df[dummy] * df[stat]
            interaction_cols.append(col)

    # --- WhoScored position-specific interaction terms ---
    ws_stats = [
        "tackles_per_game", "interceptions_per_game", "clearances_per_game",
        "blocks_per_game", "aerials_won", "key_passes_per_game",
        "dribbles_per_game", "crosses_per_game", "long_balls_per_game",
        "through_balls_per_game", "avg_passes_per_game", "shots_per_game",
        "xg", "xg_per_shot", "fouled_per_game",
    ]
    # Short position labels for column names
    ws_pos_prefixes = [
        ("striker",  "is_striker"),
        ("winger",   "is_winger"),
        ("attmid",   "is_attacking_mid"),
        ("cm",       "is_central_mid"),
        ("cdm",      "is_cdm"),
        ("fullback", "is_fullback"),
        ("cb",       "is_centre_back"),
        ("gk",       "is_goalkeeper"),
    ]
    for stat in ws_stats:
        if stat not in df.columns:
            continue
        for pos_prefix, dummy in ws_pos_prefixes:
            # e.g. tackles_striker, tackles_winger ...
            stat_short = stat.replace("_per_game", "").replace("_per_shot", "_p90")
            col = f"{stat_short}_{pos_prefix}"
            df[col] = df[dummy] * pd.to_numeric(df[stat], errors="coerce").fillna(0)
            interaction_cols.append(col)

    print("\n  Position interaction terms (mean | max):")
    for col in interaction_cols:
        print(f"    {col:<40} mean={df[col].mean():.4f}  max={df[col].max():.4f}")

    # --- Select and order the final feature columns ---
    base_cols = [
        "player_name", "club", "nationality", "position",
        "age", "age_squared", "market_value_eur", "log_market_value",
        "minutes_played", "goals", "assists",
        "goals_per_90", "assists_per_90",
        "contract_months_remaining", "team_league_position",
        "is_top4", "is_top6", "is_bottom6", "is_historic_top6", "is_promoted",
    ]
    dummy_cols = expected_position_dummies + expected_nationality_dummies

    # Include any extra columns from cleaned.csv
    other_cols = [c for c in df.columns if c not in base_cols + dummy_cols + interaction_cols
                  and not c.startswith("_") and c not in ("position_group", "nationality_group",
                                                           "fb_player_name", "fb_club", "match_score")]

    final_cols = base_cols + dummy_cols + interaction_cols + other_cols
    final_cols = [c for c in final_cols if c in df.columns]

    df = df[final_cols]

    # Drop rows where essential modelling inputs are null
    essential = ["age", "log_market_value", "minutes_played", "goals_per_90",
                 "assists_per_90", "contract_months_remaining"]
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped:
        print(f"\n  Dropped {dropped} rows with null essential features.")

    print(f"\nFinal feature matrix: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved feature matrix to {OUT_PATH}")
    print(df.describe().to_string())

    # Auto-merge keeper stats so features.csv is always complete
    print("\n  Auto-merging keeper stats...")
    _keeper_script = os.path.join(os.path.dirname(__file__), "..", "scraper", "parse_keeper_stats.py")
    result = subprocess.run(
        [sys.executable, _keeper_script],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  Keeper stats merged successfully.")
    else:
        print(f"  WARNING: Keeper stats merge failed:\n{result.stderr}")


if __name__ == "__main__":
    main()
