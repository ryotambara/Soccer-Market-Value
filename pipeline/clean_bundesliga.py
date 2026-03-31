"""
pipeline/clean_bundesliga.py

Normalises positions and nationalities, creates dummy variables,
and saves cleaned data for the Bundesliga 2025-26 season model.

Position priority:
  1. TM position (position_tm) — authoritative, uses POSITION_MAP
  2. WhoScored position group (position_group_ws) — fallback

Run: python pipeline/clean_bundesliga.py
"""

import pandas as pd
import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_BL_2526 = os.path.join(BASE_DIR, "data", "processed", "bundesliga", "2025-26")
MERGED_PATH   = os.path.join(_PROC_BL_2526, "merged.csv")
OUT_PATH      = os.path.join(_PROC_BL_2526, "cleaned.csv")

# ---------------------------------------------------------------------------
# Regression dummy mapping — Transfermarkt labels → model dummy groups.
# position_tm (raw TM label) is kept separately for display.
# Centre-Back is baseline (no dummy created).
# ---------------------------------------------------------------------------
REGRESSION_MAP = {
    "Centre-Forward":     "striker",
    "Second Striker":     "striker",
    "Left Winger":        "winger",
    "Right Winger":       "winger",
    "Left Midfield":      "winger",
    "Right Midfield":     "winger",
    "Attacking Midfield": "attacking_mid",
    "Central Midfield":   "central_mid",
    "Defensive Midfield": "cdm",
    "Left-Back":          "fullback",
    "Right-Back":         "fullback",
    "Left Wing-Back":     "fullback",
    "Right Wing-Back":    "fullback",
    "Goalkeeper":         "goalkeeper",
    "Centre-Back":        "centre_back",
}

POSITION_DUMMIES = [
    "striker", "winger", "attacking_mid", "central_mid",
    "cdm", "fullback", "goalkeeper",
]
# "centre_back" is the reference group — intentionally omitted

# ---------------------------------------------------------------------------
# Nationality mapping — identical to PL version
# ---------------------------------------------------------------------------
AFRICAN_NATIONS = {
    "Senegal", "Nigeria", "Ghana", "Ivory Coast", "Cameroon", "Egypt",
    "Algeria", "Morocco", "Tunisia", "Mali", "Guinea",
    "Democratic Republic of Congo", "Congo DR", "Gabon", "South Africa",
    "Sierra Leone", "Benin", "Burkina Faso", "Ethiopia", "Kenya",
    "Tanzania", "Uganda", "Zimbabwe", "Zambia", "Angola", "Cape Verde",
    "Gambia", "Guinea-Bissau", "Liberia", "Mozambique", "Namibia",
    "Sudan", "Togo",
}

ASIAN_NATIONS = {
    "Japan", "South Korea", "China", "Iran", "Saudi Arabia", "Qatar",
    "United Arab Emirates", "Iraq", "Syria", "Lebanon", "Jordan",
    "Indonesia", "Australia", "Thailand", "Vietnam", "India", "Pakistan",
    "Uzbekistan", "Kazakhstan", "Israel",
}

SOUTH_AMERICAN_OTHER = {
    "Colombia", "Chile", "Peru", "Ecuador", "Uruguay", "Venezuela",
    "Paraguay", "Bolivia",
}


def map_nationality(nat: str) -> str:
    if not isinstance(nat, str) or nat.strip() == "":
        return "other_europe"
    nat = nat.strip()
    nat_lower = nat.lower()
    if nat_lower in ("brazil", "brasil"):
        return "brazilian"
    if nat_lower == "france":
        return "french"
    if nat_lower in ("england", "united kingdom", "british"):
        return "english"
    if nat_lower in ("spain", "españa"):
        return "spanish"
    if nat_lower in ("germany", "deutschland"):
        return "german"
    if nat_lower == "argentina":
        return "argentinian"
    if nat_lower == "portugal":
        return "portuguese"
    if nat in AFRICAN_NATIONS or nat_lower in {n.lower() for n in AFRICAN_NATIONS}:
        return "african"
    if nat in ASIAN_NATIONS or nat_lower in {n.lower() for n in ASIAN_NATIONS}:
        return "asian"
    if nat in SOUTH_AMERICAN_OTHER or nat_lower in {n.lower() for n in SOUTH_AMERICAN_OTHER}:
        return "south_american_other"
    return "other_europe"


NATIONALITY_DUMMIES = [
    "brazilian", "french", "english", "spanish", "german",
    "argentinian", "portuguese", "african", "asian", "south_american_other",
]


def resolve_position_group(row: pd.Series) -> str:
    tm_pos = str(row.get("position_tm", "")).strip()
    if tm_pos and tm_pos != "nan":
        group = REGRESSION_MAP.get(tm_pos)
        if group:
            return group

    ws_group = str(row.get("position_group_ws", "")).strip()
    if ws_group and ws_group != "nan":
        return ws_group

    return "centre_back"


def apply_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df["position_group"] = df.apply(resolve_position_group, axis=1)

    known = set(POSITION_DUMMIES) | {"centre_back"}
    bad = df[~df["position_group"].isin(known)]
    if not bad.empty:
        print(f"  WARNING: {len(bad)} players with unrecognised position_group:")
        print(bad[["player_name", "position_tm", "position_group_ws",
                   "position_group"]].to_string())
        df.loc[~df["position_group"].isin(known), "position_group"] = "centre_back"

    for group in POSITION_DUMMIES:
        col = f"is_{group}"
        df[col] = (df["position_group"] == group).astype(int)
        print(f"  {col}: {df[col].sum()} players")

    return df


def apply_nationality_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df["nationality_group"] = df["nationality"].apply(map_nationality)

    for group in NATIONALITY_DUMMIES:
        col = f"is_{group}"
        df[col] = (df["nationality_group"] == group).astype(int)
        print(f"  {col}: {df[col].sum()} players")

    baseline_count = (df["nationality_group"] == "other_europe").sum()
    print(f"  baseline (other_europe): {baseline_count} players")
    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 60)
    print("Pipeline Bundesliga — Clean & Normalise")
    print("=" * 60)

    print(f"\nLoading {MERGED_PATH}...")
    df = pd.read_csv(MERGED_PATH, encoding="utf-8")
    print(f"  Input rows: {len(df)}")

    before = len(df)
    df = df.dropna(subset=["market_value_eur"])
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with null market_value_eur.")

    # ── Position dummies ──────────────────────────────────────────
    print(f"\nApplying position mapping...")
    df = apply_position_dummies(df)

    print("\n  Position source breakdown:")
    has_tm = df["position_tm"].notna() & (df["position_tm"].astype(str) != "nan")
    print(f"    TM position used:        {has_tm.sum()}")
    print(f"    WhoScored fallback used: {(~has_tm).sum()}")

    # ── Nationality dummies ───────────────────────────────────────
    print(f"\nApplying nationality mapping...")
    df = apply_nationality_dummies(df)

    # ── Numeric coercion ──────────────────────────────────────────
    numeric_cols = [
        "market_value_eur", "age", "minutes_played",
        "goals", "assists", "goals_per_90", "assists_per_90",
        "contract_months_remaining",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["goals_per_90"]   = df["goals_per_90"].fillna(0.0)
    df["assists_per_90"] = df["assists_per_90"].fillna(0.0)

    # ── Contract months: fill missing with median ─────────────────
    if "contract_months_remaining" in df.columns:
        df["contract_months_remaining"] = pd.to_numeric(
            df["contract_months_remaining"], errors="coerce"
        )
        median_contract = df["contract_months_remaining"].median()
        if pd.isna(median_contract):
            median_contract = 18.0
            print(f"\n  contract_months_remaining entirely missing — "
                  f"using fixed value of 18 months.")
        else:
            missing_contract = df["contract_months_remaining"].isna().sum()
            if missing_contract > 0:
                print(f"\n  Filling {missing_contract} missing contract months "
                      f"with median ({median_contract:.0f}).")
        df["contract_months_remaining"] = (
            df["contract_months_remaining"].fillna(median_contract).astype(int)
        )
    else:
        df["contract_months_remaining"] = 18

    # ── Fill stat NaN with 0 ──────────────────────────────────────
    stat_cols = [
        "rating", "tackles_per_game", "interceptions_per_game",
        "fouls_per_game", "clearances_per_game", "dribbled_past_per_game",
        "shots_per_game", "key_passes_per_game", "dribbles_per_game",
        "fouled_per_game", "avg_passes_per_game", "pass_success_pct",
        "crosses_per_game", "long_balls_per_game", "through_balls_per_game",
        "xg", "xg_per_shot", "xg_p90",
        "blocks_per_game", "aerials_won", "xg_diff", "yellow_cards",
    ]
    filled = 0
    for col in stat_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            filled += n
    print(f"\n  Filled {filled} NaN values across stat columns with 0.")

    if "xg_diff" in df.columns:
        still_zero = (df["xg_diff"] == 0.0).sum()
        if still_zero > len(df) * 0.5:
            print(f"  NOTE: xg_diff was 0 for {still_zero} players — "
                  f"computing from goals − xg.")
            df["xg_diff"] = (
                pd.to_numeric(df["goals"], errors="coerce") -
                pd.to_numeric(df["xg"], errors="coerce")
            ).fillna(0.0)
    else:
        df["xg_diff"] = (
            pd.to_numeric(df["goals"], errors="coerce") -
            pd.to_numeric(df["xg"], errors="coerce")
        ).fillna(0.0)

    print(f"\nFinal cleaned rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved cleaned data to {OUT_PATH}")


if __name__ == "__main__":
    main()
