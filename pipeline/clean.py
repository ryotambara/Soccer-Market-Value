"""
pipeline/clean.py

Normalises positions and nationalities, creates dummy variables,
and saves cleaned data.

Run: python pipeline/clean.py
"""

import pandas as pd
import os

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_PL_2526 = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26")
MERGED_PATH  = os.path.join(_PROC_PL_2526, "merged.csv")
OUT_PATH     = os.path.join(_PROC_PL_2526, "cleaned.csv")

# ---------------------------------------------------------------------------
# Position mapping — raw Transfermarkt labels → standardised groups
# Centre-Back is the baseline (no dummy created)
# ---------------------------------------------------------------------------
POSITION_MAP = {
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
    "Centre-Back":        "centre_back",   # baseline — no dummy
}

POSITION_DUMMIES = ["striker", "winger", "attacking_mid", "central_mid",
                    "cdm", "fullback", "goalkeeper"]
# "centre_back" is the reference group — intentionally omitted

# ---------------------------------------------------------------------------
# Nationality mapping
# Baseline = "other_europe" — no dummy created for this group
# ---------------------------------------------------------------------------
AFRICAN_NATIONS = {
    "Senegal", "Nigeria", "Ghana", "Ivory Coast", "Cameroon", "Egypt",
    "Algeria", "Morocco", "Tunisia", "Mali", "Guinea", "Democratic Republic of Congo",
    "Congo DR", "Gabon", "South Africa", "Sierra Leone", "Benin", "Burkina Faso",
    "Ethiopia", "Kenya", "Tanzania", "Uganda", "Zimbabwe", "Zambia", "Angola",
    "Cape Verde", "Gambia", "Guinea-Bissau", "Liberia", "Mozambique",
    "Namibia", "Sudan", "Togo",
}

ASIAN_NATIONS = {
    "Japan", "South Korea", "China", "Iran", "Saudi Arabia", "Qatar",
    "United Arab Emirates", "Iraq", "Syria", "Lebanon", "Jordan",
    "Indonesia", "Australia",  # Australia in AFC
    "Thailand", "Vietnam", "India", "Pakistan", "Uzbekistan", "Kazakhstan",
    "Israel",  # Historically grouped with Asia in football
}

SOUTH_AMERICAN_OTHER = {
    "Colombia", "Chile", "Peru", "Ecuador", "Uruguay", "Venezuela",
    "Paraguay", "Bolivia",
}


def map_nationality(nat: str) -> str:
    """Map a raw nationality string to our standardised nationality group."""
    if not isinstance(nat, str) or nat.strip() == "":
        return "other_europe"

    nat = nat.strip()

    nat_lower = nat.lower()
    if nat_lower in ("brazil", "brasil"):
        return "brazilian"
    if nat_lower in ("france",):
        return "french"
    if nat_lower in ("england", "united kingdom", "british"):
        return "english"
    if nat_lower in ("spain", "españa"):
        return "spanish"
    if nat_lower in ("germany", "deutschland"):
        return "german"
    if nat_lower in ("argentina",):
        return "argentinian"
    if nat_lower in ("portugal",):
        return "portuguese"

    if nat in AFRICAN_NATIONS or nat_lower in {n.lower() for n in AFRICAN_NATIONS}:
        return "african"
    if nat in ASIAN_NATIONS or nat_lower in {n.lower() for n in ASIAN_NATIONS}:
        return "asian"
    if nat in SOUTH_AMERICAN_OTHER or nat_lower in {n.lower() for n in SOUTH_AMERICAN_OTHER}:
        return "south_american_other"

    # Default baseline group
    return "other_europe"


NATIONALITY_DUMMIES = [
    "brazilian", "french", "english", "spanish", "german",
    "argentinian", "portuguese", "african", "asian", "south_american_other",
]
# "other_europe" is the reference group — intentionally omitted


def apply_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw position column to position group, then create dummy columns."""
    df["position_group"] = df["position"].map(POSITION_MAP)

    # For unknown positions, attempt a partial match
    unknown_mask = df["position_group"].isna()
    if unknown_mask.any():
        print(f"  WARNING: {unknown_mask.sum()} players have unrecognised positions:")
        print(df.loc[unknown_mask, ["player_name", "position"]].to_string())
        # Default unknown to centre_back (baseline) so they're still usable
        df.loc[unknown_mask, "position_group"] = "centre_back"

    for group in POSITION_DUMMIES:
        col = f"is_{group}"
        df[col] = (df["position_group"] == group).astype(int)
        count = df[col].sum()
        print(f"  {col}: {count} players")

    return df


def apply_nationality_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Map nationality to group and create dummy columns."""
    df["nationality_group"] = df["nationality"].apply(map_nationality)

    for group in NATIONALITY_DUMMIES:
        col = f"is_{group}"
        df[col] = (df["nationality_group"] == group).astype(int)
        count = df[col].sum()
        print(f"  {col}: {count} players")

    baseline_count = (df["nationality_group"] == "other_europe").sum()
    print(f"  baseline (other_europe): {baseline_count} players")

    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 60)
    print("Pipeline — Clean & Normalise")
    print("=" * 60)

    print(f"\nLoading {MERGED_PATH}...")
    df = pd.read_csv(MERGED_PATH, encoding="utf-8")
    print(f"  Input rows: {len(df)}")

    # Drop rows with null market values (should already be done, but defensive)
    before = len(df)
    df = df.dropna(subset=["market_value_eur"])
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with null market_value_eur.")

    print(f"\nApplying position mapping...")
    df = apply_position_dummies(df)

    print(f"\nApplying nationality mapping...")
    df = apply_nationality_dummies(df)

    # Ensure numeric types for key columns
    numeric_cols = [
        "market_value_eur", "age", "minutes_played",
        "goals", "assists", "goals_per_90", "assists_per_90",
        "contract_months_remaining",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill per-90 NaN with 0 for goalkeepers/very low minute players
    df["goals_per_90"] = df["goals_per_90"].fillna(0.0)
    df["assists_per_90"] = df["assists_per_90"].fillna(0.0)

    # Fill missing contract months with median
    median_contract = df["contract_months_remaining"].median()
    missing_contract = df["contract_months_remaining"].isna().sum()
    if missing_contract > 0:
        print(f"\n  Filling {missing_contract} missing contract months with median ({median_contract:.0f}).")
        df["contract_months_remaining"] = df["contract_months_remaining"].fillna(median_contract)

    df["contract_months_remaining"] = df["contract_months_remaining"].astype(int)

    # Fill NaN in WhoScored columns with 0
    ws_cols = [
        "rating", "tackles_per_game", "interceptions_per_game",
        "fouls_per_game", "clearances_per_game", "dribbled_past_per_game",
        "blocks_per_game", "key_passes_per_game", "dribbles_per_game",
        "fouled_per_game", "avg_passes_per_game", "crosses_per_game",
        "long_balls_per_game", "through_balls_per_game",
        "xg", "xg_diff", "xg_per_shot",
        "aerials_won", "shots_per_game", "pass_success_pct", "yellow_cards",
    ]
    filled_ws = 0
    for col in ws_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            filled_ws += n
    print(f"\n  Filled {filled_ws} NaN values across WhoScored columns with 0.")

    print(f"\nFinal cleaned rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved cleaned data to {OUT_PATH}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
