"""
pipeline/features_bundesliga.py

Feature engineering for the Bundesliga 2025-26 season model.

Saves:
  data/processed/bundesliga/features.csv
  data/processed/bundesliga/age_mean.json

Run: python pipeline/features_bundesliga.py
"""

import json
import subprocess
import sys
import pandas as pd
import numpy as np
import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_BL_2526 = os.path.join(BASE_DIR, "data", "processed", "bundesliga", "2025-26")
CLEANED_PATH  = os.path.join(_PROC_BL_2526, "cleaned.csv")
OUT_PATH      = os.path.join(_PROC_BL_2526, "features.csv")
AGE_MEAN_PATH = os.path.join(_PROC_BL_2526, "age_mean.json")

# ---------------------------------------------------------------------------
# Historic top Bundesliga clubs (equivalent of PL Big Six)
# All known name variants to handle both WS and TM name forms
# ---------------------------------------------------------------------------
HISTORIC_TOP_CLUBS = {
    "Bayern Munich", "FC Bayern München", "FC Bayern Munich",
    "Borussia Dortmund", "BVB",
    "Bayer 04 Leverkusen", "Bayer Leverkusen",
    "RB Leipzig",
}

# ---------------------------------------------------------------------------
# Bundesliga 2025-26 standings
# ---------------------------------------------------------------------------
LEAGUE_TABLE = {
    "Bayern Munich":            1,
    "FC Bayern München":        1,
    "Borussia Dortmund":        2,
    "RB Leipzig":               3,
    "VfB Stuttgart":            4,
    "TSG 1899 Hoffenheim":      5,
    "Hoffenheim":               5,
    "Bayer 04 Leverkusen":      6,
    "Bayer Leverkusen":         6,
    "Eintracht Frankfurt":      7,
    "SC Freiburg":              8,
    "1.FC Union Berlin":        9,
    "FC Union Berlin":          9,
    "Union Berlin":             9,
    "FC Augsburg":              10,
    "Augsburg":                 10,
    "Hamburger SV":             11,
    "Hamburg":                  11,
    "Borussia Mönchengladbach": 12,
    "Borussia Monchengladbach": 12,
    "Gladbach":                 12,
    "1.FSV Mainz 05":           13,
    "Mainz 05":                 13,
    "Mainz":                    13,
    "1.FC Köln":                14,
    "FC Köln":                  14,
    "Cologne":                  14,
    "SV Werder Bremen":         15,
    "Werder Bremen":            15,
    "Bremen":                   15,
    "FC St. Pauli":             16,
    "St. Pauli":                16,
    "St Pauli":                 16,
    "VfL Wolfsburg":            17,
    "Wolfsburg":                17,
    "1.FC Heidenheim 1846":     18,
    "FC Heidenheim":            18,
    "Heidenheim":               18,
}

_LEAGUE_TABLE_LOWER = {k.lower(): v for k, v in LEAGUE_TABLE.items()}


def get_league_position(club: str) -> int:
    if not isinstance(club, str):
        raise ValueError(f"Club value is not a string: {club!r}")
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
    print("Pipeline Bundesliga — Feature Engineering")
    print("=" * 60)

    print(f"\nLoading {CLEANED_PATH}...")
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8")
    print(f"  Input rows: {len(df)}")

    # ── 500 min filter ────────────────────────────────────────────
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce")
    df = df[df["minutes_played"] >= 500].copy()
    print(f"  After 500 min filter: {len(df)} players")

    # ── Log market value ──────────────────────────────────────────
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
    df = df[df["market_value_eur"] > 0].copy()
    df["log_market_value"] = np.log(df["market_value_eur"])
    print(f"\n  log_market_value range: "
          f"{df['log_market_value'].min():.2f} – {df['log_market_value'].max():.2f}")

    # ── Age squared (reference only — not used in regression) ─────
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_squared"] = df["age"] ** 2

    # ── Team league position ──────────────────────────────────────
    df["team_league_position"] = df["club"].apply(get_league_position)
    print(f"\n  team_league_position distribution:")
    print(df["team_league_position"].value_counts().sort_index().to_string())

    # ── Team tier dummies ─────────────────────────────────────────
    # Top 4:    Bayern, Dortmund, Leipzig, Stuttgart     (pos 1-4)
    # Top 6:    Hoffenheim, Leverkusen                   (pos 5-6)
    # Bottom 6: Mainz, Köln, Bremen, St Pauli,           (pos 13-18)
    #           Wolfsburg, Heidenheim
    # Mid table (7-12): baseline, no dummy
    df["is_top4"]    = (df["team_league_position"] <= 4).astype(int)
    df["is_top6"]    = (df["team_league_position"].isin([5, 6])).astype(int)
    df["is_bottom6"] = (df["team_league_position"] >= 13).astype(int)
    mid_mask = ~df["team_league_position"].isin(
        list(range(1, 7)) + list(range(13, 19))
    )

    print(f"\n  Team tier counts:")
    print(f"    is_top4    = {df['is_top4'].sum()} players  "
          f"(Bayern, Dortmund, Leipzig, Stuttgart)")
    print(f"    is_top6    = {df['is_top6'].sum()} players  "
          f"(Hoffenheim, Leverkusen)")
    print(f"    mid table  = {mid_mask.sum()} players  "
          f"(Frankfurt–Mönchengladbach, baseline)")
    print(f"    is_bottom6 = {df['is_bottom6'].sum()} players  "
          f"(Mainz–Heidenheim)")

    # ── Historic top club dummy ───────────────────────────────────
    df["is_historic_top6"] = df["club"].isin(HISTORIC_TOP_CLUBS).astype(int)
    print(f"  is_historic_top6: {df['is_historic_top6'].sum()} players")

    # ── Promoted clubs dummy (newly promoted to Bundesliga 2025-26)
    PROMOTED_2025_26 = [
        "Hamburger SV", "Hamburg",
        "1.FC Köln", "FC Köln", "Cologne",
        "1.FC Union Berlin", "FC Union Berlin", "Union Berlin",
    ]
    df["is_promoted"] = df["club"].isin(PROMOTED_2025_26).astype(int)
    print(f"  is_promoted: {df['is_promoted'].sum()} players")

    # ── Verify position/nationality dummies exist ─────────────────
    expected_position_dummies = [
        "is_striker", "is_winger", "is_attacking_mid",
        "is_central_mid", "is_cdm", "is_fullback", "is_goalkeeper",
    ]
    expected_nationality_dummies = [
        "is_brazilian", "is_french", "is_english", "is_spanish",
        "is_german", "is_argentinian", "is_portuguese",
        "is_african", "is_asian", "is_south_american_other",
    ]
    for col in expected_position_dummies + expected_nationality_dummies:
        if col not in df.columns:
            df[col] = 0
            print(f"  WARNING: created missing dummy {col} as zeros")

    # ── is_centre_back (for interaction terms, NOT in regression) ─
    pos_sum = sum(df.get(c, 0) for c in [
        "is_striker", "is_winger", "is_attacking_mid", "is_central_mid",
        "is_cdm", "is_fullback", "is_goalkeeper",
    ])
    df["is_centre_back"] = (1 - pos_sum).clip(lower=0)

    # ── Center age ────────────────────────────────────────────────
    age_mean = float(df["age"].mean())
    df["age_centered"]    = df["age"] - age_mean
    df["age_centered_sq"] = df["age_centered"] ** 2
    print(f"\n  Age centering: mean age = {age_mean:.4f}")
    print(f"  age_centered range: "
          f"{df['age_centered'].min():.1f} to {df['age_centered'].max():.1f}")

    with open(AGE_MEAN_PATH, "w", encoding="utf-8") as _f:
        json.dump({"age_mean": age_mean}, _f)
    print(f"  Saved age_mean → {AGE_MEAN_PATH}")

    # ── Position-specific age interactions ────────────────────────
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
    interaction_cols = []
    new_cols = {}
    for prefix, dummy in age_pos_pairs:
        col_age    = f"{prefix}_age"
        col_age_sq = f"{prefix}_age_sq"
        new_cols[col_age]    = df[dummy] * df["age_centered"]
        new_cols[col_age_sq] = df[dummy] * df["age_centered_sq"]
        interaction_cols.extend([col_age, col_age_sq])
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── Goals/assists × position ──────────────────────────────────
    ga_cols = {}
    for prefix, dummy in age_pos_pairs:
        for stat in ("goals_per_90", "assists_per_90"):
            col = f"{prefix}_{stat}"
            ga_cols[col] = df[dummy] * df[stat]
            interaction_cols.append(col)
    df = pd.concat([df, pd.DataFrame(ga_cols, index=df.index)], axis=1)

    # ── WhoScored stat × position interactions ────────────────────
    ws_stats = [
        "tackles_per_game", "interceptions_per_game", "clearances_per_game",
        "blocks_per_game", "aerials_won", "key_passes_per_game",
        "dribbles_per_game", "crosses_per_game", "long_balls_per_game",
        "through_balls_per_game", "avg_passes_per_game", "shots_per_game",
        "xg", "xg_per_shot", "fouled_per_game",
    ]
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
    ws_interaction_cols = {}
    for stat in ws_stats:
        if stat not in df.columns:
            continue
        stat_vals = pd.to_numeric(df[stat], errors="coerce").fillna(0)
        for pos_prefix, dummy in ws_pos_prefixes:
            stat_short = stat.replace("_per_game", "").replace("_per_shot", "_p90")
            col = f"{stat_short}_{pos_prefix}"
            ws_interaction_cols[col] = df[dummy] * stat_vals
            interaction_cols.append(col)
    df = pd.concat([df, pd.DataFrame(ws_interaction_cols, index=df.index)], axis=1)

    # ── Select and order final columns ────────────────────────────
    base_cols = [
        "player_name", "club", "nationality", "position_group",
        "age", "age_squared", "market_value_eur", "log_market_value",
        "minutes_played", "goals", "assists",
        "goals_per_90", "assists_per_90",
        "team_league_position",
        "is_top4", "is_top6", "is_bottom6", "is_historic_top6", "is_promoted",
    ]
    dummy_cols = expected_position_dummies + expected_nationality_dummies

    other_cols = [
        c for c in df.columns
        if c not in base_cols + dummy_cols + interaction_cols
        and not c.startswith("_")
        and c not in (
            "position_group", "nationality_group", "position_group_ws",
            "position_tm", "position_ws", "fb_player_name", "fb_club",
            "match_score", "age_centered", "age_centered_sq",
        )
    ]

    final_cols = base_cols + dummy_cols + interaction_cols + other_cols
    final_cols = [c for c in final_cols if c in df.columns]
    if "is_centre_back" not in final_cols and "is_centre_back" in df.columns:
        final_cols.append("is_centre_back")

    df = df[final_cols]

    essential = [
        "age", "log_market_value", "minutes_played",
        "goals_per_90", "assists_per_90",
    ]
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped:
        print(f"\n  Dropped {dropped} rows with null essential features.")

    print(f"\nFinal feature matrix: {len(df)} rows × {len(df.columns)} columns")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved → {OUT_PATH}")

    print("\nPosition group distribution:")
    if "position_group" in df.columns:
        print(df["position_group"].value_counts().to_string())

    # Auto-merge keeper stats so features.csv is always complete
    print("\n  Auto-merging keeper stats...")
    _keeper_script = os.path.join(
        os.path.dirname(__file__), "..", "scraper",
        "parse_keeper_stats_bundesliga.py"
    )
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
