"""
pipeline/merge_2024.py

Merges two data sources for the 2024-25 season:

  1. whoscored_2024_processed.csv   — PRIMARY: end-of-season stats, goals,
                                      assists, minutes, per-game stats, xG,
                                      blocks, aerials, club, age
  2. transfermarkt_raw.csv          — market value, nationality, TM position
                                      (matched by name only — no club column)

Output: data/processed/2024-25/merged.csv

Note on Football_Player_Data-Analysis2024:20205.csv:
  That file is a mid-season snapshot (Salah: 25G, 2485 mins vs WhoScored's
  29G, 3380 mins end-of-season). Using WhoScored as primary gives better
  alignment with end-of-season market values.

Run: python pipeline/merge_2024.py
"""

import os
import pandas as pd
from rapidfuzz import fuzz, process

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_PL_2425  = os.path.join(BASE_DIR, "data", "raw", "premier_league", "2024-25")
_PROC_PL_2425 = os.path.join(BASE_DIR, "data", "processed", "premier_league", "2024-25")

WS_PATH  = os.path.join(_RAW_PL_2425,  "whoscored.csv")
TM_PATH  = os.path.join(_RAW_PL_2425,  "transfermarkt.csv")
OUT_PATH = os.path.join(_PROC_PL_2425, "merged.csv")

# Higher threshold for name-only matching (no club to disambiguate)
TM_FUZZY_THRESHOLD = 90


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def merge_tm(ws_df: pd.DataFrame, tm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuzzy-match WhoScored players against TM raw by name only.
    Adds: market_value_eur, nationality, position_tm, contract_months_remaining.
    """
    print(f"\nMatching {len(ws_df)} WhoScored players → {len(tm_df)} TM players "
          f"(name-only, threshold {TM_FUZZY_THRESHOLD})...")

    tm = tm_df.copy()
    tm["_name_key"] = tm["player_name"].apply(normalise_name)
    tm_names = tm["_name_key"].tolist()

    market_values   = []
    nationalities   = []
    positions_tm    = []
    contract_months = []
    match_scores    = []
    unmatched       = []

    for _, row in ws_df.iterrows():
        ws_name = normalise_name(row["player_name"])

        result = process.extractOne(
            ws_name,
            tm_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=TM_FUZZY_THRESHOLD,
        )

        if result:
            _best, score, idx = result
            tm_row = tm.iloc[idx]
            market_values.append(tm_row.get("market_value_eur"))
            nationalities.append(tm_row.get("nationality"))
            positions_tm.append(tm_row.get("position"))
            contract_months.append(tm_row.get("contract_months_remaining"))
            match_scores.append(score)
            if score < 95:
                print(f"  [{score:.0f}] '{row['player_name']}' ({row['club']}) "
                      f"→ '{tm_row['player_name']}'")
        else:
            market_values.append(None)
            nationalities.append(None)
            positions_tm.append(None)
            contract_months.append(None)
            match_scores.append(None)
            unmatched.append(f"{row['player_name']} ({row['club']})")

    out = ws_df.copy()
    out["market_value_eur"]          = market_values
    out["nationality"]               = nationalities
    out["position_tm"]               = positions_tm
    out["contract_months_remaining"] = contract_months
    out["tm_match_score"]            = match_scores

    matched = sum(1 for v in market_values if v is not None)
    print(f"\n  TM matched:   {matched} / {len(ws_df)}")
    print(f"  TM unmatched: {len(unmatched)}")
    if unmatched:
        print("  Unmatched (no market value — will be dropped):")
        for p in unmatched[:20]:
            print(f"    {p}")

    return out


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 60)
    print("Pipeline 2024-25 — Merge")
    print("=" * 60)

    if not os.path.exists(WS_PATH):
        print(f"ERROR: Missing {WS_PATH}")
        print("  Run: python scraper/whoscored_parse_2024.py")
        raise SystemExit(1)
    ws_df = pd.read_csv(WS_PATH, encoding="utf-8")
    print(f"\nWhoScored 2024-25: {len(ws_df)} players")

    if not os.path.exists(TM_PATH):
        print(f"ERROR: Missing {TM_PATH}")
        raise SystemExit(1)
    tm_df = pd.read_csv(TM_PATH, encoding="utf-8")
    tm_df = tm_df.dropna(subset=["market_value_eur"])
    print(f"Transfermarkt raw: {len(tm_df)} players (after dropping null values)")

    merged = merge_tm(ws_df, tm_df)

    before = len(merged)
    merged = merged.dropna(subset=["market_value_eur"])
    dropped = before - len(merged)
    if dropped:
        print(f"\nDropped {dropped} players with no TM market value.")
    print(f"After TM merge: {len(merged)} players")

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_PATH}")

    preview_cols = [
        "player_name", "club", "age", "position_tm", "nationality",
        "market_value_eur", "goals", "minutes_played", "rating",
    ]
    print("\nSample rows:")
    print(merged[preview_cols].head(8).to_string(index=False))

    print("\nMarket value distribution:")
    mv = pd.to_numeric(merged["market_value_eur"], errors="coerce")
    print(f"  Min: €{mv.min():,.0f}   Median: €{mv.median():,.0f}   Max: €{mv.max():,.0f}")

    print("\nPosition breakdown (from TM raw):")
    print(merged["position_tm"].value_counts().to_string())


if __name__ == "__main__":
    main()
