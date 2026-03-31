"""
pipeline/merge.py

Merges Transfermarkt and FBref data on (player_name, club).
Uses fuzzy matching for name mismatches with a threshold of 88.

Run: python pipeline/merge.py
"""

import pandas as pd
import os
from rapidfuzz import fuzz, process

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_PL_2526  = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2025-26")
_PROC_PL_2526 = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26")

TM_PATH  = os.path.join(_RAW_PL_2526,  "transfermarkt.csv")
FB_PATH  = os.path.join(_RAW_PL_2526,  "fbref_stats.csv")
WS_PATH  = os.path.join(_RAW_PL_2526,  "whoscored.csv")
OUT_PATH = os.path.join(_PROC_PL_2526, "merged.csv")

FUZZY_THRESHOLD = 88


def normalise_name(name: str) -> str:
    """Lowercase, strip accents-aware normalisation for matching."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def normalise_club(club: str) -> str:
    """Normalise club names for matching — handle common abbreviations."""
    if not isinstance(club, str):
        return ""
    club = club.strip().lower()
    # Common Transfermarkt vs FBref club name discrepancies
    replacements = {
        "manchester city": "manchester city",
        "man city": "manchester city",
        "manchester utd": "manchester united",
        "manchester united": "manchester united",
        "man utd": "manchester united",
        "tottenham hotspur": "tottenham hotspur",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "newcastle united": "newcastle united",
        "newcastle utd": "newcastle united",
        "newcastle": "newcastle united",
        "nottm forest": "nottingham forest",
        "nottingham forest": "nottingham forest",
        "nott'ham forest": "nottingham forest",
        "wolverhampton wanderers": "wolves",
        "wolves": "wolves",
        "west ham united": "west ham",
        "west ham utd": "west ham",
        "west ham": "west ham",
        "brighton & hove albion": "brighton",
        "brighton": "brighton",
        "aston villa": "aston villa",
        "crystal palace": "crystal palace",
        "brentford": "brentford",
        "fulham": "fulham",
        "everton": "everton",
        "chelsea": "chelsea",
        "arsenal": "arsenal",
        "liverpool": "liverpool",
        "leicester city": "leicester",
        "leicester": "leicester",
        "ipswich town": "ipswich",
        "ipswich": "ipswich",
        "southampton": "southampton",
        "bournemouth": "bournemouth",
        "luton town": "luton",
        "luton": "luton",
        "sheffield united": "sheffield utd",
        "sheffield utd": "sheffield utd",
        "burnley": "burnley",
    }
    return replacements.get(club, club)


def fuzzy_match_player(tm_df: pd.DataFrame, fb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match rows between TM and FBref dataframes using:
    1. Exact match on normalised (name, club)
    2. Fuzzy name match within same normalised club
    3. Fuzzy name match cross-club (lower threshold) as last resort

    Returns a merged DataFrame with TM and FBref columns.
    """
    print("Preparing name/club keys...")

    tm = tm_df.copy()
    fb = fb_df.copy()

    tm["_name_key"] = tm["player_name"].apply(normalise_name)
    tm["_club_key"] = tm["club"].apply(normalise_club)

    fb["_name_key"] = fb["player_name"].apply(normalise_name)
    fb["_club_key"] = fb["club"].apply(normalise_club)

    matched_rows = []
    unmatched_tm = []
    unmatched_fb = set(range(len(fb)))

    fb_by_club: dict[str, pd.DataFrame] = {club: grp for club, grp in fb.groupby("_club_key")}

    print(f"Matching {len(tm)} TM players against {len(fb)} FBref players...")

    for _, tm_row in tm.iterrows():
        tm_name = tm_row["_name_key"]
        tm_club = tm_row["_club_key"]

        matched = False

        # --- Pass 1: exact club match, fuzzy name ---
        club_group = fb_by_club.get(tm_club, pd.DataFrame())
        if not club_group.empty:
            fb_names = club_group["_name_key"].tolist()
            result = process.extractOne(
                tm_name,
                fb_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if result:
                best_name, score, idx = result
                fb_idx = club_group.index[idx]
                fb_row = fb.loc[fb_idx]
                print(f"  MATCH [{score:.0f}]: '{tm_row['player_name']}' ({tm_row['club']}) "
                      f"<-> '{fb_row['player_name']}' ({fb_row['club']})")
                merged = {**tm_row.to_dict(), **{
                    "fb_player_name": fb_row["player_name"],
                    "fb_club": fb_row["club"],
                    "minutes_played": fb_row.get("minutes_played"),
                    "goals": fb_row.get("goals"),
                    "assists": fb_row.get("assists"),
                    "goals_per_90": fb_row.get("goals_per_90"),
                    "assists_per_90": fb_row.get("assists_per_90"),
                    "match_score": score,
                }}
                matched_rows.append(merged)
                unmatched_fb.discard(fb_idx)
                matched = True

        # --- Pass 2: cross-club fuzzy (handles loan moves / team changes) ---
        if not matched:
            all_fb_names = fb["_name_key"].tolist()
            result = process.extractOne(
                tm_name,
                all_fb_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=92,  # Stricter threshold for cross-club
            )
            if result:
                best_name, score, idx = result
                fb_row = fb.iloc[idx]
                print(f"  CROSS-CLUB MATCH [{score:.0f}]: '{tm_row['player_name']}' ({tm_row['club']}) "
                      f"<-> '{fb_row['player_name']}' ({fb_row['club']})")
                merged = {**tm_row.to_dict(), **{
                    "fb_player_name": fb_row["player_name"],
                    "fb_club": fb_row["club"],
                    "minutes_played": fb_row.get("minutes_played"),
                    "goals": fb_row.get("goals"),
                    "assists": fb_row.get("assists"),
                    "goals_per_90": fb_row.get("goals_per_90"),
                    "assists_per_90": fb_row.get("assists_per_90"),
                    "match_score": score,
                }}
                matched_rows.append(merged)
                unmatched_fb.discard(fb.index[idx])
                matched = True

        if not matched:
            print(f"  NO MATCH: '{tm_row['player_name']}' ({tm_row['club']})")
            unmatched_tm.append(tm_row["player_name"])

    print(f"\nMatched: {len(matched_rows)} players")
    print(f"Unmatched TM players: {len(unmatched_tm)}")
    print(f"Unmatched FBref players (below minutes threshold or no TM entry): {len(unmatched_fb)}")

    result_df = pd.DataFrame(matched_rows)

    # Drop internal key columns
    result_df = result_df.drop(columns=["_name_key", "_club_key"], errors="ignore")

    return result_df


def merge_positions_csv(merged_df, positions_path, threshold=90):
    """Supplement position_tm from transfermarkt_positions.csv via fuzzy name matching.

    The positions CSV is the primary source; fall back to existing position column
    for players with no match.
    """
    if not os.path.exists(positions_path):
        print(f"  Positions CSV not found at {positions_path} — skipping.")
        return merged_df

    pos_df = pd.read_csv(positions_path, encoding="utf-8")
    if pos_df.empty or "position_tm" not in pos_df.columns:
        print(f"  Positions CSV empty or missing position_tm column — skipping.")
        return merged_df

    pos_df["_name_key"] = pos_df["player_name"].apply(normalise_name)
    pos_names = pos_df["_name_key"].tolist()

    new_positions = []
    matched_count = 0
    for _, row in merged_df.iterrows():
        name_key = normalise_name(row.get("player_name", ""))
        result = process.extractOne(
            name_key, pos_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            _, _, idx = result
            new_positions.append(pos_df.iloc[idx]["position_tm"])
            matched_count += 1
        else:
            # Fall back to existing position column
            new_positions.append(row.get("position", row.get("position_tm", "")))

    merged_df = merged_df.copy()
    merged_df["position_tm"] = new_positions
    print(f"  Positions matched: {matched_count} / {len(merged_df)}")
    return merged_df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 60)
    print("Pipeline — Merge Transfermarkt + FBref")
    print("=" * 60)

    print(f"\nLoading {TM_PATH}...")
    tm_df = pd.read_csv(TM_PATH, encoding="utf-8")
    print(f"  Transfermarkt rows: {len(tm_df)}")

    print(f"\nLoading {FB_PATH}...")
    fb_df = pd.read_csv(FB_PATH, encoding="utf-8")
    print(f"  FBref rows: {len(fb_df)}")

    # Drop TM players with no market value
    tm_df = tm_df.dropna(subset=["market_value_eur"])
    print(f"  TM rows after dropping null market values: {len(tm_df)}")

    merged = fuzzy_match_player(tm_df, fb_df)

    # Drop rows with missing key stats
    before = len(merged)
    merged = merged.dropna(subset=["market_value_eur", "minutes_played"])
    print(f"\nDropped {before - len(merged)} rows with null market_value or minutes.")
    print(f"Final merged rows: {len(merged)}")

    # ── Third merge: WhoScored stats (left join) ────────────────────────────
    if os.path.exists(WS_PATH):
        print(f"\nLoading {WS_PATH}...")
        ws_df = pd.read_csv(WS_PATH, encoding="utf-8")
        print(f"  WhoScored rows: {len(ws_df)}")

        ws_names = ws_df["player_name"].tolist()
        ws_matched = 0

        # Columns to bring in from WhoScored (exclude duplicates already in merged)
        ws_stat_cols = [c for c in ws_df.columns
                        if c not in ("player_name", "club", "age", "position",
                                     "apps", "goals", "assists")]

        for col in ws_stat_cols:
            merged[col] = None

        for idx, row in merged.iterrows():
            result = process.extractOne(
                normalise_name(row["player_name"]),
                [normalise_name(n) for n in ws_names],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=85,
            )
            if result is None:
                continue
            _, score, ws_idx = result
            ws_row = ws_df.iloc[ws_idx]
            for col in ws_stat_cols:
                merged.at[idx, col] = ws_row.get(col)
            ws_matched += 1

        print(f"  WhoScored players matched: {ws_matched} / {len(merged)}")
    else:
        print(f"\n  WARNING: {WS_PATH} not found — skipping WhoScored merge.")

    # ── Supplement position_tm from dedicated positions CSV ─────────────────
    pos_path = os.path.join(_RAW_PL_2526, "transfermarkt_positions.csv")
    if not os.path.exists(pos_path):
        # Try 2024-25 as fallback
        pos_path = os.path.join(
            os.path.dirname(_RAW_PL_2526),
            "2024-25", "transfermarkt_positions.csv"
        )
    print(f"\nMerging positions from positions CSV...")
    merged = merge_positions_csv(merged, pos_path)

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved merged data to {OUT_PATH}")
    print(merged.head(5).to_string())


if __name__ == "__main__":
    main()
