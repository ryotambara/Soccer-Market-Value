"""
scraper/parse_keeper_stats.py

Parses goalkeeper stats CSVs for both seasons, fuzzy-matches players into
the existing features files, and adds GK-specific interaction columns.

Input files:
  data/raw/keepers_2025-26.csv
  data/raw/keepers_2024-25.csv

Output: updates in-place
  data/processed/features.csv
  data/processed/2024-25/features.csv

Run: python scraper/parse_keeper_stats.py
"""

import os
import re
import pandas as pd
from rapidfuzz import fuzz, process
from typing import Optional, List, Tuple

BASE = os.path.dirname(__file__)
RAW_DIR  = os.path.join(BASE, "..", "data", "raw")
PROC_DIR = os.path.join(BASE, "..", "data", "processed")

SEASONS = [
    {
        "label":        "2025-26",
        "raw_path":     os.path.join(RAW_DIR,  "keepers_2025-26.csv"),
        "features_path": os.path.join(PROC_DIR, "features.csv"),
    },
    {
        "label":        "2024-25",
        "raw_path":     os.path.join(RAW_DIR,  "keepers_2024-25.csv"),
        "features_path": os.path.join(PROC_DIR, "2024-25", "features.csv"),
    },
]

FUZZY_THRESHOLD = 85

# GK interaction columns written into the features file
GK_INTERACTION_COLS = [
    "gk_save_pct",
    "gk_cs_per_90",
    "gk_ga_per_90",
    "gk_sota_per_90",
    "gk_pk_save_pct",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean_float(val) -> float:
    s = str(val).strip().replace(",", "")
    if s in ("", "-", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_age(val) -> Optional[int]:
    """Handle both '33-169' (FBRef days format) and plain '31'."""
    s = str(val).strip()
    if not s or s == "nan":
        return None
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _parse_nation(val) -> str:
    """'br BRA' → 'BRA',  'ENG' → 'ENG'"""
    s = str(val).strip()
    # Take the last whitespace-separated token if it's 2-3 upper-case letters
    parts = s.split()
    for part in reversed(parts):
        if re.match(r"^[A-Z]{2,3}$", part):
            return part
    return s


def parse_keeper_csv(path: str) -> pd.DataFrame:
    """
    Parse a keeper CSV (row 1 = section headers, row 2 = column headers, row 3+ = data).
    Returns a tidy DataFrame with one row per keeper.
    """
    # Read with the second row as header (skiprows=1 skips the section header)
    raw = pd.read_csv(path, skiprows=1, dtype=str)

    records: List[dict] = []

    for _, row in raw.iterrows():
        rk = str(row.get("Rk", "")).strip()
        # Skip sub-headers repeated mid-table or blank rows
        if not rk or not rk.isdigit():
            continue

        player = str(row.get("Player", "")).strip()
        if not player or player == "Player":
            continue

        squad  = str(row.get("Squad", "")).strip()
        nation = _parse_nation(row.get("Nation", ""))
        age    = _parse_age(row.get("Age", ""))

        mins_90s = _clean_float(row.get("90s", 0))
        ga       = _clean_float(row.get("GA", 0))
        ga90     = _clean_float(row.get("GA90", 0))
        sota     = _clean_float(row.get("SoTA", 0))
        save_pct = _clean_float(row.get("Save%", 0))
        cs       = _clean_float(row.get("CS", 0))
        pk_att   = _clean_float(row.get("PKatt", 0))
        pk_sv    = _clean_float(row.get("PKsv", 0))
        minutes  = int(_clean_float(row.get("Min", 0)))

        cs_per_90   = cs   / mins_90s if mins_90s > 0 else 0.0
        sota_per_90 = sota / mins_90s if mins_90s > 0 else 0.0
        pk_save_pct = pk_sv / pk_att  if pk_att  > 0 else 0.0

        records.append({
            "player_name": player,
            "club":        squad,
            "nation":      nation,
            "age":         age,
            "save_pct":    save_pct,
            "cs_per_90":   round(cs_per_90,   4),
            "ga_per_90":   ga90,
            "sota_per_90": round(sota_per_90, 4),
            "pk_save_pct": round(pk_save_pct, 4),
            "minutes_gk":  minutes,
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["player_name"])
    return df


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def fuzzy_merge_keepers(
    features: pd.DataFrame,
    keepers: pd.DataFrame,
    threshold: int = FUZZY_THRESHOLD,
) -> Tuple[pd.DataFrame, int]:
    """
    Left-join keeper stats into features by fuzzy player_name match.
    Returns (updated_features, n_matched).
    """
    keeper_names = keepers["player_name"].tolist()

    # Initialise GK stat columns to NaN (filled to 0 after merge for non-GKs)
    stat_cols = ["save_pct", "cs_per_90", "ga_per_90", "sota_per_90",
                 "pk_save_pct", "minutes_gk"]
    for col in stat_cols:
        features[col] = float("nan")

    matched = 0
    for i, feat_row in features.iterrows():
        feat_name = str(feat_row["player_name"]).strip()

        result = process.extractOne(
            feat_name,
            keeper_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            continue

        _best, score, idx = result
        krow = keepers.iloc[idx]
        for col in stat_cols:
            features.at[i, col] = krow[col]
        matched += 1

    return features, matched


# ---------------------------------------------------------------------------
# GK interaction columns
# ---------------------------------------------------------------------------

def add_gk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute is_goalkeeper × keeper_stat interaction columns.
    Non-GK players (and GKs with no matched stats) get 0.
    """
    gk = df["is_goalkeeper"].astype(float)

    df["gk_save_pct"]    = (gk * df["save_pct"].fillna(0.0)).round(4)
    df["gk_cs_per_90"]   = (gk * df["cs_per_90"].fillna(0.0)).round(4)
    df["gk_ga_per_90"]   = (gk * df["ga_per_90"].fillna(0.0)).round(4)
    df["gk_sota_per_90"] = (gk * df["sota_per_90"].fillna(0.0)).round(4)
    df["gk_pk_save_pct"] = (gk * df["pk_save_pct"].fillna(0.0)).round(4)

    # Fill raw stat columns to 0 for non-GKs
    for col in ["save_pct", "cs_per_90", "ga_per_90", "sota_per_90",
                "pk_save_pct", "minutes_gk"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_season(season: dict) -> None:
    label        = season["label"]
    raw_path     = season["raw_path"]
    feat_path    = season["features_path"]

    print(f"\n{'=' * 60}")
    print(f"Season {label}")
    print(f"{'=' * 60}")

    if not os.path.exists(raw_path):
        print(f"  ERROR: Keeper CSV not found: {raw_path}")
        return
    if not os.path.exists(feat_path):
        print(f"  ERROR: Features file not found: {feat_path}")
        return

    # ── Parse keepers ───────────────────────────────────────────────────────
    keepers = parse_keeper_csv(raw_path)
    print(f"\n  Total keepers parsed: {len(keepers)}")

    # Verification: Alisson
    alisson = keepers[keepers["player_name"].str.contains("Alisson", na=False)]
    if not alisson.empty:
        print(f"\n  Alisson stats:")
        print(alisson[["player_name", "club", "save_pct", "cs_per_90",
                        "ga_per_90", "sota_per_90", "pk_save_pct",
                        "minutes_gk"]].to_string(index=False))
    else:
        print("  Alisson: not found in keeper CSV.")

    # ── Load features ────────────────────────────────────────────────────────
    features = pd.read_csv(feat_path, encoding="utf-8")
    print(f"\n  Features rows: {len(features)}  "
          f"GK rows: {(features['is_goalkeeper'] == 1).sum()}")

    # Drop stale GK interaction columns if re-running
    stale = ["gk_goals_per_90", "gk_assists_per_90"] + GK_INTERACTION_COLS + \
            ["save_pct", "cs_per_90", "ga_per_90", "sota_per_90",
             "pk_save_pct", "minutes_gk"]
    features = features.drop(columns=[c for c in stale if c in features.columns])

    # ── Fuzzy merge ──────────────────────────────────────────────────────────
    print(f"\n  Fuzzy-matching keepers into features (threshold={FUZZY_THRESHOLD})...")
    features, n_matched = fuzzy_merge_keepers(features, keepers)
    print(f"  Keepers matched to features: {n_matched} / {len(keepers)}")

    # ── GK interactions ──────────────────────────────────────────────────────
    features = add_gk_interactions(features)

    # ── Save ─────────────────────────────────────────────────────────────────
    features.to_csv(feat_path, index=False, encoding="utf-8")
    print(f"\n  Saved updated features → {feat_path}")
    print(f"  New columns added: {GK_INTERACTION_COLS}")

    # Sanity check: show GK interaction values for a few keepers
    gk_rows = features[features["is_goalkeeper"] == 1][
        ["player_name", "club", "gk_save_pct", "gk_cs_per_90",
         "gk_ga_per_90", "gk_pk_save_pct"]
    ].head(6)
    print(f"\n  Sample GK interaction values:")
    print(gk_rows.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("Goalkeeper Stats Parser — both seasons")
    print("=" * 60)

    for season in SEASONS:
        process_season(season)

    print(f"\n{'=' * 60}")
    print("Done. Run regression scripts to use the new GK features.")
    print("  python model/regression.py")
    print("  python model/regression_2024.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
