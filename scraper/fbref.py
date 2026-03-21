"""
scraper/fbref.py

Scrapes 2024-25 Premier League standard stats from FBref.
Extracts: player_name, club, minutes_played, goals, assists,
          goals_per_90, assists_per_90

Run: python scraper/fbref.py
"""

import requests
import pandas as pd
import time
import random
import os
import io

FBREF_URL = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://fbref.com/",
}

MIN_MINUTES = 500
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2025-26", "fbref_raw.csv")


def fetch_fbref_table(url: str) -> pd.DataFrame:
    """Fetch the FBref standard stats HTML and parse the main player table."""
    print(f"Fetching: {url}")
    time.sleep(random.uniform(2, 3))

    session = requests.Session()
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    print("  Parsing HTML tables...")
    # FBref standard stats table has id="stats_standard"
    # pandas read_html can parse it directly
    try:
        tables = pd.read_html(io.StringIO(response.text), attrs={"id": "stats_standard"})
        if tables:
            df = tables[0]
            print(f"  Found table with shape: {df.shape}")
            return df, response.text
    except ValueError:
        print("  read_html with id attr failed, trying all tables...")

    # Fallback: read all tables and find the right one
    try:
        tables = pd.read_html(io.StringIO(response.text))
        print(f"  Found {len(tables)} tables total.")
        for i, t in enumerate(tables):
            if "Player" in str(t.columns.tolist()) or "Gls" in str(t.columns.tolist()):
                print(f"  Using table index {i} (shape {t.shape})")
                return t, response.text
    except Exception as e:
        print(f"  ERROR: {e}")

    return pd.DataFrame(), response.text


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FBref tables often have multi-level column headers.
    Flatten them and identify the columns we need.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Join level 0 and level 1 — e.g. ('Performance', 'Gls') → 'Performance_Gls'
        df.columns = ["_".join(col).strip() if col[0] != col[1] else col[1]
                      for col in df.columns.values]
        print("  Flattened MultiIndex columns.")

    # Rename columns to standardised names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "player" in col_lower and "player_name" not in col_map.values():
            col_map[col] = "player_name"
        elif col_lower in ("squad", "team", "club") and "club" not in col_map.values():
            col_map[col] = "club"
        elif col_lower in ("min", "minutes", "mp_min", "playing time_min") and "minutes_played" not in col_map.values():
            col_map[col] = "minutes_played"
        elif col_lower in ("90s", "90s_playing time", "playing time_90s") and "90s" not in col_map.values():
            col_map[col] = "90s"
        elif col_lower in ("gls", "performance_gls", "goals") and "goals" not in col_map.values():
            col_map[col] = "goals"
        elif col_lower in ("ast", "performance_ast", "assists") and "assists" not in col_map.values():
            col_map[col] = "assists"

    df = df.rename(columns=col_map)
    print(f"  Columns after normalisation: {list(df.columns)}")
    return df


def clean_fbref(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw FBref dataframe into the format we need."""
    df = normalise_columns(df)

    # Drop header-repeat rows (FBref inserts the header row mid-table)
    if "player_name" in df.columns:
        df = df[df["player_name"] != "Player"]
        df = df[df["player_name"].notna()]
        df = df[df["player_name"].str.strip() != ""]

    # Drop 'Did not play' / summary rows
    for col in ["goals", "assists", "minutes_played", "90s"]:
        if col in df.columns:
            df = df[df[col].notna()]

    # Compute minutes_played from 90s column if direct minutes not available
    if "minutes_played" not in df.columns and "90s" in df.columns:
        df["90s"] = pd.to_numeric(df["90s"], errors="coerce")
        df["minutes_played"] = (df["90s"] * 90).round(0)
        print("  Computed minutes_played from 90s column.")
    else:
        df["minutes_played"] = pd.to_numeric(df.get("minutes_played", pd.Series()), errors="coerce")

    df["goals"] = pd.to_numeric(df.get("goals", 0), errors="coerce").fillna(0)
    df["assists"] = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0)

    # Filter minimum minutes
    df = df[df["minutes_played"] >= MIN_MINUTES].copy()
    print(f"  Players with >= {MIN_MINUTES} minutes: {len(df)}")

    # Per-90 stats — avoid division by zero
    safe_90s = df["minutes_played"] / 90
    df["goals_per_90"] = df["goals"] / safe_90s
    df["assists_per_90"] = df["assists"] / safe_90s

    # Round
    df["goals_per_90"] = df["goals_per_90"].round(4)
    df["assists_per_90"] = df["assists_per_90"].round(4)

    # Keep only columns we need
    keep = ["player_name", "club", "minutes_played", "goals", "assists",
            "goals_per_90", "assists_per_90"]
    available = [c for c in keep if c in df.columns]
    df = df[available].reset_index(drop=True)

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=" * 60)
    print("FBref Scraper — 2024-25 Premier League Standard Stats")
    print("=" * 60)

    raw_df, _ = fetch_fbref_table(FBREF_URL)

    if raw_df.empty:
        print("ERROR: Could not retrieve the FBref stats table.")
        return

    print(f"\nRaw table shape: {raw_df.shape}")
    print(f"Raw columns: {list(raw_df.columns[:20])}")

    df = clean_fbref(raw_df)

    if df.empty:
        print("ERROR: No data after cleaning.")
        return

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Done. Saved {len(df)} players to {OUTPUT_PATH}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
