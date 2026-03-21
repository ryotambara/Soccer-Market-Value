"""
scraper/fbref_parse.py

Parses a browser copy-paste from FBref Player Standard Stats (2024-25 PL)
and writes data/raw/fbref_processed.csv.

Usage:
    python scraper/fbref_parse.py data/raw/fbref_paste.txt
"""

import sys
import os
import pandas as pd

MIN_MINUTES = 500

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_PL_2526 = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2025-26")
OUTPUT_PATH  = os.path.join(_RAW_PL_2526, "fbref_processed.csv")


def parse(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # ── Step 1: keep only tab-containing lines ──────────────────────────────
    tab_lines = [l.rstrip("\n") for l in raw_lines if "\t" in l]

    # ── Step 2: keep lines whose first token is a number or "Rk" ───────────
    clean_lines = []
    for line in tab_lines:
        first = line.split("\t")[0].strip()
        if first == "Rk" or first.isdigit():
            clean_lines.append(line)

    if not clean_lines:
        print("ERROR: No valid rows found. Make sure the file contains tab-separated FBref data.")
        sys.exit(1)

    # ── Step 3: first "Rk" line is the header ───────────────────────────────
    header_line = next((l for l in clean_lines if l.split("\t")[0].strip() == "Rk"), None)
    if header_line is None:
        print("ERROR: Could not find header row starting with 'Rk'.")
        sys.exit(1)

    columns = [c.strip() for c in header_line.split("\t")]
    data_lines = [l for l in clean_lines if l.split("\t")[0].strip() != "Rk"]

    # ── Step 4: parse into DataFrame ────────────────────────────────────────
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < len(columns):
            parts += [""] * (len(columns) - len(parts))
        else:
            parts = parts[: len(columns)]
        rows.append(dict(zip(columns, parts)))

    df = pd.DataFrame(rows)

    # ── Cleaning ─────────────────────────────────────────────────────────────

    # Drop repeated header rows and empty player rows
    df = df[df["Player"] != "Player"]
    df = df[df["Player"].notna() & (df["Player"].str.strip() != "")]

    # Remove commas from Min and cast
    df["Min"] = df["Min"].str.replace(",", "", regex=False).str.strip()
    df = df[df["Min"].str.match(r"^\d+$", na=False)]
    df["Min"] = df["Min"].astype(int)

    # Drop below minimum minutes
    df = df[df["Min"] >= MIN_MINUTES]

    # Cast Gls and Ast
    df["Gls"] = pd.to_numeric(df["Gls"], errors="coerce").fillna(0)
    df["Ast"] = pd.to_numeric(df["Ast"], errors="coerce").fillna(0)

    # Aggregate players who appear more than once (mid-season transfers)
    # Keep last Squad (most recent club), sum Min/Gls/Ast
    df = (
        df.groupby("Player", as_index=False)
        .agg(
            Squad=("Squad", "last"),
            Min=("Min", "sum"),
            Gls=("Gls", "sum"),
            Ast=("Ast", "sum"),
        )
    )

    # Re-apply minutes filter after aggregation
    df = df[df["Min"] >= MIN_MINUTES].copy()

    # Per-90 rates
    ninety = df["Min"] / 90
    df["goals_per_90"] = (df["Gls"] / ninety).round(4)
    df["assists_per_90"] = (df["Ast"] / ninety).round(4)

    # Rename
    df = df.rename(columns={
        "Player": "player_name",
        "Squad": "club",
        "Min": "minutes_played",
        "Gls": "goals",
        "Ast": "assists",
    })

    df["minutes_played"] = df["minutes_played"].astype(int)
    df["goals"] = df["goals"].astype(int)
    df["assists"] = df["assists"].astype(int)

    return df[["player_name", "club", "minutes_played", "goals", "assists",
               "goals_per_90", "assists_per_90"]]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_RAW_DIR, "fbref_paste.txt")

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Parsing: {path}")
    df = parse(path)

    os.makedirs(_RAW_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nTotal players (≥{MIN_MINUTES} min): {len(df)}")
    print(f"Saved → {OUTPUT_PATH}")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
