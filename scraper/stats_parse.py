"""
scraper/stats_parse.py

Reads a browser copy-paste from FBref (any format — tab or comma separated)
and fills in stats columns in data/raw/stats_2025-26_entry.csv.

Matching is fuzzy so minor name differences (accents, abbreviations) are handled.

Usage:
    python scraper/stats_parse.py

Reads:  data/raw/stats_2025-26_paste.txt
Updates: data/raw/stats_2025-26_entry.csv
"""

import os
import re
import pandas as pd
from rapidfuzz import fuzz, process

PASTE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stats_2025-26_paste.txt")
ENTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stats_2025-26_entry.csv")

FUZZY_THRESHOLD = 85


# ── Parse the paste file ────────────────────────────────────────────────────

def detect_sep(line: str) -> str:
    return "\t" if line.count("\t") >= line.count(",") else ","


def parse_paste(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.readlines()

    # Remove BOM
    if raw and raw[0].startswith("\ufeff"):
        raw[0] = raw[0][1:]

    # Keep only lines with a tab OR comma (skip title / instruction lines)
    lines = [l.rstrip("\n") for l in raw if "\t" in l or l.count(",") >= 4]

    if not lines:
        print("ERROR: No structured data found in paste file.")
        print("Make sure you pasted tab-separated or comma-separated data.")
        raise SystemExit(1)

    sep = detect_sep(lines[0])

    # Find the header row — contains "Player" and "Min" (or "Minutes")
    header_idx = None
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in line.split(sep)]
        if any(p in ("Player", "Rk") for p in parts) and any(
            p in ("Min", "Minutes", "Mins") for p in parts
        ):
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: Could not find header row with 'Player' and 'Min' columns.")
        print("First 3 lines of your file:")
        for l in lines[:3]:
            print(" ", l[:120])
        raise SystemExit(1)

    columns = [p.strip().strip('"') for p in lines[header_idx].split(sep)]
    data_lines = lines[header_idx + 1 :]

    # Store raw parts as lists (not dicts) to avoid duplicate column name collisions.
    # FBref reuses "Gls" and "Ast" for both totals and per-90 columns — dict(zip(...))
    # would let the per-90 value overwrite the total. Positional indexing is reliable.
    rows = []
    for line in data_lines:
        parts = [p.strip().strip('"') for p in line.split(sep)]

        # Skip repeated header rows
        if parts[0] in ("Rk", "Player"):
            continue
        if not parts[0]:
            continue

        rows.append(parts)

    # Build DataFrame with integer column positions
    df = pd.DataFrame(rows)

    # Remove rows where the player name position (col 1) is blank or a header value
    df = df[df[1].notna() & (df[1].str.strip() != "") & (df[1] != "Player")]

    return df


# Column positions in FBref Player Standard Stats browser paste:
# 0=Rk, 1=Player, 2=Nation, 3=Pos, 4=Squad, 5=Age, 6=Born,
# 7=MP, 8=Starts, 9=Min, 10=90s, 11=Gls, 12=Ast, ...
# (confirmed from Salah row: Min=1,819 @ 9, Gls=5 @ 11, Ast=6 @ 12)
_IDX_PLAYER = 1
_IDX_SQUAD  = 4
_IDX_MIN    = 9
_IDX_GLS    = 11
_IDX_AST    = 12


def extract_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pull out player_name, club, minutes, goals, assists using positional indices."""

    def to_int(s: "pd.Series") -> "pd.Series":
        return (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": "0", "-": "0"})
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    out = pd.DataFrame()
    out["player_name"]   = raw_df[_IDX_PLAYER].str.strip()
    out["club"]          = raw_df[_IDX_SQUAD].str.strip() if _IDX_SQUAD in raw_df.columns else ""
    out["minutes_played"] = to_int(raw_df[_IDX_MIN])
    out["goals"]          = to_int(raw_df[_IDX_GLS])
    out["assists"]        = to_int(raw_df[_IDX_AST])

    # Skip "2 squads" aggregate rows
    out = out[~out["club"].str.match(r"^\d+\s+squad", case=False, na=False)]
    out = out[out["player_name"].str.strip() != ""]

    # If a player appears multiple times (mid-season transfer), aggregate
    out = (
        out.groupby("player_name", as_index=False)
        .agg(club=("club", "last"),
             minutes_played=("minutes_played", "sum"),
             goals=("goals", "sum"),
             assists=("assists", "sum"))
    )

    # Compute per-90
    ninety = out["minutes_played"] / 90
    ninety = ninety.replace(0, float("nan"))
    out["goals_per_90"]   = (out["goals"]   / ninety).round(4).fillna(0)
    out["assists_per_90"] = (out["assists"] / ninety).round(4).fillna(0)

    return out


# ── Fuzzy merge into entry file ─────────────────────────────────────────────

def merge_into_entry(stats: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
    entry = entry.copy()

    stats_names = stats["player_name"].tolist()
    matched = 0
    unmatched = []

    for idx, row in entry.iterrows():
        entry_name = str(row["player_name"]).strip()

        result = process.extractOne(
            entry_name.lower(),
            [n.lower() for n in stats_names],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )

        if result is None:
            unmatched.append(entry_name)
            continue

        _, score, match_idx = result
        stats_row = stats.iloc[match_idx]

        entry.at[idx, "minutes_played"] = stats_row["minutes_played"]
        entry.at[idx, "goals"]          = stats_row["goals"]
        entry.at[idx, "assists"]        = stats_row["assists"]
        entry.at[idx, "goals_per_90"]   = stats_row["goals_per_90"]
        entry.at[idx, "assists_per_90"] = stats_row["assists_per_90"]
        matched += 1

    print(f"\nMatched:   {matched} players")
    print(f"Unmatched: {len(unmatched)} players")
    if unmatched:
        print("Could not find stats for:")
        for name in unmatched[:20]:
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    return entry


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PASTE_PATH):
        print(f"ERROR: Paste file not found: {PASTE_PATH}")
        raise SystemExit(1)

    if not os.path.exists(ENTRY_PATH):
        print(f"ERROR: Entry file not found: {ENTRY_PATH}")
        raise SystemExit(1)

    print(f"Reading paste: {PASTE_PATH}")
    raw_df = parse_paste(PASTE_PATH)
    print(f"  Raw rows parsed: {len(raw_df)}")

    stats = extract_stats(raw_df)
    print(f"  Players extracted: {len(stats)}")

    # Verify Salah row as a sanity check
    salah = stats[stats["player_name"].str.contains("Salah", case=False, na=False)]
    if not salah.empty:
        print("\n  [VERIFY] Mohamed Salah row:")
        print(salah[["player_name", "club", "minutes_played", "goals", "assists",
                      "goals_per_90", "assists_per_90"]].to_string(index=False))
    else:
        print("  [VERIFY] Salah not found in parsed data.")

    print(f"\nLoading entry file: {ENTRY_PATH}")
    entry = pd.read_csv(ENTRY_PATH, encoding="utf-8")
    print(f"  Entry rows: {len(entry)}")

    updated = merge_into_entry(stats, entry)

    updated.to_csv(ENTRY_PATH, index=False, encoding="utf-8")
    print(f"\nSaved updated entry file → {ENTRY_PATH}")

    filled = updated["minutes_played"].notna() & (updated["minutes_played"] != "")
    print(f"Players with stats filled: {filled.sum()} / {len(updated)}")

    print("\nFirst 5 rows:")
    print(updated.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
