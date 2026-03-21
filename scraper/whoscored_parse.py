"""
scraper/whoscored_parse.py

Parses data/raw/2025:2026 player data.csv (WhoScored export)
and writes data/raw/whoscored_processed.csv.

Run: python scraper/whoscored_parse.py
"""

import os
import re
import pandas as pd

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_PL_2526 = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2025-26")
SOURCE_PATH  = os.path.join(_BASE_DIR, "data", "raw", "2025:2026 player data.csv")
OUTPUT_PATH  = os.path.join(_RAW_PL_2526, "whoscored.csv")

PL_CLUBS = [
    "Man City", "Arsenal", "Liverpool", "Chelsea",
    "Man Utd", "Newcastle", "Aston Villa", "Tottenham",
    "Brighton", "Brentford", "Fulham", "Wolves",
    "Everton", "Crystal Palace", "Nottm Forest",
    "Bournemouth", "West Ham", "Leicester", "Ipswich",
    "Southampton", "Sunderland", "Leeds", "Burnley",
    "Sheffield Utd", "Luton",
]
# Sort longest first so "Nottm Forest" matches before "Forest" etc.
PL_CLUBS_SORTED = sorted(PL_CLUBS, key=len, reverse=True)


def parse_apps(apps_str: str) -> float:
    """'28(1)' → 29.0,  '27' → 27.0,  '-' → 0.0"""
    s = str(apps_str).strip()
    if s in ("-", "", "nan"):
        return 0.0
    m = re.match(r"(\d+)\((\d+)\)", s)
    if m:
        return float(int(m.group(1)) + int(m.group(2)))
    try:
        return float(s)
    except ValueError:
        return 0.0


def clean_val(v) -> float:
    """Replace '-' and blank with 0.0, cast to float."""
    s = str(v).strip()
    if s in ("-", "", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_player_field(raw: str):
    """
    Parse the messy Player column into (player_name, club, age, position).

    Example inputs:
      '1\nErling HaalandMan City, 25, FW'
      'Lamare BogardeAston Villa, 22, D(R),DMC'
    """
    # Remove leading rank number + newline
    text = re.sub(r"^\d+\n", "", str(raw)).strip()

    # Extract age: standalone number between two commas, value 15-50
    age = None
    age_match = re.search(r",\s*(\d{1,2}),", text)
    if age_match:
        candidate = int(age_match.group(1))
        if 15 <= candidate <= 50:
            age = candidate

    # Extract position: everything after the last comma
    position = ""
    last_comma = text.rfind(",")
    if last_comma != -1:
        position = text[last_comma + 1:].strip()

    # Extract club: find which PL club name appears in the string
    club = ""
    for c in PL_CLUBS_SORTED:
        if c in text:
            club = c
            break

    # Extract player name: remove club, age pattern, position, rank
    name = text
    if club:
        name = name.replace(club, "", 1)
    # Remove ", age, position" suffix
    name = re.sub(r",\s*\d{1,2}\s*,.*$", "", name).strip()
    # Remove any stray leading digits/newlines
    name = re.sub(r"^\d+\n?", "", name).strip()

    return name, club, age, position


def main():
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source file not found:\n  {SOURCE_PATH}")
        raise SystemExit(1)

    print(f"Reading: {SOURCE_PATH}")

    # Read with pandas — Player column is quoted and may contain newlines
    raw = pd.read_csv(SOURCE_PATH, header=0, dtype=str)
    print(f"  Raw rows: {len(raw)}  Columns: {len(raw.columns)}")

    records = []

    for _, row in raw.iterrows():
        vals = row.tolist()

        # Ensure we have enough columns (pad if short)
        while len(vals) < 39:
            vals.append("-")

        # ── Player field ────────────────────────────────────────────────
        player_name, club, age, position = parse_player_field(vals[0])

        if not player_name:
            continue

        # ── Apps ────────────────────────────────────────────────────────
        apps = parse_apps(vals[1])

        # ── Stats by index ───────────────────────────────────────────────
        mins              = clean_val(vals[2])
        goals             = clean_val(vals[3])
        assists           = clean_val(vals[4])
        yellow_cards      = clean_val(vals[5])
        shots_per_game    = clean_val(vals[7])
        pass_success_pct  = clean_val(vals[8])
        aerials_won       = clean_val(vals[9])
        rating            = clean_val(vals[11])
        tackles_per_game       = clean_val(vals[12])
        interceptions_per_game = clean_val(vals[13])
        fouls_per_game         = clean_val(vals[14])
        clearances_per_game    = clean_val(vals[16])
        dribbled_past_per_game = clean_val(vals[17])
        blocks_per_game        = clean_val(vals[18])
        key_passes_per_game    = clean_val(vals[21])
        dribbles_per_game      = clean_val(vals[22])
        fouled_per_game        = clean_val(vals[23])
        avg_passes_per_game    = clean_val(vals[28])
        crosses_per_game       = clean_val(vals[30])
        long_balls_per_game    = clean_val(vals[31])
        through_balls_per_game = clean_val(vals[32])
        xg                     = clean_val(vals[33])
        xg_diff                = clean_val(vals[35])
        xg_per_shot            = clean_val(vals[38])

        records.append({
            "player_name":             player_name,
            "club":                    club,
            "age":                     age,
            "position":                position,
            "apps":                    apps,
            "mins":                    mins,
            "goals":                   goals,
            "assists":                 assists,
            "yellow_cards":            yellow_cards,
            "shots_per_game":          shots_per_game,
            "pass_success_pct":        pass_success_pct,
            "aerials_won":             aerials_won,
            "rating":                  rating,
            "tackles_per_game":        tackles_per_game,
            "interceptions_per_game":  interceptions_per_game,
            "fouls_per_game":          fouls_per_game,
            "clearances_per_game":     clearances_per_game,
            "dribbled_past_per_game":  dribbled_past_per_game,
            "blocks_per_game":         blocks_per_game,
            "key_passes_per_game":     key_passes_per_game,
            "dribbles_per_game":       dribbles_per_game,
            "fouled_per_game":         fouled_per_game,
            "avg_passes_per_game":     avg_passes_per_game,
            "crosses_per_game":        crosses_per_game,
            "long_balls_per_game":     long_balls_per_game,
            "through_balls_per_game":  through_balls_per_game,
            "xg":                      xg,
            "xg_diff":                 xg_diff,
            "xg_per_shot":             xg_per_shot,
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["player_name", "club"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nTotal players parsed: {len(df)}")
    print(f"Saved → {OUTPUT_PATH}")

    verify_cols = [
        "player_name", "club", "age", "goals",
        "tackles_per_game", "key_passes_per_game", "xg", "rating",
    ]
    print("\nFirst 5 rows:")
    print(df[verify_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
