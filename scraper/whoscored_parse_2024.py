"""
scraper/whoscored_parse_2024.py

Parses data/raw/2024:2025 player data.csv (WhoScored export)
and writes data/raw/whoscored_2024_processed.csv.

The 2024:2025 file has 41 columns (vs 39 in 2025-26) because it has
an extra empty second column from the double-comma in the header
(Player,,Apps,...) and a trailing second Rating column. All stat
indices are therefore +1 relative to the 2025-26 parser.

Run: python scraper/whoscored_parse_2024.py
"""

import os
import re
import pandas as pd

SOURCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "2024:2025 player data.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "whoscored_2024_processed.csv"
)

# 2024-25 PL clubs as they appear in the WhoScored player field
PL_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man Utd", "Newcastle",
    "Nottm Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
]
PL_CLUBS_SORTED = sorted(PL_CLUBS, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Column index map for 2024:2025 WhoScored format (41 columns total)
# Header: Player,,Apps,Mins,Goals,Assists,Yel,Red,SpG,PS%,AerialsWon,MotM,
#         Rating,Tackles,Inter,Fouls,Offsides,Clear,Drb,Blocks,OwnG,
#         SpG,KeyP,Drb,Fouled,Off,Disp,UnsTch,KeyP,AvgP,PS%,
#         Crosses,LongB,ThrB,xG,Goals,xGDiff,xG/90,Shots,xG/Shots,Rating
# ---------------------------------------------------------------------------
_IDX = {
    "apps":                    2,
    "mins":                    3,
    "goals":                   4,
    "assists":                 5,
    "yellow_cards":            6,
    # 7 = Red
    "shots_per_game":          8,
    "pass_success_pct":        9,
    "aerials_won":             10,
    # 11 = MotM
    "rating":                  12,
    "tackles_per_game":        13,
    "interceptions_per_game":  14,
    "fouls_per_game":          15,
    # 16 = Offsides
    "clearances_per_game":     17,
    "dribbled_past_per_game":  18,
    "blocks_per_game":         19,
    # 20 = OwnG
    # 21 = SpG (second copy — detailed section)
    "key_passes_per_game":     22,
    "dribbles_per_game":       23,
    "fouled_per_game":         24,
    # 25 = Off, 26 = Disp, 27 = UnsTch, 28 = KeyP (2nd)
    "avg_passes_per_game":     29,
    "pass_success_pct_v2":     30,   # same stat, second block
    "crosses_per_game":        31,
    "long_balls_per_game":     32,
    "through_balls_per_game":  33,
    "xg":                      34,
    # 35 = Goals (second copy)
    "xg_diff":                 36,
    # 37 = xG/90
    # 38 = Shots (total)
    "xg_per_shot":             39,
    # 40 = Rating (second copy)
}


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
    s = str(v).strip()
    if s in ("-", "", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def map_ws_position(pos: str) -> str:
    """Map WhoScored position codes to standard position groups."""
    p = pos.strip().upper() if pos.strip() else ""
    if not p:
        return "centre_back"
    if p == "GK" or p.startswith("GK"):
        return "goalkeeper"
    if p in ("FW", "ST", "CF", "SS"):
        return "striker"
    if p.startswith("FW") and p != "FW":
        return "winger"
    if p in ("LB", "RB", "LWB", "RWB"):
        return "fullback"
    if p in ("CB", "DC"):
        return "centre_back"
    if p in ("DMC", "DM", "CDM"):
        return "cdm"
    if p in ("CM", "MC"):
        return "central_mid"
    if p in ("AM", "CAM"):
        return "attacking_mid"
    if p in ("LW", "RW", "LM", "RM"):
        return "winger"
    if "AM" in p:
        inner = p.replace("AM", "").strip("()")
        if not inner or "C" in inner:
            return "attacking_mid"
        return "winger"
    if p.startswith("WB"):
        return "fullback"
    if "M(" in p:
        inner = p.split("(", 1)[1].rstrip(")") if "(" in p else ""
        if "D" in inner:
            return "cdm"
        if "C" in inner:
            return "central_mid"
        return "winger"
    if p.startswith("D("):
        inner = p[2:].rstrip(")")
        if "C" in inner:
            return "centre_back"
        return "fullback"
    return "centre_back"


def parse_player_field(raw: str):
    """
    Parse the messy Player column into (player_name, club, age, position).
    Example: '1\\nMohamed SalahLiverpool, 33, AM(CLR),FW'
    """
    text = re.sub(r"^\d+\n", "", str(raw)).strip()

    age = None
    age_match = re.search(r",\s*(\d{1,2}),", text)
    if age_match:
        candidate = int(age_match.group(1))
        if 15 <= candidate <= 50:
            age = candidate

    position = ""
    last_comma = text.rfind(",")
    if last_comma != -1:
        position = text[last_comma + 1:].strip()

    club = ""
    for c in PL_CLUBS_SORTED:
        if c in text:
            club = c
            break

    name = text
    if club:
        name = name.replace(club, "", 1)
    name = re.sub(r",\s*\d{1,2}\s*,.*$", "", name).strip()
    name = re.sub(r"^\d+\n?", "", name).strip()

    return name, club, age, position


def main():
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source file not found:\n  {SOURCE_PATH}")
        raise SystemExit(1)

    print(f"Reading: {SOURCE_PATH}")

    raw = pd.read_csv(SOURCE_PATH, header=0, dtype=str)
    print(f"  Raw rows: {len(raw)}  Columns: {len(raw.columns)}")

    records = []

    for _, row in raw.iterrows():
        vals = row.tolist()

        # Pad to at least 41 columns
        while len(vals) < 41:
            vals.append("-")

        player_name, club, age, position = parse_player_field(vals[0])

        if not player_name or not club:
            continue

        apps = parse_apps(vals[_IDX["apps"]])
        mins = clean_val(vals[_IDX["mins"]])

        goals   = clean_val(vals[_IDX["goals"]])
        assists = clean_val(vals[_IDX["assists"]])

        # Compute per-90 from totals
        mins_90       = mins / 90.0 if mins > 0 else 1.0
        goals_per_90  = round(goals / mins_90, 4)
        assists_per_90 = round(assists / mins_90, 4)

        records.append({
            "player_name":             player_name,
            "club":                    club,
            "age":                     age,
            "position":                position,
            "position_group_ws":       map_ws_position(position),
            "apps":                    apps,
            "minutes_played":          mins,
            "goals":                   goals,
            "assists":                 assists,
            "goals_per_90":            goals_per_90,
            "assists_per_90":          assists_per_90,
            "yellow_cards":            clean_val(vals[_IDX["yellow_cards"]]),
            "shots_per_game":          clean_val(vals[_IDX["shots_per_game"]]),
            "pass_success_pct":        clean_val(vals[_IDX["pass_success_pct"]]),
            "aerials_won":             clean_val(vals[_IDX["aerials_won"]]),
            "rating":                  clean_val(vals[_IDX["rating"]]),
            "tackles_per_game":        clean_val(vals[_IDX["tackles_per_game"]]),
            "interceptions_per_game":  clean_val(vals[_IDX["interceptions_per_game"]]),
            "fouls_per_game":          clean_val(vals[_IDX["fouls_per_game"]]),
            "clearances_per_game":     clean_val(vals[_IDX["clearances_per_game"]]),
            "dribbled_past_per_game":  clean_val(vals[_IDX["dribbled_past_per_game"]]),
            "blocks_per_game":         clean_val(vals[_IDX["blocks_per_game"]]),
            "key_passes_per_game":     clean_val(vals[_IDX["key_passes_per_game"]]),
            "dribbles_per_game":       clean_val(vals[_IDX["dribbles_per_game"]]),
            "fouled_per_game":         clean_val(vals[_IDX["fouled_per_game"]]),
            "avg_passes_per_game":     clean_val(vals[_IDX["avg_passes_per_game"]]),
            "crosses_per_game":        clean_val(vals[_IDX["crosses_per_game"]]),
            "long_balls_per_game":     clean_val(vals[_IDX["long_balls_per_game"]]),
            "through_balls_per_game":  clean_val(vals[_IDX["through_balls_per_game"]]),
            "xg":                      clean_val(vals[_IDX["xg"]]),
            "xg_diff":                 clean_val(vals[_IDX["xg_diff"]]),
            "xg_per_shot":             clean_val(vals[_IDX["xg_per_shot"]]),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["player_name", "club"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nTotal players parsed: {len(df)}")
    print(f"Saved → {OUTPUT_PATH}")

    verify = ["player_name", "club", "age", "minutes_played", "goals",
              "assists", "blocks_per_game", "aerials_won", "xg", "rating"]
    print("\nFirst 5 rows:")
    print(df[verify].head(5).to_string(index=False))

    print("\nVerification — Salah (expect: 29G, 18A, 3380 mins):")
    salah = df[df["player_name"].str.contains("Salah", na=False)]
    if not salah.empty:
        print(salah[verify].to_string(index=False))
    else:
        print("  Salah not found — check PL club list.")

    print("\nVerification — Haaland (expect: 22G, 2742 mins):")
    haaland = df[df["player_name"].str.contains("Haaland", na=False)]
    if not haaland.empty:
        print(haaland[verify].to_string(index=False))


if __name__ == "__main__":
    main()
