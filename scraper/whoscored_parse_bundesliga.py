"""
scraper/whoscored_parse_bundesliga.py

Parses data/raw/bundesliga_2025-26_players.csv (WhoScored export)
and writes data/raw/whoscored_bundesliga_processed.csv.

The Bundesliga 2025-26 file uses the standard 39-column WhoScored
format.  Club names in the player field are short WhoScored names
(e.g. "Bayern", "Dortmund") which are mapped to full Transfermarkt
club names after parsing.

Run: python scraper/whoscored_parse_bundesliga.py
"""

import os
import re
import pandas as pd

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_BL_2526 = os.path.join(_BASE_DIR, "data", "raw", "Bundesliga", "2025-26")
SOURCE_PATH  = os.path.join(_RAW_BL_2526, "whoscored.csv")
OUTPUT_PATH  = os.path.join(_RAW_BL_2526, "whoscored_processed.csv"
)

# ---------------------------------------------------------------------------
# All known WhoScored name variants for each Bundesliga club.
# Sorted longest-first at runtime so e.g. "Borussia Dortmund" matches
# before "Dortmund" and the full string is removed from player names.
# ---------------------------------------------------------------------------
BL_CLUBS = [
    # Full / alternate names that WhoScored may embed in the player field
    "Borussia Dortmund",
    "Borussia Mönchengladbach",
    "Borussia Monchengladbach",  # ASCII fallback
    "Borussia M.",               # WhoScored abbreviated form
    "Bayer Leverkusen",
    "Eintracht Frankfurt",
    "RB Leipzig",
    "Union Berlin",
    "St. Pauli",                 # with period — common WhoScored form
    # Short names
    "Bayern",
    "Dortmund",
    "Leverkusen",
    "Leipzig",
    "RBL",                       # WhoScored abbreviation for RB Leipzig
    "Stuttgart",
    "Frankfurt",
    "Hoffenheim",
    "Wolfsburg",
    "Freiburg",
    "Werder",                    # WhoScored short form for Werder Bremen
    "Bremen",
    "Hamburg",
    "Augsburg",
    "Gladbach",
    "Mainz",
    "Cologne",
    "Köln",                      # German spelling
    "Koln",                      # ASCII fallback
    "St Pauli",                  # without period
    "Heidenheim",
]
# Sort longest first so longer names take priority over substrings
BL_CLUBS_SORTED = sorted(BL_CLUBS, key=len, reverse=True)

# ---------------------------------------------------------------------------
# Map every variant → canonical full Transfermarkt club name
# ---------------------------------------------------------------------------
CLUB_NAME_MAP = {
    "Bayern":                   "Bayern Munich",
    "Borussia Dortmund":        "Borussia Dortmund",
    "Dortmund":                 "Borussia Dortmund",
    "Bayer Leverkusen":         "Bayer 04 Leverkusen",
    "Leverkusen":               "Bayer 04 Leverkusen",
    "RB Leipzig":               "RB Leipzig",
    "Leipzig":                  "RB Leipzig",
    "Stuttgart":                "VfB Stuttgart",
    "Eintracht Frankfurt":      "Eintracht Frankfurt",
    "Frankfurt":                "Eintracht Frankfurt",
    "Hoffenheim":               "TSG 1899 Hoffenheim",
    "Wolfsburg":                "VfL Wolfsburg",
    "Freiburg":                 "SC Freiburg",
    "Bremen":                   "SV Werder Bremen",
    "Hamburg":                  "Hamburger SV",
    "Augsburg":                 "FC Augsburg",
    "Borussia Mönchengladbach": "Borussia Mönchengladbach",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Borussia M.":              "Borussia Mönchengladbach",
    "Gladbach":                 "Borussia Mönchengladbach",
    "RBL":                      "RB Leipzig",
    "Werder":                   "SV Werder Bremen",
    "Mainz":                    "1.FSV Mainz 05",
    "Cologne":                  "1.FC Köln",
    "Köln":                     "1.FC Köln",
    "Koln":                     "1.FC Köln",
    "Union Berlin":             "1.FC Union Berlin",
    "St. Pauli":                "FC St. Pauli",
    "St Pauli":                 "FC St. Pauli",
    "Heidenheim":               "1.FC Heidenheim 1846",
}

# ---------------------------------------------------------------------------
# Column indices for 39-column WhoScored 2025-26 format
# Header: Player,Apps,Mins,Goals,Assists,Yel,Red,SpG,PS%,AerialsWon,MotM,
#         Rating,Tackles,Inter,Fouls,Offsides,Clear,Drb,Blocks,OwnG,SpG,
#         KeyP,Drb,Fouled,Off,Disp,UnsTch,KeyP,AvgP,PS%,Crosses,LongB,ThrB,
#         xG,Goals,xGDiff,xG/90,Shots,xG/Shots
# ---------------------------------------------------------------------------


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


# Fragments that may remain at the end of a player name after the primary
# club-removal step.  Sorted longest-first so e.g. "Borussia Dortmund" is
# tried before "Borussia M." or "Bremen".
TRAILING_FRAGMENTS = sorted([
    "Borussia Dortmund",
    "Union Berlin",
    "Borussia M.",
    "St. Pauli",
    "Leverkusen",
    "Hoffenheim",
    "Heidenheim",
    "Wolfsburg",
    "Frankfurt",
    "Stuttgart",
    "Freiburg",
    "Gladbach",
    "Augsburg",
    "Mainz",
    "Bremen",
    "Werder",
    "Hamburg",
    "Bayern",
    "Koln",
    "RBL",
], key=len, reverse=True)


def parse_player_field(raw: str):
    """
    Parse the messy Player column into (player_name, club_short, age, position).
    Example: '1\\nHarry KaneBayern, 32, AM(C),FW'
    Returns club_short — caller maps to full name.
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

    club_short = ""
    for c in BL_CLUBS_SORTED:
        if c in text:
            club_short = c
            break

    name = text
    if club_short:
        name = name.replace(club_short, "", 1)
    name = re.sub(r",\s*\d{1,2}\s*,.*$", "", name).strip()
    name = re.sub(r"^\d+\n?", "", name).strip()

    # Strip any trailing club fragment left after primary removal
    for frag in TRAILING_FRAGMENTS:
        if name.endswith(frag):
            name = name[: -len(frag)].strip()
            break

    return name, club_short, age, position


def main():
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source file not found:\n  {SOURCE_PATH}")
        raise SystemExit(1)

    print(f"Reading: {SOURCE_PATH}")

    raw = pd.read_csv(SOURCE_PATH, header=0, dtype=str)
    print(f"  Raw rows: {len(raw)}  Columns: {len(raw.columns)}")

    # ── DIAGNOSTIC: scan every player field and find the raw club string ──
    # Extract the text between player-name start and the ", age," separator.
    # For rows where NO known club matches, print the raw suffix so you can
    # see exactly what WhoScored uses and add it to BL_CLUBS / CLUB_NAME_MAP.
    print("\n--- DIAGNOSTIC: raw club strings found in CSV ---")
    raw_club_hits = {}    # club_short → count
    unmatched_prefixes = []

    for _, row in raw.iterrows():
        vals = row.tolist()
        text = re.sub(r"^\d+\n", "", str(vals[0])).strip()
        age_m = re.search(r",\s*\d{1,2}\s*,", text)
        if not age_m:
            continue
        prefix = text[:age_m.start()]   # "PlayerNameClubName"

        found = ""
        for c in BL_CLUBS_SORTED:
            if c in prefix:
                found = c
                break

        if found:
            raw_club_hits[found] = raw_club_hits.get(found, 0) + 1
        else:
            # Show the last 40 chars of the prefix — that's where the club sits
            unmatched_prefixes.append(prefix[-40:].strip())

    print("  Matched club strings and counts:")
    for club, count in sorted(raw_club_hits.items(), key=lambda x: x[0]):
        full = CLUB_NAME_MAP.get(club, club)
        print(f"    '{club}' ({count} players) → '{full}'")

    if unmatched_prefixes:
        unique_unmatched = sorted(set(unmatched_prefixes))
        print(f"\n  WARNING: {len(unmatched_prefixes)} rows with NO club match.")
        print("  Raw tail of player field (last 40 chars before age) — "
              "add the club name to BL_CLUBS/CLUB_NAME_MAP:")
        for p in unique_unmatched[:30]:
            print(f"    '{p}'")
    else:
        print("\n  All rows matched a known club.")
    print("--- END DIAGNOSTIC ---\n")

    # ── Main parsing loop ────────────────────────────────────────────────────
    records = []

    for _, row in raw.iterrows():
        vals = row.tolist()

        # Pad to at least 39 columns
        while len(vals) < 39:
            vals.append("-")

        player_name, club_short, age, position = parse_player_field(vals[0])

        if not player_name or not club_short:
            continue

        # Map short/variant club name to full TM name
        club = CLUB_NAME_MAP.get(club_short, club_short)

        apps = parse_apps(vals[1])
        mins = clean_val(vals[2])

        goals   = clean_val(vals[3])
        assists = clean_val(vals[4])

        # Compute per-90 from totals
        mins_90        = mins / 90.0 if mins > 0 else 1.0
        goals_per_90   = round(goals / mins_90, 4)
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
            "yellow_cards":            clean_val(vals[5]),
            "shots_per_game":          clean_val(vals[7]),
            "pass_success_pct":        clean_val(vals[8]),
            "aerials_won":             clean_val(vals[9]),
            "rating":                  clean_val(vals[11]),
            "tackles_per_game":        clean_val(vals[12]),
            "interceptions_per_game":  clean_val(vals[13]),
            "fouls_per_game":          clean_val(vals[14]),
            "clearances_per_game":     clean_val(vals[16]),
            "dribbled_past_per_game":  clean_val(vals[17]),
            "blocks_per_game":         clean_val(vals[18]),
            "key_passes_per_game":     clean_val(vals[21]),
            "dribbles_per_game":       clean_val(vals[22]),
            "fouled_per_game":         clean_val(vals[23]),
            "avg_passes_per_game":     clean_val(vals[28]),
            "crosses_per_game":        clean_val(vals[30]),
            "long_balls_per_game":     clean_val(vals[31]),
            "through_balls_per_game":  clean_val(vals[32]),
            "xg":                      clean_val(vals[33]),
            "xg_diff":                 clean_val(vals[35]),
            "xg_per_shot":             clean_val(vals[38]),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["player_name", "club"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nTotal players parsed: {len(df)}")
    print(f"Saved → {OUTPUT_PATH}")

    # ── Players per club ─────────────────────────────────────────────────────
    print("\nPlayers per club (parsed output):")
    club_counts = df["club"].value_counts().sort_index()
    for club, cnt in club_counts.items():
        print(f"  {club:<40} {cnt:>3} players")
    missing_clubs = [
        full for full in CLUB_NAME_MAP.values()
        if full not in club_counts.index
    ]
    missing_clubs = sorted(set(missing_clubs))
    if missing_clubs:
        print(f"\n  WARNING: {len(missing_clubs)} expected clubs have 0 players:")
        for c in missing_clubs:
            print(f"    {c}")

    # ── Malformed name check ─────────────────────────────────────────────────
    # Flag names that contain digits or look like they still have club fragments
    club_fragments = [
        "Bayern", "Borussia", "Bayer", "Leverkusen", "Leipzig",
        "Stuttgart", "Frankfurt", "Hoffenheim", "Wolfsburg", "Freiburg",
        "Bremen", "Hamburg", "Augsburg", "Gladbach", "Mainz",
        "Köln", "Koln", "Cologne", "Union", "Pauli", "Heidenheim",
        "Dortmund", "Eintracht", "Werder", "Hamburger", "RBL",
    ]
    fragment_pat = re.compile(
        r"\d|" + "|".join(re.escape(f) for f in club_fragments)
    )
    malformed = df[df["player_name"].str.contains(fragment_pat, na=False)]
    if not malformed.empty:
        print(f"\n  WARNING: {len(malformed)} player names look malformed "
              f"(contain digits or club name fragments):")
        for _, r in malformed.iterrows():
            print(f"    '{r['player_name']}' — club={r['club']}")
    else:
        print("\n  All player names look clean (no digits or club fragments).")

    # ── Spot checks ──────────────────────────────────────────────────────────
    verify = ["player_name", "club", "age", "minutes_played", "goals",
              "assists", "blocks_per_game", "aerials_won", "xg", "rating"]
    print("\nFirst 5 rows:")
    print(df[verify].head(5).to_string(index=False))

    print("\nVerification — Kane (expect: Bayern Munich, high goals):")
    kane = df[df["player_name"].str.contains("Kane", na=False)]
    if not kane.empty:
        print(kane[verify].to_string(index=False))
    else:
        print("  Kane not found — check Bundesliga club list.")


if __name__ == "__main__":
    main()
