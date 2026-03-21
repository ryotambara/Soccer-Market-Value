"""
scraper/parse_football_data_2024.py

Parses data/raw/Football_Player_Data-Analysis2024:20205.csv
(multi-league WhoScored export), filters to PL 2024-25 clubs,
maps WhoScored position codes to standard position groups,
and writes data/raw/football_data_2024_processed.csv.

Run: python scraper/parse_football_data_2024.py
"""

import csv
import os
import pandas as pd

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_PL_2425 = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2024-25")
SOURCE_PATH  = os.path.join(_RAW_PL_2425, "player_data.csv")
OUTPUT_PATH  = os.path.join(_RAW_PL_2425, "football_data_processed.csv")

# 2024-25 PL clubs as they appear in the source file
PL_TEAMS_2024 = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man Utd", "Newcastle",
    "Nottm Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
}


def map_ws_position(pos1: str, pos2: str = "") -> str:
    """
    Map WhoScored position codes (Football_Player_Data format) to
    standard position groups used throughout the pipeline.

    Codes seen in PL 2024-25 data:
      D(C), D(CR), D(CL), D(CLR), D(LR) — centre-backs / fullbacks
      LB, RB                              — fullbacks
      AM(CLR), AM(CL), AM(CR), AM(C)     — attacking mids
      AM(LR), AM(L), AM(R)               — wingers
      M(CLR), M(CL), M(CR), M(C), M(LR) — central mids / wingers
      DMC                                 — defensive mids
      FW                                  — strikers
      GK                                  — goalkeepers
    """
    p = pos1.strip().upper() if pos1.strip() else pos2.strip().upper()
    if not p:
        return "centre_back"

    # ── Goalkeeper ────────────────────────────────────────────────
    if p == "GK" or p.startswith("GK"):
        return "goalkeeper"

    # ── Striker ───────────────────────────────────────────────────
    if p in ("FW", "ST", "CF", "SS"):
        return "striker"
    # FW(L/R) = wide forward → winger
    if p.startswith("FW") and p != "FW":
        return "winger"

    # ── Shorthand fullback / CB codes ─────────────────────────────
    if p in ("LB", "RB"):
        return "fullback"
    if p in ("LWB", "RWB", "WBL", "WBR"):
        return "fullback"
    if p in ("CB", "DC"):
        return "centre_back"

    # ── Shorthand midfield codes ──────────────────────────────────
    if p in ("DMC", "DM", "CDM"):
        return "cdm"
    if p in ("CM", "MC"):
        return "central_mid"
    if p in ("AM", "CAM"):
        return "attacking_mid"
    if p in ("LW", "RW", "LM", "RM"):
        return "winger"

    # ── Attacking Midfield: AM(…) ─────────────────────────────────
    if "AM" in p:
        inner = p.replace("AM", "").strip("()")
        # Center component → attacking mid; pure wide → winger
        if not inner or "C" in inner:
            return "attacking_mid"
        return "winger"

    # ── Wing-backs ────────────────────────────────────────────────
    if p.startswith("WB"):
        return "fullback"

    # ── Midfield: M(…) ───────────────────────────────────────────
    if "M(" in p or p.startswith("M("):
        inner = p.split("(", 1)[1].rstrip(")") if "(" in p else ""
        if "D" in inner:
            return "cdm"
        if "C" in inner:
            return "central_mid"
        # M(L), M(R), M(LR) = wide midfielders → winger
        return "winger"

    # ── Defence: D(…) ────────────────────────────────────────────
    if p.startswith("D("):
        inner = p[2:].rstrip(")")
        # Any C in the code → primarily a centre-back
        # D(C)=pure CB, D(CL/CR/LC/RC/CLR)=CB who plays wide → still CB
        if "C" in inner:
            return "centre_back"
        # D(L), D(R), D(LR) → fullback
        return "fullback"

    return "centre_back"  # default


def clean_float(val: str) -> float:
    s = str(val).strip()
    if s in ("", "-", "nan", "N/A"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Source file not found:\n  {SOURCE_PATH}")
        raise SystemExit(1)

    print(f"Reading: {SOURCE_PATH}")

    rows_in = 0
    rows_pl = 0
    records = []

    with open(SOURCE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in += 1
            club = row.get("Player Team", "").strip()
            if club not in PL_TEAMS_2024:
                continue
            rows_pl += 1

            player_name = row.get("Player Name", "").strip()
            age = clean_float(row.get("Player Age", 0))
            mins = clean_float(row.get("Mins Played", 0))
            goals = clean_float(row.get("goal", 0))
            assists = clean_float(row.get("assistTotal", 0))

            # Per-90 stats
            mins_90 = mins / 90.0 if mins > 0 else 1.0
            goals_per_90 = goals / mins_90
            assists_per_90 = assists / mins_90

            pos1 = row.get("Position 1", "").strip()
            pos2 = row.get("Position 2", "").strip()
            position_group = map_ws_position(pos1, pos2)

            xg = clean_float(row.get("xG", 0))

            records.append({
                "player_name":             player_name,
                "club":                    club,
                "age":                     age,
                "position_ws":             pos1,          # raw WhoScored code
                "position_group_ws":       position_group,
                "minutes_played":          mins,
                "goals":                   goals,
                "assists":                 assists,
                "goals_per_90":            round(goals_per_90, 4),
                "assists_per_90":          round(assists_per_90, 4),
                "tackles_per_game":        clean_float(row.get("tacklePerGame", 0)),
                "interceptions_per_game":  clean_float(row.get("interceptionPerGame", 0)),
                "fouls_per_game":          clean_float(row.get("foulsPerGame", 0)),
                "clearances_per_game":     clean_float(row.get("clearancePerGame", 0)),
                "dribbled_past_per_game":  clean_float(row.get("wasDribbledPerGame", 0)),
                "shots_per_game":          clean_float(row.get("shotsPerGame", 0)),
                "key_passes_per_game":     clean_float(row.get("keyPassPerGame", 0)),
                "dribbles_per_game":       clean_float(row.get("dribbleWonPerGame", 0)),
                "fouled_per_game":         clean_float(row.get("foulGivenPerGame", 0)),
                "avg_passes_per_game":     clean_float(row.get("totalPassesPerGame", 0)),
                "pass_success_pct":        clean_float(row.get("passSuccess", 0)),
                "crosses_per_game":        clean_float(row.get("accurateCrossesPerGame", 0)),
                "long_balls_per_game":     clean_float(row.get("accurateLongPassPerGame", 0)),
                "through_balls_per_game":  clean_float(row.get("accurateThroughBallPerGame", 0)),
                "xg":                      xg,
                "xg_per_shot":             clean_float(row.get("xGPerShot", 0)),
                # xg/90 (for xg_p90 interaction terms)
                "xg_p90":                  clean_float(row.get("xGPerNinety", 0)),
                "rating":                  clean_float(row.get("rating", 0)),
            })

    print(f"  Total rows in file: {rows_in}")
    print(f"  PL rows: {rows_pl}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["player_name", "club"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nTotal players written: {len(df)}")
    print(f"Saved → {OUTPUT_PATH}")

    print("\nPosition group distribution:")
    print(df["position_group_ws"].value_counts().to_string())

    print("\nSample (Salah, Haaland, van Dijk):")
    sample = df[df["player_name"].str.contains("Salah|Haaland|Dijk", na=False)]
    verify = ["player_name", "club", "age", "position_ws", "position_group_ws",
              "minutes_played", "goals", "assists", "xg", "rating"]
    print(sample[verify].to_string(index=False))


if __name__ == "__main__":
    main()
