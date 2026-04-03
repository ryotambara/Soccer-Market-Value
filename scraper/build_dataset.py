"""
build_dataset.py v2 — Unified pipeline using player_id as primary key.

Key change: Use the Leagues/{League}/{League}.csv file as the source of truth
for which players are in the league (has player_id + market_value + club).
Join to player_profiles on player_id for nationality/position/age.
Join to WhoScored on fuzzy name match for performance stats.

Usage:
    python3 scraper/build_dataset.py --league liga_portugal --season 2024-25
    python3 scraper/build_dataset.py --all --season 2024-25
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

BASE_DIR = Path(__file__).resolve().parent.parent
PLAYER_DATA = BASE_DIR / "data/raw/PLAYER DATA"
LEAGUES_DIR = BASE_DIR / "data/raw/Leagues"
RAW_DIR = BASE_DIR / "data/raw"
PROC_DIR = BASE_DIR / "data/processed"

MIN_MINUTES = 500
FUZZY_THRESHOLD = 85

LEAGUE_CONFIG = {
    "premier_league": {
        "name": "Premier League",
        "leagues_folder": "Premier League",
        "league_csv": "Premier League.csv",
        "whoscored_stats": "Soccer Stats - Premier League Stats 2.csv",
        "whoscored_keepers": "Soccer Stats - Premier League Keeper Stats.csv",
        "historic_top": [
            "Man City", "Arsenal", "Liverpool",
            "Chelsea", "Man Utd", "Tottenham",
        ],
        "promoted_2024_25": ["Leicester", "Ipswich", "Southampton"],
        "league_table_2024_25": {
            "Liverpool": 1, "Arsenal": 2, "Chelsea": 3,
            "Man City": 4, "Newcastle": 5, "Aston Villa": 6,
            "Fulham": 7, "Brighton": 8, "Brentford": 9,
            "Forest": 10, "Tottenham": 11,
            "Man Utd": 12, "West Ham": 13,
            "Bournemouth": 14, "Crystal Palace": 15,
            "Wolves": 16, "Everton": 17,
            "Leicester": 18, "Ipswich": 19, "Southampton": 20,
        },
        "bottom6_threshold": 15,
        "top4_threshold": 4,
        "top6_threshold": 6,
        "whoscored_clubs": {
            "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
            "Bournemouth": "Bournemouth", "Brentford": "Brentford",
            "Brighton": "Brighton", "Chelsea": "Chelsea",
            "Crystal Palace": "Crystal Palace", "Everton": "Everton",
            "Fulham": "Fulham", "Ipswich": "Ipswich",
            "Leicester": "Leicester", "Liverpool": "Liverpool",
            "Man City": "Man City", "Man Utd": "Man Utd",
            "Newcastle": "Newcastle", "Nottingham Forest": "Forest",
            "Southampton": "Southampton", "Tottenham": "Tottenham",
            "West Ham": "West Ham", "Wolves": "Wolves",
        },
    },
    "bundesliga": {
        "name": "Bundesliga",
        "leagues_folder": "Bundesliga",
        "league_csv": "Bundesliga.csv",
        "whoscored_stats": "Soccer Stats - Bundesliga Stats 2.csv",
        "whoscored_keepers": "Soccer Stats - Bundesliga Keeper Stats.csv",
        "historic_top": [
            "Bayern Munich", "Bor. Dortmund", "B. Leverkusen", "RB Leipzig",
        ],
        "promoted_2024_25": ["FC St. Pauli", "Heidenheim", "Holstein Kiel"],
        "valid_clubs": [
            "Bayern Munich", "B. Leverkusen", "VfB Stuttgart", "RB Leipzig",
            "Bor. Dortmund", "E. Frankfurt", "TSG Hoffenheim", "FC Augsburg",
            "Werder Bremen", "Bor. M'gladbach", "VfL Wolfsburg", "Union Berlin",
            "SC Freiburg", "1.FSV Mainz 05", "FC St. Pauli", "Heidenheim",
            "Holstein Kiel", "1.FC Köln",
        ],
        "league_table_2024_25": {
            "B. Leverkusen": 1, "Bayern Munich": 2, "VfB Stuttgart": 3,
            "RB Leipzig": 4, "Bor. Dortmund": 5, "E. Frankfurt": 6,
            "TSG Hoffenheim": 7, "FC Augsburg": 8, "Werder Bremen": 9,
            "Bor. M'gladbach": 10, "VfL Wolfsburg": 11, "Union Berlin": 12,
            "SC Freiburg": 13, "1.FSV Mainz 05": 14, "FC St. Pauli": 15,
            "Heidenheim": 16, "Holstein Kiel": 17, "1.FC Köln": 18,
        },
        "bottom6_threshold": 13,
        "top4_threshold": 4,
        "top6_threshold": 6,
        "whoscored_clubs": {
            "Bayern": "Bayern Munich",
            "Borussia Dortmund": "Bor. Dortmund",
            "Leverkusen": "B. Leverkusen",
            "RBL": "RB Leipzig",
            "Eintracht Frankfurt": "E. Frankfurt",
            "Stuttgart": "VfB Stuttgart",
            "Hoffenheim": "TSG Hoffenheim",
            "Wolfsburg": "VfL Wolfsburg",
            "Freiburg": "SC Freiburg",
            "Werder Bremen": "Werder Bremen",
            "Augsburg": "FC Augsburg",
            "Borussia M.Gladbach": "Bor. M'gladbach",
            "Mainz": "1.FSV Mainz 05",
            "Köln": "1.FC Köln",
            "Union Berlin": "Union Berlin",
            "St. Pauli": "FC St. Pauli",
            "FC Heidenheim": "Heidenheim",
            "Bochum": "VfL Bochum",
            "Holstein Kiel": "Holstein Kiel",
        },
    },
    "la_liga": {
        "name": "La Liga",
        "leagues_folder": "LaLiga",
        "league_csv": "LaLiga.csv",
        "whoscored_stats": "Soccer Stats - La Liga Stats 2.csv",
        "whoscored_keepers": "Soccer Stats - La Liga Keeper Stats.csv",
        "historic_top": [
            "Barcelona", "Real Madrid", "Atlético Madrid",
        ],
        "promoted_2024_25": ["Leganés", "Real Valladolid", "Espanyol"],
        "valid_clubs": [
            "Barcelona", "Real Madrid", "Atlético Madrid", "Athletic",
            "Real Sociedad", "Villarreal", "Real Betis", "Valencia",
            "Sevilla FC", "Celta de Vigo", "RCD Mallorca", "Rayo Vallecano",
            "Getafe", "CA Osasuna", "Alavés", "Real Valladolid",
            "Leganés", "Girona", "Espanyol", "Las Palmas",
        ],
        "league_table_2024_25": {
            "Barcelona": 1, "Real Madrid": 2, "Atlético Madrid": 3,
            "Athletic": 4, "Villarreal": 5, "Real Sociedad": 6,
            "Real Betis": 7, "Celta de Vigo": 8, "Girona": 9,
            "Rayo Vallecano": 10, "Valencia": 11, "Getafe": 12,
            "CA Osasuna": 13, "Alavés": 14, "RCD Mallorca": 15,
            "Sevilla FC": 16, "Las Palmas": 17, "Espanyol": 18,
            "Leganés": 19, "Real Valladolid": 20,
        },
        "bottom6_threshold": 15,
        "top4_threshold": 4,
        "top6_threshold": 6,
        "whoscored_clubs": {
            "Barcelona": "Barcelona",
            "Real Madrid": "Real Madrid",
            "Atletico": "Atlético Madrid",
            "Athletic Club": "Athletic",
            "Real Sociedad": "Real Sociedad",
            "Villarreal": "Villarreal",
            "Real Betis": "Real Betis",
            "Valencia": "Valencia",
            "Sevilla": "Sevilla FC",
            "Celta Vigo": "Celta de Vigo",
            "Osasuna": "CA Osasuna",
            "Mallorca": "RCD Mallorca",
            "Rayo Vallecano": "Rayo Vallecano",
            "Getafe": "Getafe",
            "Deportivo Alaves": "Alavés",
            "Las Palmas": "Las Palmas",
            "Girona": "Girona",
            "Espanyol": "Espanyol",
            "Real Valladolid": "Real Valladolid",
            "Leganes": "Leganés",
        },
    },
    "serie_a": {
        "name": "Serie A",
        "leagues_folder": "Serie A",
        "league_csv": "Serie A.csv",
        "whoscored_stats": "Soccer Stats - Serie A Stats 2.csv",
        "whoscored_keepers": "Soccer Stats - Serie A Keeper Stats.csv",
        "historic_top": [
            "Juventus", "Inter", "Napoli", "AS Roma", "AC Milan",
        ],
        "promoted_2024_25": ["Como", "Parma", "Venezia"],
        "valid_clubs": [
            "Inter", "AC Milan", "Juventus", "Atalanta BC", "Bologna",
            "AS Roma", "Lazio", "Fiorentina", "Torino", "Napoli",
            "Genoa", "Cagliari", "Hellas Verona", "Udinese",
            "Lecce", "Como", "Parma", "Venezia", "Empoli", "Monza",
        ],
        "league_table_2024_25": {
            "Inter": 1, "AC Milan": 2, "Juventus": 3,
            "Atalanta BC": 4, "Bologna": 5, "AS Roma": 6,
            "Lazio": 7, "Fiorentina": 8, "Torino": 9,
            "Napoli": 10, "Empoli": 11, "Monza": 12,
            "Genoa": 13, "Cagliari": 14, "Hellas Verona": 15,
            "Udinese": 16, "Lecce": 17, "Como": 18,
            "Parma": 19, "Venezia": 20,
        },
        "bottom6_threshold": 15,
        "top4_threshold": 4,
        "top6_threshold": 6,
        "whoscored_clubs": {
            "Inter": "Inter",
            "Juventus": "Juventus",
            "Roma": "AS Roma",
            "Napoli": "Napoli",
            "Atalanta": "Atalanta BC",
            "AC Milan": "AC Milan",
            "Lazio": "Lazio",
            "Fiorentina": "Fiorentina",
            "Bologna": "Bologna",
            "Torino": "Torino",
            "Genoa": "Genoa",
            "Lecce": "Lecce",
            "Cagliari": "Cagliari",
            "Udinese": "Udinese",
            "Verona": "Hellas Verona",
            "Como": "Como",
            "Parma Calcio 1913": "Parma",
            "Venezia": "Venezia",
            "Monza": "Monza",
            "Empoli": "Empoli",
        },
    },
    "liga_portugal": {
        "name": "Liga Portugal",
        "leagues_folder": "Liga Portugal",
        "league_csv": "Liga Portugal.csv",
        "whoscored_stats": "Soccer Stats - Primeira Liga Stats 2.csv",
        "whoscored_keepers": "Soccer Stats - Primeira Liga Keeper Stats.csv",
        "historic_top": [
            "Benfica", "FC Porto", "Sporting CP",
        ],
        "promoted_2024_25": ["Avs FS", "Nacional", "Santa Clara"],
        "valid_clubs": [
            "Benfica", "Sporting CP", "FC Porto", "SC Braga",
            "Vit. Guimarães", "Casa Pia", "Avs FS", "Rio Ave",
            "Famalicão", "Moreirense", "Arouca", "Gil Vicente",
            "Estoril", "Estrela Amadora", "Nacional", "Santa Clara",
            "Boavista", "Farense",
        ],
        "league_table_2024_25": {
            "Benfica": 1, "Sporting CP": 2, "FC Porto": 3,
            "SC Braga": 4, "Vit. Guimarães": 5, "Avs FS": 6,
            "Farense": 7, "Arouca": 8, "Casa Pia": 9,
            "Rio Ave": 10, "Famalicão": 11, "Estoril": 12,
            "Gil Vicente": 13, "Nacional": 14, "Moreirense": 15,
            "Boavista": 16, "Santa Clara": 17, "Estrela Amadora": 18,
        },
        "bottom6_threshold": 13,
        "top4_threshold": 4,
        "top6_threshold": 6,
        "whoscored_clubs": {
            "Benfica": "Benfica",
            "Porto": "FC Porto",
            "Sporting": "Sporting CP",
            "Braga": "SC Braga",
            "Vitoria de Guimaraes": "Vit. Guimarães",
            "Casa Pia AC": "Casa Pia",
            "AVS Futebol SAD": "Avs FS",
            "Rio Ave": "Rio Ave",
            "Famalicao": "Famalicão",
            "Moreirense": "Moreirense",
            "Arouca": "Arouca",
            "Gil Vicente": "Gil Vicente",
            "Estoril": "Estoril",
            "Estrela da Amadora": "Estrela Amadora",
            "Nacional": "Nacional",
            "Santa Clara": "Santa Clara",
            "Boavista": "Boavista",
            "Farense": "Farense",
        },
    },
}

POSITION_TO_DUMMY = {
    "Goalkeeper":         "is_goalkeeper",
    "Centre-Back":        "is_centre_back",
    "Left-Back":          "is_left_back",
    "Right-Back":         "is_right_back",
    "Defensive Midfield": "is_cdm",
    "Central Midfield":   "is_central_mid",
    "Attacking Midfield": "is_attacking_mid",
    "Left Winger":        "is_left_winger",
    "Right Winger":       "is_right_winger",
    "Left Midfield":      "is_left_winger",
    "Right Midfield":     "is_right_winger",
    "Second Striker":     "is_striker",
    "Centre-Forward":     "is_striker",
}

POS_LABELS = {
    "is_goalkeeper":    "gk",
    "is_centre_back":   "cb",
    "is_left_back":     "lb",
    "is_right_back":    "rb",
    "is_cdm":           "cdm",
    "is_central_mid":   "cm",
    "is_attacking_mid": "am",
    "is_left_winger":   "lw",
    "is_right_winger":  "rw",
    "is_striker":       "st",
}

TM_POSITION_MAP = {
    "Attack - Centre-Forward":       "Centre-Forward",
    "Attack - Left Winger":          "Left Winger",
    "Attack - Right Winger":         "Right Winger",
    "Attack - Second Striker":       "Second Striker",
    "Attack":                        "Centre-Forward",
    "Midfield - Attacking Midfield": "Attacking Midfield",
    "Midfield - Central Midfield":   "Central Midfield",
    "Midfield - Defensive Midfield": "Defensive Midfield",
    "Midfield - Left Midfield":      "Left Midfield",
    "Midfield - Right Midfield":     "Right Midfield",
    "Midfield":                      "Central Midfield",
    "Defender - Centre-Back":        "Centre-Back",
    "Defender - Left-Back":          "Left-Back",
    "Defender - Right-Back":         "Right-Back",
    "Defender":                      "Centre-Back",
    "Goalkeeper":                    "Goalkeeper",
}

AFRICAN_NATIONS = {
    "Algeria","Angola","Benin","Burkina Faso","Burundi","Cameroon",
    "Cape Verde","Central African Republic","Chad","Comoros","Congo",
    "DR Congo","Djibouti","Egypt","Equatorial Guinea","Eritrea",
    "Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau",
    "Ivory Coast","Kenya","Lesotho","Liberia","Libya","Madagascar",
    "Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique",
    "Namibia","Niger","Nigeria","Rwanda","Senegal","Sierra Leone",
    "Somalia","South Africa","South Sudan","Sudan","Tanzania","Togo",
    "Tunisia","Uganda","Zambia","Zimbabwe","Cote d'Ivoire",
}

ASIAN_NATIONS = {
    "Japan","South Korea","China","Australia","Saudi Arabia","Iran",
    "Iraq","Qatar","UAE","Uzbekistan","Vietnam","Thailand","Indonesia",
    "Malaysia","Philippines","Bahrain","Kuwait","Oman","Jordan",
    "Syria","Lebanon","North Korea","Cambodia","Myanmar","India",
}

S_AMERICAN_OTHER = {
    "Colombia","Venezuela","Ecuador","Paraguay","Uruguay",
    "Bolivia","Chile","Peru",
}


def get_nationality_group(citizenship):
    if not isinstance(citizenship, str) or not citizenship.strip():
        return "Other European"
    primary = citizenship.strip().split("  ")[0].strip()
    if primary == "Brazil":           return "Brazilian"
    if primary == "France":           return "French"
    if primary in ("England","United Kingdom"): return "English"
    if primary == "Spain":            return "Spanish"
    if primary == "Germany":          return "German"
    if primary == "Argentina":        return "Argentine"
    if primary == "Portugal":         return "Portuguese"
    if primary in AFRICAN_NATIONS:    return "African"
    if primary in ASIAN_NATIONS:      return "Asian"
    if primary in S_AMERICAN_OTHER:   return "South American (other)"
    return "Other European"


def load_profiles():
    path = PLAYER_DATA / "player_profiles/player_profiles.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[["player_id","player_name","date_of_birth","citizenship",
             "position","current_club_name"]].copy()
    df["position_tm_raw"] = df["position"].map(TM_POSITION_MAP)
    df["nationality_group"] = df["citizenship"].apply(get_nationality_group)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["player_name_clean"] = df["player_name"].str.replace(
        r"\s*\(\d+\)$", "", regex=True).str.strip()
    return df


def load_tm_positions(league_key, season):
    path = RAW_DIR / league_key / season / "transfermarkt_positions.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["player_name_clean"] = df["player_name"].str.strip()
        return df[["player_name_clean","position_tm"]].drop_duplicates("player_name_clean")
    return pd.DataFrame()


def load_whoscored(filepath, club_map):
    """Load cleaned WhoScored Stats 2 CSV (Name/Team/Age/Mins/... format)."""
    if not os.path.exists(filepath):
        print(f"  WARNING: Not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath, low_memory=False)

    # Drop repeated header rows (e.g. "Team" in Team column)
    if "Name" in df.columns:
        df = df[df["Name"].notna() & (df["Name"] != "Name")].copy()

    df = df.rename(columns={
        "Name":        "player_name_ws",
        "Team":        "club_ws_raw",
        "Age":         "age_ws",
        "Mins":        "minutes_ws",
        "Goals":       "goals_ws",
        "Assists":     "assists_ws",
        "Yel":         "yellow_cards_ws",
        "SpG":         "shots_per_game",
        "PS%":         "pass_success_pct",
        "AerialsWon":  "aerials_won",
        "Tackles":     "tackles_per_game",
        "Inter":       "interceptions_per_game",
        "Fouls":       "fouls_per_game",
        "Clear":       "clearances_per_game",
        "Drb":         "dribbles_per_game",
        "Blocks":      "blocks_per_game",
        "KeyP":        "key_passes_per_game",
        "Fouled":      "fouled_per_game",
        "AvgP":        "avg_passes_per_game",
        "Crosses":     "crosses_per_game",
        "LongB":       "long_balls_per_game",
        "ThrB":        "through_balls_per_game",
        "xG":          "xg",
        "xGDiff":      "xg_diff",
        "xG/Shots":    "xg_per_shot",
        "Disp":        "dribbled_past_per_game",
    })

    # Map WhoScored team names to TM CSV club names
    df["club_ws"] = df["club_ws_raw"].map(
        lambda x: club_map.get(str(x).strip(), str(x).strip())
    )

    # Clean numeric columns — replace "-" and commas
    num_cols = [
        "minutes_ws", "goals_ws", "assists_ws", "yellow_cards_ws",
        "shots_per_game", "pass_success_pct", "aerials_won",
        "tackles_per_game", "interceptions_per_game", "fouls_per_game",
        "clearances_per_game", "dribbles_per_game", "blocks_per_game",
        "key_passes_per_game", "fouled_per_game", "avg_passes_per_game",
        "crosses_per_game", "long_balls_per_game", "through_balls_per_game",
        "xg", "xg_diff", "xg_per_shot", "dribbled_past_per_game",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )

    df["player_name_ws"] = df["player_name_ws"].astype(str).str.strip()

    print(f"  WhoScored: {len(df)} players loaded")
    return df


def parse_keepers(filepath):
    """Parse FBref-style keeper stats CSV."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, skiprows=1, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if "Player" not in df.columns:
        # Try without skiprows
        try:
            df = pd.read_csv(filepath, low_memory=False)
            if "Player" not in df.columns:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()

    def n(col):
        if col in df.columns:
            return pd.to_numeric(
                df[col].astype(str).str.replace(",",""), errors="coerce")
        return np.nan

    result = pd.DataFrame()
    result["player_name_gk"] = df["Player"].str.strip()
    result["club_gk"] = df.get("Squad", df.get("Team", pd.Series(dtype=str)))
    result["gk_save_pct"] = n("Save%")
    result["gk_ga_per_90"] = n("GA90")

    _90s = n("90s")
    _cs  = n("CS")
    result["gk_cs_per_90"] = _cs / _90s.replace(0, np.nan)

    _pksv = n("PKsv")
    _pkatt = n("PKatt")
    result["gk_pk_save_pct"] = _pksv / _pkatt.replace(0, np.nan)

    return result[["player_name_gk","club_gk","gk_save_pct",
                   "gk_ga_per_90","gk_cs_per_90","gk_pk_save_pct"]].copy()


def fuzzy_merge_ws(base_df, ws_df, threshold=FUZZY_THRESHOLD):
    """Fuzzy match WhoScored players to base dataframe on name."""
    ws_names = ws_df["player_name_ws"].tolist()
    matched_idx = []
    for name in base_df["player_name_clean"]:
        result = process.extractOne(
            str(name), ws_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )
        matched_idx.append(ws_names.index(result[0]) if result else -1)

    ws_rows = []
    for idx in matched_idx:
        if idx >= 0:
            ws_rows.append(ws_df.iloc[idx])
        else:
            ws_rows.append(pd.Series(dtype=object))

    ws_matched = pd.DataFrame(ws_rows).reset_index(drop=True)
    for col in ws_matched.columns:
        if col not in base_df.columns:
            base_df = base_df.copy()
            base_df[col] = ws_matched[col].values

    return base_df


def build_league(league_key, season):
    cfg = LEAGUE_CONFIG[league_key]
    print(f"\n{'='*60}")
    print(f"Building: {cfg['name']} {season}")
    print(f"{'='*60}")

    out_dir = PROC_DIR / league_key / season
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load league CSV (market values + club) ──────────────
    print("1. Loading league market values...")
    league_csv = LEAGUES_DIR / cfg["leagues_folder"] / cfg["league_csv"]
    if not league_csv.exists():
        print(f"   ERROR: {league_csv} not found")
        return None

    ldf = pd.read_csv(league_csv)
    print(f"   {len(ldf)} rows in league CSV")
    print(f"   Columns: {ldf.columns.tolist()}")

    # Expected columns: player_id, player_name, club_team_at_date, market_value
    # Rename for consistency
    rename_map = {}
    for col in ldf.columns:
        cl = col.lower()
        if "market_value" in cl or "value" in cl:
            rename_map[col] = "market_value_eur"
        elif "club" in cl or "team" in cl:
            rename_map[col] = "club"
        elif "player_name" in cl or "name" in cl and "player" in cl.lower():
            rename_map[col] = "player_name_tm"

    ldf = ldf.rename(columns=rename_map)

    if cfg.get("valid_clubs") and "club" in ldf.columns:
        before = len(ldf)
        ldf = ldf[ldf["club"].isin(cfg["valid_clubs"])]
        filtered = before - len(ldf)
        if filtered > 0:
            print(f"   Filtered out {filtered} rows from non-league clubs")

    if "market_value_eur" not in ldf.columns:
        print("   ERROR: No market value column found")
        return None

    # Load supplementary promoted clubs data if present
    supp_path = LEAGUES_DIR / cfg["leagues_folder"] / "promoted_clubs_2024_25.csv"
    if supp_path.exists():
        supp = pd.read_csv(supp_path)
        supp = supp.rename(columns=rename_map)  # same rename as main CSV
        ldf = pd.concat([ldf, supp], ignore_index=True)
        ldf = ldf.sort_values("market_value_eur", ascending=False)
        ldf = ldf.drop_duplicates(subset="player_id", keep="first")
        print(f"   After adding promoted clubs: {len(ldf)} rows")

    # Keep only non-zero market values
    ldf = ldf[ldf["market_value_eur"] > 0].copy()
    print(f"   {len(ldf)} players with market value > 0")

    # ── 2. Join player profiles ─────────────────────────────────
    print("2. Joining player profiles...")
    profiles = load_profiles()
    df = ldf.merge(profiles, on="player_id", how="left")

    has_profile = df["citizenship"].notna().sum()
    print(f"   Profile matched: {has_profile}/{len(df)}")

    # ── 3. TM Positions ─────────────────────────────────────────
    print("3. Loading Transfermarkt positions...")
    tm_pos = load_tm_positions(league_key, season)
    if not tm_pos.empty:
        df = df.merge(tm_pos, left_on="player_name_clean",
                      right_on="player_name_clean", how="left")
        print(f"   TM positions exact match: {df['position_tm'].notna().sum()}")

    # Fall back to profile position
    if "position_tm" not in df.columns:
        df["position_tm"] = df["position_tm_raw"]
    else:
        df["position_tm"] = df["position_tm"].fillna(df["position_tm_raw"])

    # ── 4. WhoScored stats ──────────────────────────────────────
    print("4. Loading WhoScored stats...")

    # Preserve TM club name before WhoScored merge so club_ws cannot overwrite it
    df["club_tm"] = df["club"].copy()

    ws_file = str(LEAGUES_DIR / cfg["leagues_folder"] / cfg["whoscored_stats"])
    ws = load_whoscored(ws_file, cfg["whoscored_clubs"])

    if not ws.empty:
        print(f"   WhoScored: {len(ws)} players parsed")
        df["player_name_clean"] = df["player_name_clean"].fillna(
            df.get("player_name_tm", df.get("player_name", "")))
        df = fuzzy_merge_ws(df, ws, threshold=FUZZY_THRESHOLD)
        print(f"   xG data for: {df['xg'].notna().sum()} players")
    else:
        print("   WARNING: No WhoScored data")

    # Restore TM club name as canonical club identity
    df["club"] = df["club_tm"]

    # ── 5. GK stats ─────────────────────────────────────────────
    print("5. Loading GK stats...")
    gk_file = str(LEAGUES_DIR / cfg["leagues_folder"] / cfg["whoscored_keepers"])
    gk = parse_keepers(gk_file)
    if not gk.empty:
        gk_names = gk["player_name_gk"].tolist()
        gk_matches = []
        for name in df["player_name_clean"]:
            result = process.extractOne(str(name), gk_names,
                                        scorer=fuzz.token_sort_ratio,
                                        score_cutoff=FUZZY_THRESHOLD)
            gk_matches.append(gk_names.index(result[0]) if result else -1)

        for col in ["gk_save_pct","gk_ga_per_90","gk_cs_per_90","gk_pk_save_pct"]:
            vals = [gk.iloc[i][col] if i >= 0 and col in gk.columns
                    else np.nan for i in gk_matches]
            df[col] = vals
        print(f"   GK stats for: {df['gk_save_pct'].notna().sum()} keepers")
    else:
        for col in ["gk_save_pct","gk_ga_per_90","gk_cs_per_90","gk_pk_save_pct"]:
            df[col] = np.nan

    # ── 6. Minutes from WhoScored ───────────────────────────────
    # Use WhoScored minutes if available
    if "minutes_ws" in df.columns:
        df["minutes_played"] = df["minutes_ws"].fillna(0)
    else:
        df["minutes_played"] = 0

    # ── 7. Goals/assists from WhoScored ────────────────────────
    df["goals"] = pd.to_numeric(df["goals_ws"], errors="coerce").fillna(0) if "goals_ws" in df.columns else pd.Series(0, index=df.index)
    df["assists"] = pd.to_numeric(df["assists_ws"], errors="coerce").fillna(0) if "assists_ws" in df.columns else pd.Series(0, index=df.index)
    df["yellow_cards"] = pd.to_numeric(df["yellow_cards_ws"], errors="coerce").fillna(0) if "yellow_cards_ws" in df.columns else pd.Series(0, index=df.index)

    # ── 8. Feature engineering ──────────────────────────────────
    print("7. Engineering features...")

    # Age
    year = int(season.split("-")[0])
    midpoint = pd.Timestamp(f"{year}-12-31")
    df["age"] = ((midpoint - df["date_of_birth"]).dt.days / 365.25).round(1)

    # Per-90
    mins90 = (df["minutes_played"] / 90).replace(0, np.nan)
    df["goals_per_90"] = (df["goals"] / mins90).fillna(0)
    df["assists_per_90"] = (df["assists"] / mins90).fillna(0)

    # 500 min filter
    df = df[df["minutes_played"] >= MIN_MINUTES].copy()
    print(f"   After {MIN_MINUTES} min filter: {len(df)} players")

    if len(df) == 0:
        print("   ERROR: No players passed the minutes filter!")
        return None

    # Log MV
    df["log_market_value"] = np.log(df["market_value_eur"])

    # Nationality
    df["nationality_group"] = df["citizenship"].apply(get_nationality_group)

    # Age centering
    age_mean = df["age"].mean()
    df["age_centered"] = df["age"] - age_mean
    df["age_centered_sq"] = df["age_centered"] ** 2

    df["league"] = cfg["name"]
    df["season"] = season

    # Club from league CSV
    if "club" not in df.columns:
        df["club"] = df.get("current_club_name", "Unknown")

    # League table / tier dummies
    table = cfg["league_table_2024_25"]
    df["team_league_position"] = df["club"].map(table)

    # Try matching on club column aliases
    if df["team_league_position"].isna().any():
        for alias, full in cfg.get("whoscored_clubs", {}).items():
            mask = df["team_league_position"].isna() & (df["club"] == alias)
            if mask.any():
                df.loc[mask, "team_league_position"] = table.get(full, 99)

    df["team_league_position"] = df["team_league_position"].fillna(10)  # mid-table default

    t4 = cfg["top4_threshold"]
    t6 = cfg["top6_threshold"]
    b  = cfg["bottom6_threshold"]
    df["is_top4"]    = (df["team_league_position"] <= t4).astype(int)
    df["is_top6"]    = ((df["team_league_position"] > t4) &
                        (df["team_league_position"] <= t6)).astype(int)
    df["is_bottom6"] = (df["team_league_position"] >= b).astype(int)

    historic_clubs = cfg["historic_top"]
    df["is_historic_top"] = df["club"].apply(
        lambda c: int(any(h.lower() in str(c).lower() for h in historic_clubs)))
    promoted = cfg["promoted_2024_25"]
    df["is_promoted"] = df["club"].apply(
        lambda c: int(any(p.lower() in str(c).lower() for p in promoted)))

    # Position dummies
    df["position_tm"] = df["position_tm"].fillna("Centre-Back")
    for dummy in POSITION_TO_DUMMY.values():
        df[dummy] = 0
    for pos, dummy in POSITION_TO_DUMMY.items():
        df.loc[df["position_tm"] == pos, dummy] = 1

    # Nationality dummies
    nat_groups = {
        "Brazilian": "is_brazilian", "French": "is_french",
        "English": "is_english", "Spanish": "is_spanish",
        "German": "is_german", "Argentine": "is_argentine",
        "Portuguese": "is_portuguese", "African": "is_african",
        "Asian": "is_asian", "South American (other)": "is_south_american_other",
    }
    for ng, col in nat_groups.items():
        df[col] = (df["nationality_group"] == ng).astype(int)

    # Build all interaction columns at once (avoid fragmentation)
    STAT_COLS = [
        "goals_per_90","assists_per_90","tackles_per_game","interceptions_per_game",
        "clearances_per_game","blocks_per_game","aerials_won","key_passes_per_game",
        "dribbles_per_game","crosses_per_game","long_balls_per_game",
        "through_balls_per_game","avg_passes_per_game","shots_per_game",
        "xg","xg_per_shot","fouled_per_game",
    ]

    interaction_cols = {}
    for dummy, pos_label in POS_LABELS.items():
        pos_vec = df[dummy].values if dummy in df.columns else np.zeros(len(df))
        for stat in STAT_COLS:
            col_name = f"{stat}_{pos_label}"
            stat_vec = pd.to_numeric(df[stat], errors="coerce").fillna(0).values \
                       if stat in df.columns else np.zeros(len(df))
            interaction_cols[col_name] = pos_vec * stat_vec
        interaction_cols[f"age_{pos_label}"]    = pos_vec * df["age_centered"].values
        interaction_cols[f"age_sq_{pos_label}"] = pos_vec * df["age_centered_sq"].values

    interaction_df = pd.DataFrame(interaction_cols, index=df.index)
    df = pd.concat([df, interaction_df], axis=1)

    # Fill numeric stat cols
    fill_cols = STAT_COLS + ["pass_success_pct","fouls_per_game","dribbled_past_per_game","xg_diff"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Clean player name
    df["player_name"] = df["player_name_clean"].fillna(
        df.get("player_name_tm", df.get("player_name", "Unknown")))

    # Select final columns
    base = ["player_id","player_name","club","league","season",
            "nationality_group","age","date_of_birth","position_tm",
            "market_value_eur","log_market_value","minutes_played",
            "goals","assists","yellow_cards","goals_per_90","assists_per_90",
            "team_league_position","is_top4","is_top6","is_bottom6",
            "is_historic_top","is_promoted","age_centered","age_centered_sq"]
    nat_cols  = list(nat_groups.values())
    pos_cols  = list(POSITION_TO_DUMMY.values())
    stat_cols = [c for c in fill_cols if c in df.columns]
    gk_cols   = [c for c in ["gk_save_pct","gk_ga_per_90","gk_cs_per_90","gk_pk_save_pct"] if c in df.columns]
    iact_cols = list(interaction_cols.keys())

    all_cols = base + nat_cols + pos_cols + stat_cols + gk_cols + iact_cols
    all_cols = [c for c in all_cols if c in df.columns]
    all_cols = list(dict.fromkeys(all_cols))

    df = df[all_cols].reset_index(drop=True)

    out_path = out_dir / "master.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} players → {out_path}")
    print(f"Columns: {len(df.columns)}")

    print("\nSanity check:")
    print(f"  Market value: €{df['market_value_eur'].min():,.0f} – €{df['market_value_eur'].max():,.0f}")
    print(f"  Age range: {df['age'].min():.1f} – {df['age'].max():.1f}")
    print(f"  Positions:\n{df['position_tm'].value_counts().to_string()}")
    print(f"  Nationalities:\n{df['nationality_group'].value_counts().head(8).to_string()}")
    print(f"  Historic top: {df['is_historic_top'].sum()} | Promoted: {df['is_promoted'].sum()}")

    # Florentino check
    flor = df[df["player_name"].str.contains("Florentino", case=False, na=False)]
    if not flor.empty:
        print(f"\nFlorentino check:")
        print(flor[["player_name","club","nationality_group","position_tm",
                     "market_value_eur","minutes_played"]].to_string())

    with open(out_dir / "age_mean.json", "w") as f:
        json.dump({"age_mean": float(age_mean), "league": cfg["name"], "season": season}, f)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league",
        choices=list(LEAGUE_CONFIG.keys()) + ["all"], default="liga_portugal")
    parser.add_argument("--season", default="2024-25")
    args = parser.parse_args()

    if args.league == "all":
        for lk in LEAGUE_CONFIG:
            try:
                build_league(lk, args.season)
            except Exception as e:
                print(f"\nERROR building {lk}: {e}")
                import traceback; traceback.print_exc()
    else:
        build_league(args.league, args.season)


if __name__ == "__main__":
    main()