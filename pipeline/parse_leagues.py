"""
Unified league data pipeline for 5 leagues (2024-25 season).

Reads from  data/raw/Leagues/{League}/
Outputs to  data/processed/{league_slug}/2024-25/{merged,cleaned,features}.csv
                                          + age_mean.json

Supported leagues:
  Premier League  → premier_league
  Bundesliga      → bundesliga_2425   (separate from existing 2025-26)
  La Liga         → la_liga
  Serie A         → serie_a
  Liga Portugal   → liga_portugal
"""

import os
import re
import json
import unicodedata
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_LEAGUES = os.path.join(BASE_DIR, "data", "raw", "Leagues")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")

# ── League config ─────────────────────────────────────────────────────────────
# Each entry:  raw_subdir, processed_subdir, season, stats2_file, stats1_file, keeper_file, main_csv, ws_csv (or None)
LEAGUES = [
    {
        "name":       "Premier League",
        "raw_dir":    os.path.join(RAW_LEAGUES, "Premier League"),
        "proc_dir":   os.path.join(PROC_DIR, "premier_league", "2024-25"),
        "season":     "2024-25",
        "main_csv":   "Premier League.csv",
        "ws_csv":     "whoscored.csv",          # already-processed WhoScored
        "stats1_csv": None,                      # not used when ws_csv present
        "stats2_csv": None,
        "keeper_csv": "keepers.csv",
        # team tier data (2024-25 final PL standings)
        "top4":       {"Liverpool", "Arsenal", "Chelsea", "Manchester City"},
        "top6":       {"Liverpool", "Arsenal", "Chelsea", "Manchester City", "Nottm Forest", "Manchester Utd"},
        "bottom6":    {"Leicester City", "Ipswich Town", "Southampton", "Wolves", "Everton", "Brentford"},
        "historic_top6": {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester Utd", "Tottenham"},
        "promoted":   {"Leicester City", "Ipswich Town", "Southampton"},
        "club_map":   {
            # ws.csv club → main csv club (for fuzzy matching)
            "Manchester City": "Manchester City", "Arsenal": "Arsenal",
            "Liverpool": "Liverpool", "Chelsea": "Chelsea",
        },
    },
    {
        "name":       "Bundesliga",
        "raw_dir":    os.path.join(RAW_LEAGUES, "Bundesliga"),
        "proc_dir":   os.path.join(PROC_DIR, "bundesliga", "2024-25"),
        "season":     "2024-25",
        "main_csv":   "Bundesliga.csv",
        "ws_csv":     None,
        "stats1_csv": "Soccer Stats - Bundesliga Stats.csv",
        "stats2_csv": "Soccer Stats - Bundesliga Stats 2.csv",
        "keeper_csv": "Soccer Stats - Bundesliga Keeper Stats.csv",
        "top4":       {"Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "Eintracht Frankfurt"},
        "top6":       {"Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "Eintracht Frankfurt", "RB Leipzig", "VfB Stuttgart"},
        "bottom6":    {"VfL Bochum", "Holstein Kiel", "FC Augsburg", "Werder Bremen", "TSG Hoffenheim", "FC Heidenheim"},
        "historic_top6": {"Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen"},
        "promoted":   {"Holstein Kiel", "FC St. Pauli", "FC Heidenheim"},
        "squad_map":  {
            # FBref Squad → canonical name for tier lookup
            "Bayern Munich": "Bayern Munich", "Dortmund": "Borussia Dortmund",
            "Leverkusen": "Bayer Leverkusen", "Eintracht Frankfurt": "Eintracht Frankfurt",
            "RB Leipzig": "RB Leipzig", "Stuttgart": "VfB Stuttgart",
            "Freiburg": "SC Freiburg", "Hoffenheim": "TSG Hoffenheim",
            "Mainz 05": "1.FSV Mainz 05", "Augsburg": "FC Augsburg",
            "Gladbach": "Borussia M'Gladbach", "Union Berlin": "1. FC Union Berlin",
            "Wolfsburg": "VfL Wolfsburg", "Werder Bremen": "Werder Bremen",
            "Heidenheim": "FC Heidenheim", "Holstein Kiel": "Holstein Kiel",
            "St Pauli": "FC St. Pauli", "Bochum": "VfL Bochum",
        },
    },
    {
        "name":       "La Liga",
        "raw_dir":    os.path.join(RAW_LEAGUES, "LaLiga"),
        "proc_dir":   os.path.join(PROC_DIR, "la_liga", "2024-25"),
        "season":     "2024-25",
        "main_csv":   "LaLiga.csv",
        "ws_csv":     None,
        "stats1_csv": "Soccer Stats - La Liga Stats.csv",
        "stats2_csv": "Soccer Stats - La Liga Stats 2.csv",
        "keeper_csv": "Soccer Stats - La Liga Keeper Stats.csv",
        "top4":       {"Real Madrid", "FC Barcelona", "Atletico Madrid", "Athletic Club"},
        "top6":       {"Real Madrid", "FC Barcelona", "Atletico Madrid", "Athletic Club", "Villarreal", "Real Sociedad"},
        "bottom6":    {"Espanyol", "Las Palmas", "Valladolid", "Valencia", "Almeria", "Cadiz"},
        "historic_top6": {"Real Madrid", "FC Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Athletic Club"},
        "promoted":   {"Espanyol", "Las Palmas", "Valladolid"},
        "squad_map":  {
            "Real Madrid": "Real Madrid", "Barcelona": "FC Barcelona",
            "Atlético Madrid": "Atletico Madrid", "Athletic Club": "Athletic Club",
            "Villarreal": "Villarreal", "Sociedad": "Real Sociedad",
            "Betis": "Real Betis", "Sevilla": "Sevilla",
            "Valencia": "Valencia", "Osasuna": "Osasuna",
            "Girona": "Girona", "Getafe": "Getafe",
            "Las Palmas": "Las Palmas", "Espanyol": "Espanyol",
            "Alavés": "Deportivo Alavés", "Celta Vigo": "Celta Vigo",
            "Rayo Vallecano": "Rayo Vallecano", "Leganés": "Leganés",
            "Valladolid": "Valladolid", "Mallorca": "Mallorca",
        },
    },
    {
        "name":       "Serie A",
        "raw_dir":    os.path.join(RAW_LEAGUES, "Serie A"),
        "proc_dir":   os.path.join(PROC_DIR, "serie_a", "2024-25"),
        "season":     "2024-25",
        "main_csv":   "Serie A.csv",
        "ws_csv":     None,
        "stats1_csv": "Soccer Stats - Serie A Stats.csv",
        "stats2_csv": "Soccer Stats - Serie A Stats 2.csv",
        "keeper_csv": "Soccer Stats - Serie A Keeper Stats.csv",
        "top4":       {"Napoli", "Inter Milan", "Atalanta", "Juventus"},
        "top6":       {"Napoli", "Inter Milan", "Atalanta", "Juventus", "AC Milan", "Bologna"},
        "bottom6":    {"Como 1907", "Venezia", "Parma", "Cagliari", "Empoli", "Hellas Verona"},
        "historic_top6": {"Juventus", "Inter Milan", "AC Milan", "AS Roma", "Napoli", "Lazio"},
        "promoted":   {"Como 1907", "Venezia", "Parma"},
        "squad_map":  {
            "Inter": "Inter Milan", "Juventus": "Juventus",
            "Napoli": "Napoli", "Atalanta": "Atalanta",
            "Milan": "AC Milan", "Roma": "AS Roma",
            "Lazio": "Lazio", "Fiorentina": "Fiorentina",
            "Bologna": "Bologna", "Torino": "Torino",
            "Udinese": "Udinese", "Genoa": "Genoa",
            "Monza": "Monza", "Cagliari": "Cagliari",
            "Empoli": "Empoli", "Como": "Como 1907",
            "Venezia": "Venezia", "Parma": "Parma",
            "Lecce": "Lecce", "Hellas Verona": "Hellas Verona",
        },
    },
    {
        "name":       "Liga Portugal",
        "raw_dir":    os.path.join(RAW_LEAGUES, "Liga Portugal"),
        "proc_dir":   os.path.join(PROC_DIR, "liga_portugal", "2024-25"),
        "season":     "2024-25",
        "main_csv":   "Liga Portugal.csv",
        "ws_csv":     None,
        "stats1_csv": "Soccer Stats - Primeira Liga Stats.csv",
        "stats2_csv": "Soccer Stats - Primeira Liga Stats 2.csv",
        "keeper_csv": "Soccer Stats - Primeira Liga Keeper Stats.csv",
        "top4":       {"Sporting CP", "FC Porto", "Benfica", "SC Braga"},
        "top6":       {"Sporting CP", "FC Porto", "Benfica", "SC Braga", "Vitória SC", "Moreirense"},
        "bottom6":    {"Farense", "Nacional", "Chaves", "Casa Pia", "Rio Ave", "Estoril Praia"},
        "historic_top6": {"Benfica", "FC Porto", "Sporting CP"},
        "promoted":   {"AVS FS", "Farense", "Nacional"},
        "squad_map":  {
            "Sporting CP": "Sporting CP", "Porto": "FC Porto",
            "Benfica": "Benfica", "Braga": "SC Braga",
            "Vitória SC": "Vitória SC", "Moreirense": "Moreirense",
            "Famalicão": "Famalicão", "Arouca": "Arouca",
            "Rio Ave": "Rio Ave", "Estoril": "Estoril Praia",
            "Gil Vicente": "Gil Vicente", "Farense": "Farense",
            "Casa Pia": "Casa Pia", "Nacional": "Nacional",
            "Santa Clara": "Santa Clara", "Estrela Amadora": "Estrela Amadora",
            "AVS FS": "AVS FS", "Boavista": "Boavista",
        },
    },
]

# ── Nationality → dummy groups (same as existing pipeline) ────────────────────
NAT_GROUPS = {
    "is_english":     {"England"},
    "is_spanish":     {"Spain"},
    "is_french":      {"France"},
    "is_german":      {"Germany"},
    "is_portuguese":  {"Portugal"},
    "is_brazilian":   {"Brazil"},
    "is_argentinian": {"Argentina"},
    "is_dutch":       {"Netherlands"},
    "is_african":     {
        "Nigeria", "Senegal", "Ivory Coast", "Ghana", "Cameroon", "Morocco",
        "Mali", "Guinea", "DR Congo", "South Africa", "Tunisia", "Algeria",
        "Egypt", "Burkina Faso", "Gabon", "Gambia", "Sierra Leone", "Tanzania",
        "Uganda", "Cape Verde", "Mozambique", "Angola", "Zambia", "Zimbabwe",
        "Ethiopia", "Kenya",
    },
    "is_south_american": {
        "Brazil", "Argentina", "Colombia", "Uruguay", "Chile", "Ecuador",
        "Venezuela", "Peru", "Paraguay", "Bolivia",
    },
    "is_scandinavian": {
        "Sweden", "Norway", "Denmark", "Finland", "Iceland",
    },
}

POSITION_GROUPS = {
    "striker":   {"FW", "CF", "ST"},
    "winger":    {"LW", "RW", "LM", "RM"},
    "attmid":    {"AM", "SS", "10"},
    "cm":        {"CM", "MF", "M"},
    "cdm":       {"DM", "CDM", "6"},
    "fullback":  {"LB", "RB", "WB", "LWB", "RWB"},
    "cb":        {"CB", "DF"},
    "gk":        {"GK"},
}

# FBref Pos code → position_group
FBREF_POS_MAP = {
    "GK":    "gk",
    "DF":    "cb",
    "DF,MF": "fullback",
    "MF,DF": "fullback",
    "DF,FW": "cb",
    "MF":    "cm",
    "MF,FW": "attmid",
    "FW,MF": "winger",
    "FW":    "striker",
}

# WhoScored position_group_ws → position_group
WS_POS_MAP = {
    "striker":      "striker",
    "winger":       "winger",
    "attmid":       "attmid",
    "central_mid":  "cm",
    "cdm":          "cdm",
    "fullback":     "fullback",
    "cb":           "cb",
    "centre_back":  "cb",
    "goalkeeper":   "gk",
    "gk":           "gk",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def norm(s):
    """Normalize name: lowercase, strip accents."""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def parse_market_value(v):
    """Parse '75.0 m', '45m', '150k', or raw int → float euros."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().replace("\u202f", "").replace(",", "")
    m = re.match(r"([\d.]+)\s*([mMkK]?)", s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return np.nan
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return num * 1_000_000
    elif unit == "k":
        return num * 1_000
    return num


def parse_mins(v):
    """'1,545' → 1545.0"""
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return np.nan


def safe_float(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def safe_int(v, default=0):
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return default


def infer_pos_group(pos_code, source="fbref"):
    """Map a raw position code to our 8-class position_group."""
    if not pos_code or str(pos_code) == "nan":
        return "cm"
    pos = str(pos_code).strip()
    if source == "fbref":
        return FBREF_POS_MAP.get(pos, FBREF_POS_MAP.get(pos.split(",")[0], "cm"))
    elif source == "ws":
        return WS_POS_MAP.get(pos.lower().replace(" ", "_"), "cm")
    # WhoScored raw position code like 'AM(CLR)', 'D(CR)', etc.
    if "GK" in pos:
        return "gk"
    if pos.startswith("D(") or pos == "D":
        return "cb"
    if "WB" in pos or pos in ("D(L)", "D(R)"):
        return "fullback"
    if "DM" in pos:
        return "cdm"
    if pos.startswith("M(") or pos == "M":
        return "cm"
    if pos.startswith("AM(") or "AM" in pos:
        return "attmid"
    if "W" in pos or pos in ("M(R)", "M(L)"):
        return "winger"
    if pos.startswith("FW") or pos == "FW":
        return "striker"
    return "cm"


def read_fbref(path, is_keeper=False):
    """Read FBref CSV (2-row header) → clean DataFrame."""
    raw = pd.read_csv(path, header=None, skiprows=1, dtype=str)
    # Row 0 is the actual header
    if is_keeper:
        cols = ["Rk","Player","Nation","Pos","Squad","Age","Born","MP","Starts","Min","90s",
                "GA","GA90","SoTA","Saves","Save_pct","W","D","L","CS","CS_pct",
                "PKatt","PKA","PKsv","PKm","PK_save_pct","Matches"]
    else:
        cols = ["Rk","Player","Nation","Pos","Squad","Age","Born","MP","Starts","Min","90s",
                "Gls_t","Ast_t","G+A","G-PK","PK","PKatt","CrdY","CrdR",
                "Gls90","Ast90","G+A90","G-PK90","G+A-PK90","Matches"]
    raw = raw.iloc[1:]  # skip the header row we already named
    raw.columns = cols[: len(raw.columns)]
    raw = raw[raw["Rk"] != "Rk"].dropna(subset=["Player"]).reset_index(drop=True)
    return raw


def parse_stats2(path):
    """Parse WhoScored Stats 2 CSV → DataFrame with player_name + per-game cols."""
    df = pd.read_csv(path, dtype=str)
    df = df[df["Player"].notna() & df["Apps"].notna()].copy()

    records = []
    for _, row in df.iterrows():
        raw = str(row["Player"]).strip()
        # Remove rank prefix:  '3\nHarry Kane...'
        raw = re.sub(r"^\d+\n", "", raw)
        # Extract age: ', {2-digit-age}, '
        m = re.search(r", (\d{2}), (.*)$", raw)
        age = safe_int(m.group(1), 0) if m else 0
        pos_raw = m.group(2).strip() if m else ""
        name_club = raw[: m.start()] if m else raw

        mins = parse_mins(row.get("Mins"))
        goals = safe_float(row.get("Goals"), 0)
        assists = safe_float(row.get("Assists"), 0)
        nineties = mins / 90.0 if mins and mins > 0 else np.nan

        # Market value from Unnamed: 1
        mv_raw = row.get("Unnamed: 1") or row.get("Market Value")
        market_value_ws = parse_market_value(mv_raw)

        r = {
            "name_club":       name_club.strip(),
            "age_ws":          age,
            "pos_raw_ws":      pos_raw,
            "position_group":  infer_pos_group(pos_raw, source="raw"),
            "apps":            safe_float(row.get("Apps", "").replace("(","").split("(")[0]),
            "minutes_played":  mins,
            "goals":           goals,
            "assists":         assists,
            "goals_per_90":    goals / nineties if nineties and nineties > 0 else 0.0,
            "assists_per_90":  assists / nineties if nineties and nineties > 0 else 0.0,
            "yellow_cards":    safe_float(row.get("Yel"), 0),
            "shots_per_game":  safe_float(row.get("SpG"), np.nan),
            "pass_success_pct": safe_float(row.get("PS%"), np.nan),
            "aerials_won":     safe_float(row.get("AerialsWon"), np.nan),
            "rating":          safe_float(row.get("Rating"), np.nan),
            "tackles_per_game": safe_float(row.get("Tackles"), np.nan),
            "interceptions_per_game": safe_float(row.get("Inter"), np.nan),
            "fouls_per_game":  safe_float(row.get("Fouls"), np.nan),
            "clearances_per_game": safe_float(row.get("Clear"), np.nan),
            "dribbled_past_per_game": safe_float(row.get("Drb"), np.nan),
            "blocks_per_game": safe_float(row.get("Blocks"), np.nan),
            "key_passes_per_game": safe_float(row.get("KeyP"), np.nan),
            "dribbles_per_game": safe_float(row.get("Drb.1"), np.nan),
            "fouled_per_game": safe_float(row.get("Fouled"), np.nan),
            "avg_passes_per_game": safe_float(row.get("AvgP"), np.nan),
            "crosses_per_game": safe_float(row.get("Crosses"), np.nan),
            "long_balls_per_game": safe_float(row.get("LongB"), np.nan),
            "through_balls_per_game": safe_float(row.get("ThrB"), np.nan),
            "xg":              safe_float(row.get("xG"), np.nan),
            "xg_diff":         safe_float(row.get("xGDiff"), np.nan),
            "xg_per_shot":     safe_float(row.get("xG/Shots"), np.nan),
            "market_value_ws": market_value_ws,
        }
        records.append(r)

    return pd.DataFrame(records)


def match_names(query_names, target_names, prefix_mode=False):
    """
    Match query names to target names (normalized, accent-stripped).

    prefix_mode=True: used for Stats2 where query is 'PlayerNameClub' concatenated.
      Finds the longest target name that is a prefix of the query.
    prefix_mode=False: exact → last-name fallback.
    Returns dict: query_original → best_match_target or None.
    """
    target_norm = {norm(t): t for t in target_names}
    # Sort by length descending for prefix matching (longest wins)
    target_sorted = sorted(target_norm.items(), key=lambda x: len(x[0]), reverse=True)

    result = {}
    for q in query_names:
        qn = norm(q)
        if prefix_mode:
            matched = None
            for tn, t in target_sorted:
                if qn.startswith(tn):
                    matched = t
                    break
            result[q] = matched
        else:
            if qn in target_norm:
                result[q] = target_norm[qn]
            else:
                # Try last name match
                parts = qn.split()
                if len(parts) >= 2:
                    last = parts[-1]
                    candidates = [t for tn, t in target_norm.items() if tn.endswith(last)]
                    if len(candidates) == 1:
                        result[q] = candidates[0]
                        continue
                result[q] = None
    return result


# ── Per-league parsers ────────────────────────────────────────────────────────

def load_pl_whoscored(raw_dir):
    """Load already-processed PL whoscored.csv."""
    path = os.path.join(raw_dir, "whoscored.csv")
    df = pd.read_csv(path)
    # Ensure required columns exist
    for col in ["xg", "xg_diff"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_fbref_stats(raw_dir, stats1_file, stats2_file):
    """Load FBref Stats1 + Stats2, merge on player name, return unified DataFrame."""
    stats1_path = os.path.join(raw_dir, stats1_file)
    stats2_path = os.path.join(raw_dir, stats2_file)

    # --- FBref Stats1 ---
    df1 = read_fbref(stats1_path, is_keeper=False)
    df1["minutes_played"] = df1["Min"].apply(parse_mins)
    df1["age"]   = df1["Age"].apply(lambda x: safe_int(x, 0))
    df1["goals"] = df1["Gls_t"].apply(lambda x: safe_float(x, 0))
    df1["assists"] = df1["Ast_t"].apply(lambda x: safe_float(x, 0))
    df1["yellow_cards"] = df1["CrdY"].apply(lambda x: safe_float(x, 0))
    df1["nineties"] = df1["minutes_played"] / 90.0
    df1["goals_per_90"]   = df1["goals"] / df1["nineties"].replace(0, np.nan)
    df1["assists_per_90"] = df1["assists"] / df1["nineties"].replace(0, np.nan)
    df1["position_group_fbref"] = df1["Pos"].apply(lambda x: infer_pos_group(x, "fbref"))
    df1 = df1.rename(columns={"Player": "player_name", "Squad": "squad"})[
        ["player_name", "squad", "age", "minutes_played", "goals", "assists",
         "goals_per_90", "assists_per_90", "yellow_cards", "position_group_fbref"]
    ]

    # --- Stats2 ---
    df2 = parse_stats2(stats2_path)

    # Match Stats2 name_club → Stats1 player_name (prefix mode: 'HarryKaneBayern' → 'Harry Kane')
    fbref_names = list(df1["player_name"].unique())
    matches = match_names(df2["name_club"].tolist(), fbref_names, prefix_mode=True)
    df2["player_name"] = df2["name_club"].map(matches)

    # Merge (left join on Stats1, bring in WhoScored per-game stats from Stats2)
    ws_cols = [
        "player_name", "shots_per_game", "pass_success_pct", "aerials_won", "rating",
        "tackles_per_game", "interceptions_per_game", "fouls_per_game", "clearances_per_game",
        "dribbled_past_per_game", "blocks_per_game", "key_passes_per_game", "dribbles_per_game",
        "fouled_per_game", "avg_passes_per_game", "crosses_per_game", "long_balls_per_game",
        "through_balls_per_game", "xg", "xg_diff", "xg_per_shot",
        "position_group", "market_value_ws",
    ]
    df2_clean = df2[ws_cols].dropna(subset=["player_name"]).drop_duplicates("player_name")
    merged = df1.merge(df2_clean, on="player_name", how="left")

    # Use Stats2 position if available, fallback to FBref
    merged["position_group"] = merged["position_group"].fillna(merged["position_group_fbref"])
    merged.drop(columns=["position_group_fbref"], inplace=True)

    return merged, df1


def load_keeper_stats(raw_dir, keeper_file):
    """Parse FBref keeper CSV → dict: player_name → GK stats."""
    path = os.path.join(raw_dir, keeper_file)
    if not os.path.exists(path):
        return {}
    df = read_fbref(path, is_keeper=True)
    df["minutes_played"] = df["Min"].apply(parse_mins)
    df["nineties"]       = df["minutes_played"] / 90.0
    df["gk_save_pct"]    = df["Save_pct"].apply(lambda x: safe_float(x, np.nan))
    df["gk_ga_per_90"]   = df["GA90"].apply(lambda x: safe_float(x, np.nan))
    df["gk_sota_per_90"] = (df["SoTA"].apply(lambda x: safe_float(x, 0))
                             / df["nineties"].replace(0, np.nan))
    df["gk_cs_per_90"]   = (df["CS"].apply(lambda x: safe_float(x, 0))
                             / df["nineties"].replace(0, np.nan))
    pk_att = df["PKatt"].apply(lambda x: safe_float(x, 0))
    pk_sv  = df["PKsv"].apply(lambda x: safe_float(x, 0))
    df["gk_pk_save_pct"] = np.where(pk_att > 0, pk_sv / pk_att * 100, np.nan)

    gk_map = {}
    for _, row in df.iterrows():
        name = str(row["Player"]).strip()
        gk_map[name] = {
            "gk_save_pct":    row["gk_save_pct"],
            "gk_ga_per_90":   row["gk_ga_per_90"],
            "gk_sota_per_90": row["gk_sota_per_90"],
            "gk_cs_per_90":   row["gk_cs_per_90"],
            "gk_pk_save_pct": row["gk_pk_save_pct"],
        }
    return gk_map


def attach_gk_stats(df, gk_map):
    """Merge GK stats into player DataFrame using normalized name matching."""
    gk_names = list(gk_map.keys())
    matches = match_names(df["player_name"].tolist(), gk_names)
    gk_cols = ["gk_save_pct", "gk_ga_per_90", "gk_sota_per_90", "gk_cs_per_90", "gk_pk_save_pct"]
    for col in gk_cols:
        df[col] = np.nan
    for i, row in df.iterrows():
        matched = matches.get(row["player_name"])
        if matched and matched in gk_map:
            for col in gk_cols:
                df.at[i, col] = gk_map[matched].get(col, np.nan)
    return df


def load_market_values(raw_dir, main_csv):
    """Load main league CSV, return dict: norm(player_name) → market_value_eur."""
    path = os.path.join(raw_dir, main_csv)
    df = pd.read_csv(path, dtype=str)
    df["market_value"] = df["market_value"].apply(lambda x: safe_float(x, np.nan))
    # Take most recent value per player
    df = df.sort_values("date_unix").drop_duplicates("player_name", keep="last")
    mv_map = {}
    for _, row in df.iterrows():
        key = norm(str(row["player_name"]))
        val = safe_float(row["market_value"], np.nan)
        if not np.isnan(val) and val > 0:
            mv_map[key] = val
    return mv_map, df


def attach_market_values(df, mv_map):
    """Attach market_value_eur to player DataFrame via normalized name lookup."""
    def lookup(name):
        n = norm(str(name))
        if n in mv_map:
            return mv_map[n]
        # Try last name only
        parts = n.split()
        if len(parts) >= 2:
            last = parts[-1]
            candidates = [(k, v) for k, v in mv_map.items() if k.endswith(last)]
            if len(candidates) == 1:
                return candidates[0][1]
        return np.nan
    df["market_value_eur"] = df["player_name"].apply(lookup)
    return df


def assign_team_tiers(df, squad_col, cfg):
    """Add is_top4, is_top6, is_bottom6, is_historic_top6, is_promoted, team_league_position."""
    squad_map = cfg.get("squad_map", {})
    top4     = cfg["top4"]
    top6     = cfg["top6"]
    bottom6  = cfg["bottom6"]
    historic = cfg["historic_top6"]
    promoted = cfg["promoted"]

    def canonical(s):
        s = str(s).strip()
        return squad_map.get(s, s)

    df["club_canonical"] = df[squad_col].apply(canonical)
    df["is_top4"]            = df["club_canonical"].isin(top4).astype(int)
    df["is_top6"]            = df["club_canonical"].isin(top6).astype(int)
    df["is_bottom6"]         = df["club_canonical"].isin(bottom6).astype(int)
    df["is_historic_top6"]   = df["club_canonical"].isin(historic).astype(int)
    df["is_promoted"]        = df["club_canonical"].isin(promoted).astype(int)

    # Approximate league position (1=top, 20=bottom)
    pos_map = {}
    all_clubs = sorted(df["club_canonical"].unique())
    for club in all_clubs:
        if club in top4:
            pos_map[club] = 3
        elif club in top6:
            pos_map[club] = 5
        elif club in bottom6:
            pos_map[club] = 18
        else:
            pos_map[club] = 10
    df["team_league_position"] = df["club_canonical"].map(pos_map).fillna(10)
    return df


# ── Nationality handling ──────────────────────────────────────────────────────

def assign_nationality_dummies(df, nat_col="nationality"):
    for dummy, nations in NAT_GROUPS.items():
        df[dummy] = df[nat_col].apply(
            lambda x: 1 if str(x).strip() in nations else 0
        )
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

POSITION_ORDER = ["striker","winger","attmid","cm","cdm","fullback","cb","gk"]

def add_position_dummies(df, pos_col="position_group"):
    for p in POSITION_ORDER:
        df[f"is_{p}"] = (df[pos_col] == p).astype(int)
    return df


def add_age_features(df, age_col="age"):
    age_mean = float(df[age_col].mean())
    df["age_centered"]    = df[age_col] - age_mean
    df["age_squared"]     = df["age_centered"] ** 2
    df["age_mean_ref"]    = age_mean
    return df, age_mean


def add_interaction_features(df, pos_col="position_group"):
    """
    Add position×stat interaction terms (same logic as existing features.py).
    Goals/assists are prefix_first=True: {pp}_{stat}
    WhoScored stats are prefix_first=False: {stat}_{pp}
    Age terms per position.
    """
    ws_stats = [
        "tackles_per_game", "interceptions_per_game", "clearances_per_game",
        "blocks_per_game", "aerials_won", "key_passes_per_game", "dribbles_per_game",
        "avg_passes_per_game", "shots_per_game", "xg", "fouled_per_game",
    ]
    for pp in POSITION_ORDER:
        mask = (df[pos_col] == pp).astype(float)
        # Goals/assists
        for stat in ["goals_per_90", "assists_per_90"]:
            col = f"{pp}_{stat}"
            df[col] = df[stat].fillna(0) * mask
        # Age interactions
        df[f"{pp}_age"]    = df["age_centered"] * mask
        df[f"{pp}_age_sq"] = df["age_squared"]  * mask
        # WS stats
        for stat in ws_stats:
            if stat in df.columns:
                col = f"{stat}_{pp}"
                df[col] = df[stat].fillna(0) * mask
    return df


# ── Main pipeline per league ──────────────────────────────────────────────────

def run_league(cfg):
    name     = cfg["name"]
    raw_dir  = cfg["raw_dir"]
    proc_dir = cfg["proc_dir"]
    season   = cfg["season"]

    print(f"\n{'='*60}")
    print(f"  Processing: {name} ({season})")
    print(f"{'='*60}")

    os.makedirs(proc_dir, exist_ok=True)

    # ── 1. Load player stats ──
    if cfg.get("ws_csv"):
        # Premier League: use already-processed WhoScored CSV
        df = load_pl_whoscored(raw_dir)
        df = df.rename(columns={"club": "squad"})
        # Load market values from main CSV
        mv_map, main_df = load_market_values(raw_dir, cfg["main_csv"])
        df = attach_market_values(df, mv_map)
        # Also attach nationality from main_df
        nat_map = {}
        for _, row in main_df.iterrows():
            nat_map[norm(str(row["player_name"]))] = str(row.get("league", ""))
        # PL main CSV doesn't have nationality — leave blank for now
        df["nationality"] = df.get("nationality", "")
        # Club column
        squad_col = "squad"
    else:
        # Other leagues: parse FBref Stats1 + Stats2
        df, df1 = load_fbref_stats(raw_dir, cfg["stats1_csv"], cfg["stats2_csv"])
        # Market values
        mv_map, main_df = load_market_values(raw_dir, cfg["main_csv"])
        df = attach_market_values(df, mv_map)
        df["nationality"] = ""
        squad_col = "squad"

    # ── 2. GK stats ──
    gk_map = load_keeper_stats(raw_dir, cfg["keeper_csv"])
    df = attach_gk_stats(df, gk_map)

    # ── 3. Team tiers ──
    if cfg.get("ws_csv"):
        # PL: map WhoScored club names to canonical
        df["squad"] = df["squad"].replace({
            "Man City": "Manchester City", "Man United": "Manchester Utd",
            "Spurs": "Tottenham", "Wolves": "Wolves",
            "Nottingham Forest": "Nottm Forest", "Nott'm Forest": "Nottm Forest",
        })
    df = assign_team_tiers(df, squad_col, cfg)

    # ── 4. Basic filters ──
    df = df[df["minutes_played"].apply(lambda x: safe_float(x, 0)) >= 450].copy()
    df = df[df["market_value_eur"].notna() & (df["market_value_eur"] > 0)].copy()
    print(f"  Players after filters: {len(df)}")

    if len(df) < 30:
        print(f"  WARNING: only {len(df)} players — skipping {name}.")
        return

    # ── 5. Core derived features ──
    df["log_market_value"] = np.log(df["market_value_eur"])
    df["age"] = df["age"].apply(lambda x: safe_float(x, np.nan))
    df = df[df["age"].notna() & (df["age"] > 14) & (df["age"] < 45)].copy()
    df["age"] = df["age"].astype(float)
    df, age_mean = add_age_features(df)
    df["league"]  = name
    df["season"]  = season
    df["club"]    = df["club_canonical"]

    # ── 6. Position dummies + nationality dummies ──
    if "position_group" not in df.columns or df["position_group"].isna().all():
        # PL uses position_group_ws from whoscored.csv
        if "position_group_ws" in df.columns:
            df["position_group"] = df["position_group_ws"].apply(
                lambda x: infer_pos_group(x, source="ws")
            )
        else:
            df["position_group"] = "cm"
    df = add_position_dummies(df)
    df = assign_nationality_dummies(df)

    # ── 7. Interaction features ──
    df = add_interaction_features(df)

    # ── 8. Save merged + features ──
    merged_path   = os.path.join(proc_dir, "merged.csv")
    features_path = os.path.join(proc_dir, "features.csv")
    age_mean_path = os.path.join(proc_dir, "age_mean.json")

    df.to_csv(merged_path, index=False)
    df.to_csv(features_path, index=False)

    with open(age_mean_path, "w") as f:
        json.dump({"age_mean": age_mean}, f)

    print(f"  Saved: {features_path}")
    print(f"  Age mean: {age_mean:.2f}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    for cfg in LEAGUES:
        try:
            df = run_league(cfg)
            if df is not None:
                results[cfg["name"]] = df
        except Exception as e:
            import traceback
            print(f"ERROR in {cfg['name']}: {e}")
            traceback.print_exc()

    print(f"\n\nDone. Processed {len(results)}/{len(LEAGUES)} leagues.")
    for name, df in results.items():
        print(f"  {name}: {len(df)} players")
