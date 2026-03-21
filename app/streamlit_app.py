# pip install streamlit plotly
# streamlit run app/streamlit_app.py

"""
PitchIQ — Transfer Market Intelligence
Streamlit interactive app.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC_PL_2526 = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26")
_PROC_PL_2425 = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2024-25")
_PROC_BL_2526 = os.path.join(_BASE_DIR, "data", "processed", "bundesliga", "2025-26")

RESULTS_PATH  = os.path.join(_PROC_PL_2526, "results.csv")
FEATURES_PATH = os.path.join(_PROC_PL_2526, "features.csv")
COEF_PATH     = os.path.join(_PROC_PL_2526, "model_coefficients.json")

RESULTS_2425_PATH  = os.path.join(_PROC_PL_2425, "results.csv")
FEATURES_2425_PATH = os.path.join(_PROC_PL_2425, "features.csv")
COEF_2425_PATH     = os.path.join(_PROC_PL_2425, "model_coefficients.json")

RESULTS_BL_PATH  = os.path.join(_PROC_BL_2526, "results.csv")
FEATURES_BL_PATH = os.path.join(_PROC_BL_2526, "features.csv")
COEF_BL_PATH     = os.path.join(_PROC_BL_2526, "model_coefficients.json")

# ── Colours ───────────────────────────────────────────────────────────────────
AMBER = "#f0a500"
BLUE  = "#5c9be0"
GREEN = "#4caf7d"

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def _load_coef_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_all_results():
    specs = [
        (RESULTS_PATH,      "Premier League", "2025-26"),
        (RESULTS_2425_PATH, "Premier League", "2024-25"),
        (RESULTS_BL_PATH,   "Bundesliga",     "2025-26"),
    ]
    parts = []
    for path, league, season in specs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8")
        df["league"] = league
        df["season"] = season
        if "valuation_label" not in df.columns:
            df["valuation_label"] = df["residual"].apply(
                lambda r: "overvalued" if r > 0.15 else ("undervalued" if r < -0.15 else "fairly valued")
            )
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True, sort=False).copy()
    combined["percentile"] = (
        combined.groupby(["league", "season"])["residual"]
        .rank(pct=True, ascending=False)
        .mul(100).round(0).astype(int)
    )
    return combined


@st.cache_data
def load_all_features():
    specs = [
        (FEATURES_PATH,      "Premier League", "2025-26"),
        (FEATURES_2425_PATH, "Premier League", "2024-25"),
        (FEATURES_BL_PATH,   "Bundesliga",     "2025-26"),
    ]
    parts = []
    for path, league, season in specs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8")
        df["league"] = league
        df["season"] = season
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def fmt_eur(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    m = v / 1_000_000
    if m >= 10:
        return f"€{m:.0f}M"
    if m >= 1:
        return f"€{m:.1f}M"
    return f"€{v / 1000:.0f}k"


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


# ── Nationality / position helpers ────────────────────────────────────────────

_NAT_DUMMIES = [
    "is_brazilian", "is_french", "is_english", "is_spanish", "is_german",
    "is_argentinian", "is_portuguese", "is_african", "is_asian",
    "is_south_american_other",
]
_NAT_LABELS = {c: c.replace("is_", "").replace("_", " ").title() for c in _NAT_DUMMIES}
_NAT_LABELS["baseline"] = "Other Europe"

# Ordered list for selectbox: (display_label, dummy_col_or_None)
_NAT_OPTIONS = [
    ("Brazilian",            "is_brazilian"),
    ("French",               "is_french"),
    ("English",              "is_english"),
    ("Spanish",              "is_spanish"),
    ("German",               "is_german"),
    ("Argentinian",          "is_argentinian"),
    ("Portuguese",           "is_portuguese"),
    ("African",              "is_african"),
    ("Asian",                "is_asian"),
    ("South American Other", "is_south_american_other"),
    ("Other European",       None),
]

_POS_DUMMIES = [
    "is_striker", "is_winger", "is_attacking_mid", "is_central_mid",
    "is_cdm", "is_fullback", "is_goalkeeper",
]
_POS_LABELS = {
    "is_striker": "Striker", "is_winger": "Winger",
    "is_attacking_mid": "Attacking Mid", "is_central_mid": "Central Mid",
    "is_cdm": "CDM", "is_fullback": "Fullback", "is_goalkeeper": "Goalkeeper",
    "baseline": "Centre-Back",
}
# raw group name → display label
_POS_GROUP_TO_LABEL = {
    "striker": "Striker", "winger": "Winger",
    "attacking_mid": "Attacking Mid", "central_mid": "Central Mid",
    "cdm": "CDM", "fullback": "Fullback",
    "centre_back": "Centre-Back", "goalkeeper": "Goalkeeper",
}
# display label → position prefix used in coefficient column names
_POS_PREFIX_MAP = {
    "Striker": "striker", "Winger": "winger",
    "Attacking Mid": "attmid", "Central Mid": "cm",
    "CDM": "cdm", "Fullback": "fullback",
    "Centre-Back": "cb", "Goalkeeper": "gk",
}
# Position → key stats for bar chart
_POS_KEY_STATS = {
    "Striker":       [("Goals/90", "goals_per_90"), ("Assists/90", "assists_per_90"),
                      ("xG", "xg"), ("Shots/g", "shots_per_game")],
    "Winger":        [("Goals/90", "goals_per_90"), ("Assists/90", "assists_per_90"),
                      ("Dribbles/g", "dribbles_per_game"), ("Crosses/g", "crosses_per_game")],
    "Attacking Mid": [("Goals/90", "goals_per_90"), ("Assists/90", "assists_per_90"),
                      ("Key Pass/g", "key_passes_per_game"), ("Dribbles/g", "dribbles_per_game")],
    "Central Mid":   [("Key Pass/g", "key_passes_per_game"), ("Avg Pass/g", "avg_passes_per_game"),
                      ("Long Balls/g", "long_balls_per_game"), ("Tackles/g", "tackles_per_game")],
    "CDM":           [("Tackles/g", "tackles_per_game"), ("Intercept./g", "interceptions_per_game"),
                      ("Long Balls/g", "long_balls_per_game"), ("Clearanc./g", "clearances_per_game")],
    "Fullback":      [("Crosses/g", "crosses_per_game"), ("Dribbles/g", "dribbles_per_game"),
                      ("Tackles/g", "tackles_per_game"), ("Fouled/g", "fouled_per_game")],
    "Centre-Back":   [("Clearanc./g", "clearances_per_game"), ("Aerials Won", "aerials_won"),
                      ("Intercept./g", "interceptions_per_game"), ("Blocks/g", "blocks_per_game")],
    "Goalkeeper":    [("Save %", "gk_save_pct"), ("CS/90", "gk_cs_per_90"),
                      ("GA/90", "gk_ga_per_90"), ("SoTA/90", "gk_sota_per_90")],
}

_HISTORIC_BY_LEAGUE = {
    "Premier League": {
        "Manchester City", "Man City", "Arsenal", "Arsenal FC",
        "Liverpool", "Liverpool FC", "Chelsea", "Chelsea FC",
        "Manchester United", "Man Utd", "Manchester United FC",
        "Tottenham Hotspur", "Tottenham",
    },
    "Bundesliga": {
        "Bayern Munich", "FC Bayern München", "FC Bayern Munich",
        "Borussia Dortmund", "BVB",
        "Bayer 04 Leverkusen", "Bayer Leverkusen",
        "RB Leipzig",
    },
}
_PROMOTED_ALL = {
    "Sunderland AFC", "Sunderland", "Leeds United", "Leeds", "Burnley FC", "Burnley",
    "Leicester City", "Leicester", "Ipswich Town", "Ipswich", "Southampton FC", "Southampton",
    "Hamburger SV", "Hamburg", "1.FC Köln", "FC Köln", "Cologne",
    "1.FC Union Berlin", "FC Union Berlin", "Union Berlin",
}

# ── What-If stat definitions ───────────────────────────────────────────────────
# Tuple: (label, player_col, coef_short, prefix_first, min, max, step)
# prefix_first=True  → coef_col = f"{pp}_{coef_short}"  (goals/assists pattern)
# prefix_first=False → coef_col = f"{coef_short}_{pp}"  (WS stat×position pattern)
# prefix_first=None  → coef_col = coef_short             (direct, not pos-specific)

_WI_ATTACKING = [
    ("Goals per 90",   "goals_per_90",       "goals_per_90",   True,  0.0,  1.5,  0.05),
    ("Assists per 90", "assists_per_90",      "assists_per_90", True,  0.0,  0.6,  0.05),
    ("xG (season)",    "xg",                 "xg",             False, 0.0,  25.0, 0.5 ),
    ("Key passes/g",   "key_passes_per_game","key_passes",     False, 0.0,  3.0,  0.1 ),
    ("Shots/g",        "shots_per_game",     "shots",          False, 0.0,  4.0,  0.1 ),
    ("Dribbles/g",     "dribbles_per_game",  "dribbles",       False, 0.0,  3.0,  0.1 ),
]
_WI_DEFENSIVE = [
    ("Tackles/g",     "tackles_per_game",       "tackles",       False, 0.0, 4.0, 0.1),
    ("Intercept./g",  "interceptions_per_game", "interceptions", False, 0.0, 2.0, 0.1),
    ("Clearances/g",  "clearances_per_game",    "clearances",    False, 0.0, 8.0, 0.1),
    ("Blocks/g",      "blocks_per_game",        "blocks",        False, 0.0, 1.5, 0.1),
    ("Aerials won/g", "aerials_won",            "aerials_won",   False, 0.0, 5.0, 0.1),
]
_WI_PASSING = [
    ("Avg passes/g",    "avg_passes_per_game",   "avg_passes",    False, 0.0,  80.0, 1.0 ),
    ("Long balls/g",    "long_balls_per_game",   "long_balls",    False, 0.0,  10.0, 0.5 ),
    ("Through balls/g", "through_balls_per_game","through_balls", False, 0.0,  0.5,  0.05),
    ("Crosses/g",       "crosses_per_game",      "crosses",       False, 0.0,  2.0,  0.1 ),
    ("Pass success %",  "pass_success_pct",      "pass_success_pct", None, 50.0, 95.0, 1.0),
]
_WI_DISCIPLINE = [
    ("Fouled/g",        "fouled_per_game",       "fouled",             False, 0.0, 3.0, 0.1),
    ("Fouls/g",         "fouls_per_game",        "fouls_per_game",     None,  0.0, 2.0, 0.1),
    ("Dribbled past/g", "dribbled_past_per_game","dribbled_past_per_game", None, 0.0, 2.0, 0.1),
    ("Yellow cards",    "yellow_cards",          "yellow_cards",       None,  0,   10,  1  ),
]
_ALL_WI_DEFS = _WI_ATTACKING + _WI_DEFENSIVE + _WI_PASSING + _WI_DISCIPLINE


def _coef_col(short, pp, prefix_first):
    """Resolve the coef column name from the stat definition."""
    if prefix_first is None:
        return short
    if prefix_first:
        return f"{pp}_{short}"
    return f"{short}_{pp}"


def _wi_slider_keys(slug):
    keys = [
        f"wi_league_{slug}", f"wi_nat_{slug}",
        f"wi_contract_{slug}", f"wi_minutes_{slug}",
        f"wi_promoted_{slug}", f"wi_historic_{slug}",
    ]
    for item in _ALL_WI_DEFS:
        keys.append(f"wi_{item[1]}_{slug}")
    return keys


# ── Group helpers ──────────────────────────────────────────────────────────────

def infer_nat_group(row):
    for col in _NAT_DUMMIES:
        if col in row.index and row[col] == 1:
            return _NAT_LABELS[col]
    return "Other Europe"


def infer_nat_dummy(row):
    """Return the is_xxx dummy name for the player's nationality, or None if baseline."""
    for col in _NAT_DUMMIES:
        if col in row.index and row[col] == 1:
            return col
    return None


def infer_pos_group(row):
    """Return display label like 'Striker', 'Centre-Back', etc."""
    # Try position_group (Bundesliga) and position (PL) columns
    for col_name in ("position_group", "position"):
        pg = row.get(col_name, "")
        if not isinstance(pg, str) or not pg or pg in ("nan", ""):
            continue
        # Already a display label?
        if pg in _POS_GROUP_TO_LABEL.values():
            return pg
        # Raw group name (e.g. "striker", "centre_back")
        mapped = _POS_GROUP_TO_LABEL.get(pg)
        if mapped:
            return mapped
    # Fall back to dummy columns
    for col in _POS_DUMMIES:
        if col in row.index and row[col] == 1:
            return _POS_LABELS[col]
    return "Centre-Back"


# ── Player Detail ──────────────────────────────────────────────────────────────

def _make_wi_sliders(section_defs, player, slug):
    """Render What-If sliders for one section; return dict raw_col → new_value."""
    vals = {}
    for item in section_defs:
        label, raw_col, _short, _pf, min_v, max_v, step = item
        cur = _safe_float(player.get(raw_col), (float(min_v) + float(max_v)) / 2.0)
        cur = max(float(min_v), min(float(max_v), cur))
        key = f"wi_{raw_col}_{slug}"
        is_int = isinstance(step, int) and isinstance(min_v, int)
        if is_int:
            vals[raw_col] = st.slider(label, int(min_v), int(max_v), int(cur), int(step), key=key)
        else:
            vals[raw_col] = st.slider(label, float(min_v), float(max_v), round(cur, 4), float(step), key=key)
    return vals


def _show_player_detail(player, results_df, coefs_by_ls):
    pname     = str(player.get("player_name", ""))
    club      = str(player.get("club", ""))
    age       = player.get("age", "")
    nat       = str(player.get("nationality", ""))
    actual    = player.get("market_value_eur")
    predicted = player.get("predicted_market_value_eur")
    residual  = _safe_float(player.get("residual"))
    vlabel    = str(player.get("valuation_label", ""))
    pct       = _safe_int(player.get("percentile"))
    league    = str(player.get("league", "Premier League"))
    season    = str(player.get("season", "2025-26"))
    pos_label = infer_pos_group(player)
    slug      = pname.replace(" ", "_").replace(".", "").replace("'", "")[:30] or "p"

    # ── Header ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Player", pname)
        st.metric("Club", club)
        st.metric("League / Season", f"{league} {season}")
    with col2:
        st.metric("Age", age)
        st.metric("Nationality", nat)
        st.metric("Position", pos_label)
    with col3:
        delta_str = None
        a_f = _safe_float(actual, float("nan"))
        p_f = _safe_float(predicted, float("nan"))
        if not np.isnan(a_f) and not np.isnan(p_f):
            delta_str = fmt_eur(a_f - p_f)
        st.metric("Actual Value",    fmt_eur(actual))
        st.metric("Predicted Value", fmt_eur(predicted), delta=delta_str)
        st.metric("Percentile (league/season)", f"{pct}th")

    badge_color = {"overvalued": AMBER, "undervalued": BLUE, "fairly valued": GREEN}.get(vlabel, "#888")
    st.markdown(
        f'<span style="background:{badge_color};color:#fff;padding:4px 14px;'
        f'border-radius:4px;font-weight:bold">{vlabel.upper()}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    gauge_val = max(0.0, min(1.0, (residual + 1.5) / 3.0))
    st.caption("Residual gauge (left = undervalued, right = overvalued)")
    st.progress(gauge_val)

    # ── Position stats bar chart vs peers ──
    key_stats = _POS_KEY_STATS.get(pos_label, [])
    if key_stats:
        st.divider()
        st.subheader(f"{pos_label} — Key Stats vs Position Average")
        mask = (results_df["league"] == league) & (results_df["season"] == season)
        peers = results_df[mask].copy()
        peers["_pg"] = peers.apply(infer_pos_group, axis=1)
        peers = peers[peers["_pg"] == pos_label]

        labels, pvals, avals = [], [], []
        for lbl, col in key_stats:
            if col not in player.index:
                continue
            pv = _safe_float(player.get(col))
            av = float(peers[col].mean()) if col in peers.columns and peers[col].notna().any() else 0.0
            labels.append(lbl)
            pvals.append(round(pv, 3))
            avals.append(round(av, 3))

        if labels:
            fig = go.Figure(data=[
                go.Bar(name=pname or "Player", x=labels, y=pvals, marker_color=AMBER),
                go.Bar(name=f"{pos_label} avg",  x=labels, y=avals, marker_color=BLUE, opacity=0.7),
            ])
            fig.update_layout(
                barmode="group",
                paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0",
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=40, b=20), height=300,
            )
            st.plotly_chart(fig, width='stretch')

    # ── Stats grid ──
    st.divider()
    left_stats  = ["minutes_played", "goals", "assists", "goals_per_90", "assists_per_90",
                   "contract_months_remaining", "team_league_position"]
    right_stats = ["xg", "xg_diff", "tackles_per_game", "key_passes_per_game",
                   "pass_success_pct", "shots_per_game"]
    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption("Core stats")
        for c in left_stats:
            if c in player.index and pd.notna(player[c]):
                st.write(f"**{c.replace('_', ' ').title()}**: {player[c]}")
    with sc2:
        st.caption("WhoScored stats")
        for c in right_stats:
            if c in player.index and pd.notna(player[c]):
                st.write(f"**{c.replace('_', ' ').title()}**: {player[c]}")

    # ── What-If ──
    coefs    = coefs_by_ls.get((league, season), {})
    coef_map = coefs.get("coefficients", {}) if coefs else {}
    if not coef_map:
        st.caption("What-If unavailable — model coefficients not loaded for this league/season.")
        return

    st.divider()
    st.subheader("What-If Prediction")

    if st.button("↺ Reset to player values", key=f"reset_{slug}"):
        for k in _wi_slider_keys(slug):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Base log from model's actual prediction
    if "predicted_log_value" in player.index and pd.notna(player.get("predicted_log_value")):
        base_log = _safe_float(player["predicted_log_value"])
    else:
        base_log = np.log(max(1.0, _safe_float(predicted, 1.0)))

    pp = _POS_PREFIX_MAP.get(pos_label, "cb")

    # ── Section A: Club & Context ──
    with st.expander("Club & Context", expanded=True):
        a1, a2 = st.columns(2)
        with a1:
            has_bl_coef = "is_bundesliga" in coef_map
            if has_bl_coef:
                league_opts = ["Premier League", "Bundesliga"]
                wi_league   = st.selectbox("League", league_opts,
                                           index=1 if league == "Bundesliga" else 0,
                                           key=f"wi_league_{slug}")
            else:
                wi_league = league
                st.caption("League toggle requires combined-model coefficients (run combined regression on Model Explorer first).")

            nat_labels  = [o[0] for o in _NAT_OPTIONS]
            cur_nd      = infer_nat_dummy(player)
            cur_nl      = next((o[0] for o in _NAT_OPTIONS if o[1] == cur_nd), "Other European")
            nat_idx     = nat_labels.index(cur_nl) if cur_nl in nat_labels else len(nat_labels) - 1
            wi_nat      = st.selectbox("Nationality", nat_labels, index=nat_idx, key=f"wi_nat_{slug}")

        with a2:
            wi_contract = st.slider("Contract months remaining", 0, 96,
                                    max(0, min(96, _safe_int(player.get("contract_months_remaining"), 18))),
                                    6, key=f"wi_contract_{slug}")
            wi_minutes  = st.slider("Minutes played", 500, 3000,
                                    max(500, min(3000, _safe_int(player.get("minutes_played"), 1500))),
                                    100, key=f"wi_minutes_{slug}")

        chk1, chk2 = st.columns(2)
        wi_promoted = chk1.checkbox("Promoted club",      value=bool(player.get("is_promoted", 0)),      key=f"wi_promoted_{slug}")
        wi_historic = chk2.checkbox("Historic top club",  value=bool(player.get("is_historic_top6", 0)), key=f"wi_historic_{slug}")

    with st.expander("Attacking", expanded=False):
        att_vals = _make_wi_sliders(_WI_ATTACKING, player, slug)

    with st.expander("Defensive", expanded=False):
        def_vals = _make_wi_sliders(_WI_DEFENSIVE, player, slug)

    with st.expander("Passing", expanded=False):
        pass_vals = _make_wi_sliders(_WI_PASSING, player, slug)

    with st.expander("Discipline", expanded=False):
        disc_vals = _make_wi_sliders(_WI_DISCIPLINE, player, slug)

    # ── Compute delta ──
    delta_log = 0.0
    all_stat_vals = {**att_vals, **def_vals, **pass_vals, **disc_vals}

    for item in _ALL_WI_DEFS:
        _label, raw_col, coef_short, prefix_first, _mn, _mx, _st = item
        wi_val  = _safe_float(all_stat_vals.get(raw_col))
        old_val = _safe_float(player.get(raw_col))
        ccol    = _coef_col(coef_short, pp, prefix_first)
        delta_log += _safe_float(coef_map.get(ccol)) * (wi_val - old_val)

    # Contract & minutes
    delta_log += _safe_float(coef_map.get("contract_months_remaining")) * (
        wi_contract - _safe_float(player.get("contract_months_remaining"), 18.0)
    )
    delta_log += _safe_float(coef_map.get("minutes_played")) * (
        wi_minutes - _safe_float(player.get("minutes_played"), 1500.0)
    )

    # Promoted / historic toggles
    delta_log += _safe_float(coef_map.get("is_promoted")) * (
        (1.0 if wi_promoted else 0.0) - _safe_float(player.get("is_promoted", 0))
    )
    delta_log += _safe_float(coef_map.get("is_historic_top6")) * (
        (1.0 if wi_historic else 0.0) - _safe_float(player.get("is_historic_top6", 0))
    )

    # Nationality swap
    new_nd      = next((o[1] for o in _NAT_OPTIONS if o[0] == wi_nat), None)
    old_nat_c   = _safe_float(coef_map.get(cur_nd))   if cur_nd  else 0.0
    new_nat_c   = _safe_float(coef_map.get(new_nd))   if new_nd  else 0.0
    delta_log  += (new_nat_c - old_nat_c)

    # League toggle
    if "is_bundesliga" in coef_map:
        old_bl = 1.0 if league    == "Bundesliga" else 0.0
        new_bl = 1.0 if wi_league == "Bundesliga" else 0.0
        delta_log += _safe_float(coef_map.get("is_bundesliga")) * (new_bl - old_bl)

    wi_predicted = np.exp(base_log + delta_log)
    wi_delta     = wi_predicted - _safe_float(predicted, 0.0)
    pct_chg      = (np.exp(delta_log) - 1.0) * 100

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("What-If Predicted Value", fmt_eur(wi_predicted),
                    delta=fmt_eur(wi_delta) if abs(wi_delta) > 500 else None)
    res_col2.metric("Change from model prediction", f"{pct_chg:+.1f}%",
                    delta=f"Δ log = {delta_log:+.4f}")


# ── Page 1: Player Lookup ──────────────────────────────────────────────────────

def page_player_lookup(df, coefs_by_ls):
    st.title("Player Lookup")

    # Add computed position label column for filtering
    if "_pos_label" not in df.columns:
        df = df.copy()
        df["_pos_label"] = df.apply(infer_pos_group, axis=1)

    with st.sidebar:
        st.subheader("League / Season")
        all_leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
        all_seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
        sel_leagues = st.multiselect("League", all_leagues, default=all_leagues, key="pl_leagues")
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="pl_seasons")

        st.subheader("Sort players by")
        sort_by = st.radio(
            "Sort",
            ["Most overvalued first", "Most undervalued first", "Highest actual value first"],
            label_visibility="collapsed",
            key="pl_sort",
        )

        st.subheader("Filters")
        search = st.text_input("Search player name", "")

        pos_opts = sorted(df["_pos_label"].dropna().unique().tolist())
        pos_filter = st.multiselect("Position", options=pos_opts)

        all_clubs = sorted(df["club"].dropna().unique().tolist()) if "club" in df.columns else []
        sel_clubs = st.multiselect("Club", options=all_clubs, default=[], placeholder="All clubs")

        st.subheader("Prestige Adjustment")
        show_prestige = st.toggle(
            "Show historic top club prestige premium", value=True,
            help="OFF = strips prestige coefficient — shows performance-only value.",
        )
        if not show_prestige:
            st.caption("Predicted values adjusted to remove prestige premium.")

        nats = sorted(df["nationality"].dropna().unique().tolist()) if "nationality" in df.columns else []
        nat_filter  = st.multiselect("Nationality", options=nats)
        val_filter  = st.selectbox("Valuation", ["All", "undervalued", "overvalued", "fairly valued"])

    # Filter league/season
    working = df.copy()
    if sel_leagues:
        working = working[working["league"].isin(sel_leagues)]
    if sel_seasons:
        working = working[working["season"].isin(sel_seasons)]

    # Ensure is_historic_top6 populated
    if "is_historic_top6" not in working.columns or working["is_historic_top6"].isna().all():
        working["is_historic_top6"] = working.apply(
            lambda r: 1 if r.get("club", "") in _HISTORIC_BY_LEAGUE.get(r.get("league", ""), set()) else 0,
            axis=1,
        )

    # Prestige adjustment (per-player coef)
    working["display_predicted"] = working["predicted_market_value_eur"].copy()
    if not show_prestige:
        big6_mask = working["is_historic_top6"] == 1
        if big6_mask.any():
            def _pcoef(r):
                c = coefs_by_ls.get((r.get("league", ""), r.get("season", "")), {})
                return _safe_float(c.get("coefficients", {}).get("is_historic_top6"))
            pcoefs = working[big6_mask].apply(_pcoef, axis=1)
            lp = np.log(working.loc[big6_mask, "predicted_market_value_eur"].clip(lower=1))
            working.loc[big6_mask, "display_predicted"] = np.exp(lp - pcoefs)

    working["display_residual"] = (
        np.log(working["market_value_eur"].clip(lower=1)) -
        np.log(working["display_predicted"].clip(lower=1))
    )
    working["display_valuation"] = working["display_residual"].apply(
        lambda r: "overvalued" if r > 0.15 else ("undervalued" if r < -0.15 else "fairly valued")
    )

    res_min = float(working["display_residual"].min()) if not working.empty else -2.0
    res_max = float(working["display_residual"].max()) if not working.empty else 2.0
    with st.sidebar:
        res_range = st.slider("Residual range", res_min, res_max, (res_min, res_max), step=0.01)

    # Apply filters
    filtered = working.copy()
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]
    if pos_filter:
        filtered = filtered[filtered["_pos_label"].isin(pos_filter)]
    if sel_clubs:
        filtered = filtered[filtered["club"].isin(sel_clubs)]
    if nat_filter:
        filtered = filtered[filtered["nationality"].isin(nat_filter)]
    if val_filter != "All":
        filtered = filtered[filtered["display_valuation"] == val_filter]
    filtered = filtered[
        (filtered["display_residual"] >= res_range[0]) &
        (filtered["display_residual"] <= res_range[1])
    ]

    # Sort
    if sort_by == "Most overvalued first":
        filtered = filtered.sort_values("display_residual", ascending=False)
    elif sort_by == "Most undervalued first":
        filtered = filtered.sort_values("display_residual", ascending=True)
    else:
        filtered = filtered.sort_values("market_value_eur", ascending=False)
    filtered = filtered.reset_index(drop=True)

    if not show_prestige:
        st.info("Prestige premium removed — rankings show performance-only value.")

    # ── Build display table ──
    st.subheader(f"Players ({len(filtered)})")

    def _status(row):
        is_p = row.get("club", "") in _PROMOTED_ALL or row.get("is_promoted", 0) == 1
        is_h = row.get("is_historic_top6", 0) == 1
        if is_p and is_h:
            return "↑⭐"
        if is_p:
            return "↑"
        if is_h:
            return "⭐"
        return ""

    tbl_df = filtered.copy()
    tbl_df["Status"] = tbl_df.apply(_status, axis=1)

    display_cols = {
        "player_name": "Player", "club": "Club", "Status": "Status",
        "league": "League", "season": "Season", "_pos_label": "Position",
        "age": "Age", "market_value_eur": "Actual Value",
        "display_predicted": "Predicted Value",
        "display_residual": "Residual", "display_valuation": "Valuation",
        "percentile": "Percentile",
    }
    show_cols = [c for c in display_cols if c in tbl_df.columns]
    tbl = tbl_df[show_cols].rename(columns=display_cols).reset_index(drop=True)
    for col in ("Actual Value", "Predicted Value"):
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(fmt_eur)
    if "Residual" in tbl.columns:
        tbl["Residual"] = tbl["Residual"].round(4)

    st.caption("↑ = newly promoted  |  ⭐ = historic top club  |  Click a row to see player detail")

    # ── Clickable table (Streamlit >= 1.35) with selectbox fallback ──
    selected_player = None
    use_clickable   = False

    try:
        event = st.dataframe(
            tbl,
            width='stretch',
            height=420,
            on_select="rerun",
            selection_mode="single-row",
        )
        use_clickable = True
        sel_rows = event.selection.rows if hasattr(event, "selection") else []
        if sel_rows and sel_rows[0] < len(filtered):
            selected_player = filtered.iloc[sel_rows[0]]
    except TypeError:
        st.dataframe(tbl, width='stretch', height=420)

    # Fallback selectbox
    if not use_clickable and not filtered.empty:
        opts = ["— select a player —"] + filtered["player_name"].tolist()
        choice = st.selectbox("Select player for detail", opts, key="pl_sel_fallback")
        if choice != "— select a player —":
            m = filtered[filtered["player_name"] == choice]
            if not m.empty:
                selected_player = m.iloc[0]

    # Auto-select when exactly 1 search match (only if no row was clicked)
    if selected_player is None and len(filtered) == 1 and search:
        selected_player = filtered.iloc[0]
    elif selected_player is None and len(filtered) > 1 and search and use_clickable:
        st.caption("Multiple players matched — click a row to see the detail panel.")

    # ── Player Detail Panel ──
    if selected_player is not None:
        pname = selected_player.get("player_name", "player")
        st.divider()
        with st.expander(f"Detail — {pname}", expanded=True):
            _show_player_detail(selected_player, working, coefs_by_ls)


# ── Page 2: Model Explorer ─────────────────────────────────────────────────────

def page_model_explorer(features_df):
    st.title("Model Explorer")

    if features_df.empty:
        st.error("No features data found. Run the pipeline first.")
        return

    num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude  = {"market_value_eur", "log_market_value", "predicted_log_value",
                "residual", "predicted_market_value_eur", "is_centre_back"}

    groups = {
        "Demographics": ["age", "age_squared", "contract_months_remaining"],
        # rating removed — composite stat with high multicollinearity
        "Performance — Global": [
            "minutes_played", "xg_diff", "pass_success_pct",
            "fouls_per_game", "dribbled_past_per_game", "yellow_cards",
        ],
        "Team & League": [
            "team_league_position", "is_top4", "is_top6", "is_bottom6",
            "is_historic_top6", "is_promoted",
        ],
        "Cross-League / Season": ["is_bundesliga", "is_season_2024"],
        "Position Dummies": [
            "is_striker", "is_winger", "is_attacking_mid", "is_central_mid",
            "is_cdm", "is_fullback", "is_goalkeeper",
        ],
        "Nationality Dummies": [
            "is_brazilian", "is_french", "is_english", "is_spanish", "is_german",
            "is_argentinian", "is_portuguese", "is_african", "is_asian",
            "is_south_american_other",
        ],
        "Goals & Assists × Position": [
            c for c in num_cols
            if ("_goals_per_90" in c or "_assists_per_90" in c) and c not in exclude
        ],
        "Tackles × Position":       [c for c in num_cols if c.startswith("tackles_")],
        "Interceptions × Position": [c for c in num_cols if c.startswith("interceptions_")],
        "Clearances × Position":    [c for c in num_cols if c.startswith("clearances_")],
        "Key Passes × Position":    [c for c in num_cols if c.startswith("key_passes_")],
        "xG × Position": [
            c for c in num_cols if c.startswith("xg_") and "_pos" not in c
            and any(c.endswith(p) for p in ["_striker","_winger","_attmid","_cm","_cdm","_fullback","_cb","_gk"])
        ],
        "Shots × Position": [c for c in num_cols if c.startswith("shots_")],
        "Other WhoScored × Position": [
            c for c in num_cols
            if any(c.startswith(s) for s in [
                "aerials_won_", "dribbles_", "crosses_", "long_balls_",
                "through_balls_", "avg_passes_", "fouled_", "blocks_",
            ])
        ],
    }

    with st.sidebar:
        st.subheader("Leagues to include")
        all_leagues = sorted(features_df["league"].dropna().unique().tolist()) if "league" in features_df.columns else []
        league_checks = {}
        for lg in all_leagues:
            n = int((features_df["league"] == lg).sum())
            league_checks[lg] = st.checkbox(f"{lg} ({n})", value=True, key=f"me_lg_{lg}")

        st.subheader("Variables")
        select_all   = st.button("Select All")
        deselect_all = st.button("Deselect All")

        selected_vars = []
        for group_name, group_vars in groups.items():
            available = [v for v in group_vars if v in features_df.columns]
            if not available:
                continue
            with st.expander(group_name, expanded=(group_name == "Demographics")):
                for v in available:
                    default = True if select_all else (False if deselect_all else True)
                    if st.checkbox(v, value=default, key=f"var_{v}"):
                        selected_vars.append(v)

    sel_league_list = [lg for lg, checked in league_checks.items() if checked]
    feat = features_df.copy()
    if sel_league_list:
        feat = feat[feat["league"].isin(sel_league_list)].copy()

    if "league" in feat.columns:
        feat["is_bundesliga"]  = (feat["league"] == "Bundesliga").astype(int)
    if "season" in feat.columns:
        feat["is_season_2024"] = (feat["season"] == "2024-25").astype(int)

    st.subheader("Run Regression")
    run = st.button("▶ Run Regression", type="primary")

    if run and selected_vars:
        y_col = "log_market_value"
        if y_col not in feat.columns:
            st.error("log_market_value column missing from features data.")
            return

        y    = feat[y_col].astype(float)
        X    = feat[[v for v in selected_vars if v in feat.columns]].astype(float)
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        df_m = feat[mask].copy().reset_index(drop=True)

        X_const = sm.add_constant(X)
        res     = sm.OLS(y, X_const).fit()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²",          f"{res.rsquared:.4f}")
        m2.metric("Adj R²",      f"{res.rsquared_adj:.4f}")
        m3.metric("N",           int(res.nobs))
        m4.metric("F-statistic", f"{res.fvalue:.1f}")

        # ── Cross-league effects (prominent) ──
        cross_vars = [v for v in ["is_bundesliga", "is_season_2024"] if v in res.params.index]
        if cross_vars:
            st.subheader("Cross-League / Season Effects")
            cv_cols = st.columns(len(cross_vars))
            for i, var in enumerate(cross_vars):
                coef  = float(res.params[var])
                pval  = float(res.pvalues[var])
                prem  = (np.exp(coef) - 1.0) * 100
                label = "Bundesliga vs PL" if var == "is_bundesliga" else "2024-25 vs 2025-26"
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                cv_cols[i].metric(f"{label} premium", f"{prem:+.1f}%",
                                  delta=f"coef={coef:+.4f}  p={pval:.3f}{stars}")

            # Interpretation text
            if "is_bundesliga" in res.params.index:
                coef  = float(res.params["is_bundesliga"])
                pval  = float(res.pvalues["is_bundesliga"])
                prem  = (np.exp(coef) - 1.0) * 100
                dirn  = "more" if prem > 0 else "less"
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "(not significant)"
                st.info(
                    f"**Bundesliga vs Premier League:** Bundesliga players are valued "
                    f"**{abs(prem):.1f}% {dirn}** than equivalent PL players, "
                    f"controlling for all performance, age, and club tier factors "
                    f"(p={pval:.3f} {stars})."
                )
            if "is_season_2024" in res.params.index:
                coef  = float(res.params["is_season_2024"])
                pval  = float(res.pvalues["is_season_2024"])
                prem  = (np.exp(coef) - 1.0) * 100
                dirn  = "higher" if prem > 0 else "lower"
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "(not significant)"
                st.info(
                    f"**2024-25 vs 2025-26:** Market valuations were "
                    f"**{abs(prem):.1f}% {dirn}** in 2024-25 than 2025-26, "
                    f"all else equal (p={pval:.3f} {stars})."
                )

        # ── Coefficient table ──
        coef_df = pd.DataFrame({
            "Variable":    res.params.index,
            "Coefficient": res.params.values,
            "Std Error":   res.bse.values,
            "T-stat":      res.tvalues.values,
            "P-value":     res.pvalues.values,
        })
        coef_df["Sig"] = coef_df["P-value"].apply(
            lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        )
        coef_df = coef_df[coef_df["Variable"] != "const"]
        coef_df["_abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("_abs", ascending=False).drop(columns="_abs")

        def _color_row(row):
            if row["P-value"] < 0.05: return ["background-color: #1a3a1a"] * len(row)
            if row["P-value"] < 0.1:  return ["background-color: #3a3210"] * len(row)
            return ["background-color: #1a1a1a"] * len(row)

        st.subheader("Coefficients")
        st.dataframe(
            coef_df.style.apply(_color_row, axis=1).format({
                "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
                "T-stat": "{:.3f}", "P-value": "{:.4f}",
            }),
            width='stretch', height=400,
        )

        # ── Charts ──
        pred_log  = res.fittedvalues
        resids    = (y - pred_log).values
        pred_eur  = np.exp(pred_log.values)
        act_eur   = np.exp(y.values)

        def _vlabel(r):
            if r > 0.15:   return "overvalued"
            if r < -0.15:  return "undervalued"
            return "fairly valued"

        chart_df = pd.DataFrame({
            "residual":      resids,
            "actual_eur":    act_eur,
            "predicted_eur": pred_eur,
            "valuation":     [_vlabel(r) for r in resids],
        })
        if "league" in df_m.columns:
            chart_df["league"] = df_m["league"].values

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_h = px.histogram(
                chart_df, x="residual", nbins=40,
                title="Residual Distribution",
                color="league" if "league" in chart_df.columns else None,
                color_discrete_sequence=[AMBER],
            )
            fig_h.add_vline(x=0, line_color="white", line_dash="dash")
            fig_h.update_layout(paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0")
            st.plotly_chart(fig_h, width='stretch')
        with ch2:
            cmap = {"overvalued": AMBER, "undervalued": BLUE, "fairly valued": GREEN}
            fig_s = px.scatter(
                chart_df, x="predicted_eur", y="actual_eur",
                color="valuation", color_discrete_map=cmap,
                title="Actual vs Predicted",
                labels={"predicted_eur": "Predicted (€)", "actual_eur": "Actual (€)"},
                opacity=0.7,
            )
            mv = max(chart_df["actual_eur"].max(), chart_df["predicted_eur"].max())
            fig_s.add_shape(type="line", x0=0, y0=0, x1=mv, y1=mv,
                            line=dict(color="white", dash="dash"))
            fig_s.update_layout(paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0")
            st.plotly_chart(fig_s, width='stretch')

        if "player_name" in df_m.columns:
            chart_df["player_name"] = df_m["player_name"].values
            chart_df["club"]        = df_m["club"].values if "club" in df_m.columns else ""
            chart_df = chart_df.sort_values("residual", ascending=False)
            tu1, tu2 = st.columns(2)
            with tu1:
                st.subheader("Top 10 Overvalued")
                st.dataframe(chart_df.head(10)[["player_name", "club", "residual"]].reset_index(drop=True),
                             width='stretch')
            with tu2:
                st.subheader("Top 10 Undervalued")
                st.dataframe(chart_df.tail(10)[["player_name", "club", "residual"]].iloc[::-1].reset_index(drop=True),
                             width='stretch')

    elif run and not selected_vars:
        st.warning("Select at least one variable to run the regression.")


# ── Page 3: Nationality & Position Analysis ────────────────────────────────────

def _dummy_to_pct(coef):
    return round((np.exp(float(coef)) - 1.0) * 100, 1)


def _auto_insight(group_name, grp_df, player_col, model_pct):
    dirn = "premium" if model_pct >= 0 else "discount"
    ov   = grp_df.loc[grp_df["residual"].idxmax(), player_col]
    uv   = grp_df.loc[grp_df["residual"].idxmin(), player_col]
    return (
        f"The model estimates a **{abs(model_pct):.0f}% market {dirn}** for "
        f"{group_name} players (all else equal). "
        f"Most over-priced: **{ov}** | Most under-priced: **{uv}**."
    )


def page_nat_pos(df, coefs_by_ls):
    st.title("Nationality & Position Analysis")
    player_col = "player_name" if "player_name" in df.columns else "player"

    with st.sidebar:
        st.subheader("League / Season")
        all_leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
        all_seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
        sel_leagues = st.multiselect("League", all_leagues, default=all_leagues, key="np_leagues")
        sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="np_seasons")

    filtered = df.copy()
    if sel_leagues:
        filtered = filtered[filtered["league"].isin(sel_leagues)]
    if sel_seasons:
        filtered = filtered[filtered["season"].isin(sel_seasons)]

    primary_league = sel_leagues[0] if sel_leagues else "Premier League"
    primary_season = sel_seasons[0] if sel_seasons else "2025-26"
    coefs    = coefs_by_ls.get((primary_league, primary_season), {})
    coef_map = coefs.get("coefficients", {}) if coefs else {}

    df_view = filtered.copy()
    df_view["_nat_group"] = df_view.apply(infer_nat_group, axis=1)
    df_view["_pos_group"] = df_view.apply(infer_pos_group, axis=1)

    _nat_dummy_for = {v: k for k, v in _NAT_LABELS.items() if k != "baseline"}
    _pos_dummy_for = {v: k for k, v in _POS_LABELS.items() if k != "baseline"}

    tab_nat, tab_pos = st.tabs(["🌍 By Nationality", "⚽ By Position"])

    with tab_nat:
        st.caption(
            "**Market Premium %** = model's estimated % over-payment (+) or under-payment (−) "
            "for a player from this nationality, holding all else constant. Baseline = Other Europe (0%)."
        )
        if len(sel_leagues) > 1:
            st.info(
                f"Showing coefficients from **{primary_league} {primary_season}** "
                "(first selected league). Select a single league for league-specific coefficients."
            )

        nat_agg = []
        for grp_name, grp in df_view.groupby("_nat_group"):
            if grp.empty:
                continue
            dummy    = _nat_dummy_for.get(grp_name)
            raw_coef = _safe_float(coef_map.get(dummy)) if dummy else 0.0
            mkt_pct  = _dummy_to_pct(raw_coef)
            nat_agg.append({
                "Nationality":      grp_name,
                "Players":          len(grp),
                "Market Premium %": mkt_pct,
                "Model Coef (log)": round(raw_coef, 4),
                "Avg Actual €M":    round(grp["market_value_eur"].mean() / 1e6, 2),
                "Avg Predicted €M": round(grp["predicted_market_value_eur"].mean() / 1e6, 2),
                "Most Overvalued":  grp.loc[grp["residual"].idxmax(), player_col],
                "Most Undervalued": grp.loc[grp["residual"].idxmin(), player_col],
                "_pct":             mkt_pct,
            })
        nat_df = pd.DataFrame(nat_agg).sort_values("Market Premium %", ascending=False)

        fig_nat = px.bar(
            nat_df, x="Nationality", y="Market Premium %",
            title="Model-Estimated Market Premium/Discount by Nationality",
            color="Market Premium %",
            color_continuous_scale=[[0, BLUE], [0.5, "#888"], [1, AMBER]],
        )
        fig_nat.update_layout(
            paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0",
            coloraxis_showscale=False, xaxis_tickangle=-30,
        )
        fig_nat.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
        st.plotly_chart(fig_nat, width='stretch')
        st.dataframe(nat_df.drop(columns="_pct").reset_index(drop=True), width='stretch')
        for _, row in nat_df.iterrows():
            grp = df_view[df_view["_nat_group"] == row["Nationality"]]
            if len(grp) >= 3:
                st.info(_auto_insight(row["Nationality"], grp, player_col, row["_pct"]))

    with tab_pos:
        st.caption(
            "**Market Premium %** = model's estimated % over-payment (+) or under-payment (−) "
            "for a player in this position, holding all else constant. Baseline = Centre-Back (0%)."
        )
        hist_coef = coef_map.get("is_historic_top6")
        hist_pval = (coefs.get("pvalues", {}) or {}).get("is_historic_top6") if coefs else None
        if hist_coef is not None:
            prem_pct = _dummy_to_pct(hist_coef)
            pval_str = f"p={hist_pval:.3f}" if hist_pval is not None else "p=n/a"
            st.info(
                f"**Historic top club prestige premium: {prem_pct:+.1f}%** ({pval_str})  "
                f"— {primary_league} model estimate."
            )

        pos_agg = []
        for grp_name, grp in df_view.groupby("_pos_group"):
            if grp.empty:
                continue
            dummy    = _pos_dummy_for.get(grp_name)
            raw_coef = _safe_float(coef_map.get(dummy)) if dummy else 0.0
            mkt_pct  = _dummy_to_pct(raw_coef)
            pos_agg.append({
                "Position":         grp_name,
                "Players":          len(grp),
                "Market Premium %": mkt_pct,
                "Model Coef (log)": round(raw_coef, 4),
                "Avg Actual €M":    round(grp["market_value_eur"].mean() / 1e6, 2),
                "Avg Predicted €M": round(grp["predicted_market_value_eur"].mean() / 1e6, 2),
                "Most Overvalued":  grp.loc[grp["residual"].idxmax(), player_col],
                "Most Undervalued": grp.loc[grp["residual"].idxmin(), player_col],
                "_pct":             mkt_pct,
            })
        pos_df = pd.DataFrame(pos_agg).sort_values("Market Premium %", ascending=False)

        fig_pos = px.bar(
            pos_df, x="Position", y="Market Premium %",
            title="Model-Estimated Market Premium/Discount by Position",
            color="Market Premium %",
            color_continuous_scale=[[0, BLUE], [0.5, "#888"], [1, AMBER]],
        )
        fig_pos.update_layout(
            paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0",
            coloraxis_showscale=False,
        )
        fig_pos.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
        st.plotly_chart(fig_pos, width='stretch')
        st.dataframe(pos_df.drop(columns="_pct").reset_index(drop=True), width='stretch')

        if "xg" in df_view.columns:
            fig_xg = px.scatter(
                df_view, x="xg", y="residual", color="_pos_group",
                title="xG vs Individual Residual by Position",
                hover_data=[player_col], opacity=0.7,
            )
            fig_xg.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
            fig_xg.update_layout(paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0")
            st.plotly_chart(fig_xg, width='stretch')

        if "rating" in df_view.columns:
            fig_rat = px.scatter(
                df_view, x="rating", y="residual", color="_pos_group",
                title="Rating vs Individual Residual by Position",
                hover_data=[player_col], opacity=0.7,
            )
            fig_rat.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
            fig_rat.update_layout(paper_bgcolor="#0d1420", plot_bgcolor="#0d1420", font_color="#e8eaf0")
            st.plotly_chart(fig_rat, width='stretch')

        for _, row in pos_df.iterrows():
            grp = df_view[df_view["_pos_group"] == row["Position"]]
            if len(grp) >= 3:
                st.info(_auto_insight(row["Position"], grp, player_col, row["_pct"]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PitchIQ — Transfer Market Intelligence",
        page_icon="⚽",
        layout="wide",
    )

    results_df  = load_all_results()
    features_df = load_all_features()

    if results_df.empty:
        st.error(
            "No results data found. Run the full pipeline "
            "(merge → clean → features → regression) first."
        )
        return

    coefs_by_ls = {
        ("Premier League", "2025-26"): _load_coef_file(COEF_PATH),
        ("Premier League", "2024-25"): _load_coef_file(COEF_2425_PATH),
        ("Bundesliga",     "2025-26"): _load_coef_file(COEF_BL_PATH),
    }

    with st.sidebar:
        st.markdown("## PitchIQ")
        st.markdown("*Transfer Market Intelligence*")
        st.divider()
        page = st.radio(
            "Navigate",
            ["⚽ Player Lookup", "📊 Model Explorer", "🌍 Nationality & Position"],
            label_visibility="collapsed",
        )
        st.divider()
        leagues_loaded = sorted(results_df["league"].unique().tolist())
        st.caption(f"{len(results_df)} players | {', '.join(leagues_loaded)}")

    if page == "⚽ Player Lookup":
        page_player_lookup(results_df, coefs_by_ls)
    elif page == "📊 Model Explorer":
        page_model_explorer(features_df)
    else:
        page_nat_pos(results_df, coefs_by_ls)


if __name__ == "__main__":
    main()
