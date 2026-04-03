# pip install streamlit plotly statsmodels
# streamlit run app/streamlit_app.py

"""
The Scout Society — Transfer Market Intelligence
Complete rebuild. Single-file Streamlit app.
"""

import json
import os
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Scout Society",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

LEAGUE_DISPLAY_NAMES = {
    "premier_league": "Premier League",
    "bundesliga":     "Bundesliga",
    "la_liga":        "La Liga",
    "serie_a":        "Serie A",
    "liga_portugal":  "Liga Portugal",
}

LEAGUE_COLORS = {
    "premier_league": "#3b82f6",
    "bundesliga":     "#f59e0b",
    "la_liga":        "#ef4444",
    "serie_a":        "#10b981",
    "liga_portugal":  "#8b5cf6",
}

VALUATION_COLORS = {
    "Overvalued":  "#ef4444",
    "Undervalued": "#10b981",
    "Fair Value":  "#6b7280",
}

POSITION_TO_GROUP = {
    "Goalkeeper":         "gk",
    "Centre-Back":        "cb",
    "Left-Back":          "lb",
    "Right-Back":         "rb",
    "Defensive Midfield": "cdm",
    "Central Midfield":   "cm",
    "Attacking Midfield": "am",
    "Left Winger":        "lw",
    "Right Winger":       "rw",
    "Left Midfield":      "lw",
    "Right Midfield":     "rw",
    "Second Striker":     "st",
    "Centre-Forward":     "st",
}

STAT_TO_COEF = {
    "goals_per_90":           "goals_per_90_{pos}",
    "assists_per_90":         "assists_per_90_{pos}",
    "shots_per_game":         "shots_per_game_{pos}",
    "xg":                     "xg_{pos}",
    "key_passes_per_game":    "key_passes_per_game_{pos}",
    "dribbles_per_game":      "dribbles_per_game_{pos}",
    "fouled_per_game":        "fouled_per_game_{pos}",
    "tackles_per_game":       "tackles_per_game_{pos}",
    "interceptions_per_game": "interceptions_per_game_{pos}",
    "clearances_per_game":    "clearances_per_game_{pos}",
    "blocks_per_game":        "blocks_per_game_{pos}",
    "aerials_won":            "aerials_won_{pos}",
    "avg_passes_per_game":    "avg_passes_per_game_{pos}",
    "crosses_per_game":       "crosses_per_game_{pos}",
    "long_balls_per_game":    "long_balls_per_game_{pos}",
    "through_balls_per_game": "through_balls_per_game_{pos}",
    "xg_per_shot":            "xg_per_shot_{pos}",
}

STAT_TO_COEF_GLOBAL = {
    "minutes_played":         "minutes_played",
    "yellow_cards":           "yellow_cards",
    "pass_success_pct":       "pass_success_pct",
    "fouls_per_game":         "fouls_per_game",
    "dribbled_past_per_game": "dribbled_past_per_game",
    "xg_diff":                "xg_diff",
}

GK_STATS = {
    "gk_save_pct":    "gk_save_pct",
    "gk_ga_per_90":   "gk_ga_per_90",
    "gk_cs_per_90":   "gk_cs_per_90",
    "gk_pk_save_pct": "gk_pk_save_pct",
}

KEY_VARIABLES = [
    "is_historic_top", "is_promoted", "is_bottom6",
    "is_top4", "is_top6",
    "is_french", "is_brazilian", "is_argentine",
    "is_english", "is_spanish", "is_german", "is_portuguese",
    "is_african", "is_asian",
    "minutes_played",
]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def fmt_eur(val):
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if val >= 1e6:
        return "€{:.1f}M".format(val / 1e6)
    elif val >= 1e3:
        return "€{:.0f}k".format(val / 1e3)
    return "€{:.0f}".format(val)


def sig_stars(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def pct_premium(coef):
    try:
        return (exp(float(coef)) - 1) * 100
    except (TypeError, ValueError, OverflowError):
        return 0.0


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results(league, season="2024-25"):
    path = PROC_DIR / league / season / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_coefficients(league, season="2024-25"):
    path = PROC_DIR / league / season / "model_coefficients.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_all_results(leagues_tuple, season="2024-25"):
    dfs = []
    for league in leagues_tuple:
        df = load_results(league, season)
        if df.empty:
            continue
        df["league_key"] = league
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def apply_filters(df, position_filter, club_filter, nat_filter,
                  valuation_filter, min_minutes):
    if df.empty:
        return df
    if position_filter:
        df = df[df["position_tm"].isin(position_filter)]
    if club_filter:
        df = df[df["club"].isin(club_filter)]
    if nat_filter:
        df = df[df["nationality_group"].isin(nat_filter)]
    if valuation_filter != "All":
        df = df[df["valuation_label"] == valuation_filter]
    df = df[df["minutes_played"] >= min_minutes]
    return df



# ──────────────────────────────────────────────────────────────
# WHAT-IF HELPERS
# ──────────────────────────────────────────────────────────────
def get_wi_defaults(player):
    pid = str(player["player_id"])

    def fv(col, default=0.0):
        v = player.get(col, default)
        try:
            v = float(v)
            return default if (v != v) else v  # NaN check
        except (TypeError, ValueError):
            return float(default)

    def iv(col, default=0):
        v = player.get(col, default)
        try:
            v = int(float(v))
            return default if str(v) == "nan" else v
        except (TypeError, ValueError):
            return int(default)

    return {
        "wi_{}_age".format(pid):                    iv("age", 26),
        "wi_{}_goals_per_90".format(pid):           fv("goals_per_90"),
        "wi_{}_assists_per_90".format(pid):         fv("assists_per_90"),
        "wi_{}_shots_per_game".format(pid):         fv("shots_per_game"),
        "wi_{}_xg".format(pid):                     fv("xg"),
        "wi_{}_key_passes_per_game".format(pid):    fv("key_passes_per_game"),
        "wi_{}_dribbles_per_game".format(pid):      fv("dribbles_per_game"),
        "wi_{}_fouled_per_game".format(pid):        fv("fouled_per_game"),
        "wi_{}_tackles_per_game".format(pid):       fv("tackles_per_game"),
        "wi_{}_interceptions_per_game".format(pid): fv("interceptions_per_game"),
        "wi_{}_clearances_per_game".format(pid):    fv("clearances_per_game"),
        "wi_{}_blocks_per_game".format(pid):        fv("blocks_per_game"),
        "wi_{}_aerials_won".format(pid):            fv("aerials_won"),
        "wi_{}_avg_passes_per_game".format(pid):    fv("avg_passes_per_game"),
        "wi_{}_pass_success_pct".format(pid):       fv("pass_success_pct", 75.0),
        "wi_{}_crosses_per_game".format(pid):       fv("crosses_per_game"),
        "wi_{}_long_balls_per_game".format(pid):    fv("long_balls_per_game"),
        "wi_{}_through_balls_per_game".format(pid): fv("through_balls_per_game"),
        "wi_{}_yellow_cards".format(pid):           iv("yellow_cards"),
        "wi_{}_fouls_per_game".format(pid):         fv("fouls_per_game"),
        "wi_{}_dribbled_past_per_game".format(pid): fv("dribbled_past_per_game"),
        "wi_{}_minutes_played".format(pid):         iv("minutes_played", 1000),
        "wi_{}_xg_diff".format(pid):               fv("xg_diff"),
        "wi_{}_xg_per_shot".format(pid):           fv("xg_per_shot"),
        "wi_{}_gk_save_pct".format(pid):           fv("gk_save_pct", 70.0),
        "wi_{}_gk_ga_per_90".format(pid):          fv("gk_ga_per_90", 1.2),
        "wi_{}_gk_cs_per_90".format(pid):          fv("gk_cs_per_90", 0.3),
        "wi_{}_gk_pk_save_pct".format(pid):        fv("gk_pk_save_pct", 20.0),
    }


def compute_whatif_delta(pid, defaults, coeff_dict, pos, age_mean):
    delta = 0.0
    contributions = {}

    old_age_c = defaults["wi_{}_age".format(pid)] - age_mean
    new_age_c = st.session_state["wi_{}_age".format(pid)] - age_mean
    d_age = coeff_dict.get("age_{}".format(pos), 0) * (new_age_c - old_age_c)
    d_age_sq = coeff_dict.get("age_sq_{}".format(pos), 0) * (new_age_c ** 2 - old_age_c ** 2)
    delta += d_age + d_age_sq
    if abs(d_age + d_age_sq) > 0.001:
        contributions["age"] = d_age + d_age_sq

    for stat, coef_tmpl in STAT_TO_COEF.items():
        coef_key = coef_tmpl.format(pos=pos)
        old_val = defaults["wi_{}_{}".format(pid, stat)]
        new_val = st.session_state["wi_{}_{}".format(pid, stat)]
        c = coeff_dict.get(coef_key, 0)
        contrib = c * (new_val - old_val)
        delta += contrib
        if abs(contrib) > 0.001:
            contributions[stat] = contrib

    for stat, coef_key in STAT_TO_COEF_GLOBAL.items():
        old_val = defaults["wi_{}_{}".format(pid, stat)]
        new_val = st.session_state["wi_{}_{}".format(pid, stat)]
        c = coeff_dict.get(coef_key, 0)
        contrib = c * (new_val - old_val)
        delta += contrib
        if abs(contrib) > 0.001:
            contributions[stat] = contrib

    if pos == "gk":
        for stat, coef_key in GK_STATS.items():
            old_val = defaults["wi_{}_{}".format(pid, stat)]
            new_val = st.session_state["wi_{}_{}".format(pid, stat)]
            c = coeff_dict.get(coef_key, 0)
            contrib = c * (new_val - old_val)
            delta += contrib
            if abs(contrib) > 0.001:
                contributions[stat] = contrib

    return delta, contributions


# ──────────────────────────────────────────────────────────────
# PAGE 1 — PLAYER LOOKUP
# ──────────────────────────────────────────────────────────────
def render_player_lookup(df, selected_leagues, season,
                         position_filter, club_filter, nat_filter,
                         valuation_filter, min_minutes):
    filtered = apply_filters(
        df.copy(), position_filter, club_filter,
        nat_filter, valuation_filter, min_minutes
    )

    if filtered.empty:
        st.warning("No players match the current filters.")
        return

    # ── Metric cards ──────────────────────────────────────────
    r2_vals = []
    for lk in selected_leagues:
        c = load_coefficients(lk, season)
        if c:
            r2_vals.append(c.get("r_squared", 0))
    avg_r2 = sum(r2_vals) / len(r2_vals) if r2_vals else 0

    most_over = filtered.loc[filtered["residual"].idxmax()]
    most_under = filtered.loc[filtered["residual"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players (filtered)", len(filtered))
    c2.metric("Avg Model R²", "{:.3f}".format(avg_r2))
    c3.metric("Most Overvalued", str(most_over.get("player_name", ""))[:20])
    c4.metric("Most Undervalued", str(most_under.get("player_name", ""))[:20])

    st.markdown("---")

    # ── Color-by selector ────────────────────────────────────
    color_by = st.radio(
        "Color by",
        ["League", "Position", "Nationality", "Valuation"],
        horizontal=True,
    )

    color_col_map = {
        "League":      "league_key",
        "Position":    "position_tm",
        "Nationality": "nationality_group",
        "Valuation":   "valuation_label",
    }
    color_col = color_col_map[color_by]

    color_map = None
    if color_by == "League":
        color_map = {k: LEAGUE_COLORS[k]
                     for k in selected_leagues if k in LEAGUE_COLORS}
    elif color_by == "Valuation":
        color_map = VALUATION_COLORS

    # ── Scatter plot ─────────────────────────────────────────
    hover_data = {
        "player_name": True,
        "club": True,
        "position_tm": True,
        "market_value_eur": True,
        "predicted_market_value_eur": True,
        "residual": True,
    }

    scatter_df = filtered.copy()
    scatter_df["actual_log"] = np.log(scatter_df["market_value_eur"].clip(lower=1))
    scatter_df["pred_log"] = np.log(scatter_df["predicted_market_value_eur"].clip(lower=1))

    fig_scatter = px.scatter(
        scatter_df,
        x="predicted_market_value_eur",
        y="market_value_eur",
        color=color_col,
        color_discrete_map=color_map,
        hover_data=hover_data,
        log_x=True,
        log_y=True,
        title="Actual vs Predicted Market Value",
        labels={
            "predicted_market_value_eur": "Predicted Market Value (€)",
            "market_value_eur": "Actual Market Value (€)",
        },
        custom_data=["player_name", "club", "position_tm",
                     "market_value_eur", "predicted_market_value_eur", "residual",
                     "player_id"],
    )

    # 45° line
    mn = min(scatter_df["predicted_market_value_eur"].min(),
             scatter_df["market_value_eur"].min())
    mx = max(scatter_df["predicted_market_value_eur"].max(),
             scatter_df["market_value_eur"].max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Perfect Prediction",
            showlegend=False,
        )
    )
    fig_scatter.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Club: %{customdata[1]}<br>"
            "Position: %{customdata[2]}<br>"
            "Actual: €%{customdata[3]:,.0f}<br>"
            "Predicted: €%{customdata[4]:,.0f}<br>"
            "Residual: %{customdata[5]:.3f}<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Player table ─────────────────────────────────────────
    st.subheader("Player Table")

    display = filtered[
        ["player_name", "club", "league_key", "position_tm",
         "market_value_eur", "predicted_market_value_eur",
         "residual", "valuation_label", "player_id"]
    ].copy()

    display["League"] = display["league_key"].map(LEAGUE_DISPLAY_NAMES)
    display["Actual Value"] = display["market_value_eur"].apply(fmt_eur)
    display["Predicted Value"] = display["predicted_market_value_eur"].apply(fmt_eur)
    display["Residual %"] = (display["residual"] * 100).apply(
        lambda x: "{:+.1f}%".format(x)
    )

    show_cols = ["player_name", "club", "League", "position_tm",
                 "Actual Value", "Predicted Value", "Residual %", "valuation_label"]
    rename_map = {
        "player_name": "Player", "club": "Club",
        "position_tm": "Position", "valuation_label": "Label",
    }

    # Sort controls
    sort_col, sort_dir_col = st.columns([2, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=["Residual %", "Actual Value", "Predicted Value", "Player"],
            index=0,
            key="table_sort_by",
        )
    with sort_dir_col:
        sort_asc = st.checkbox("Ascending", value=False, key="table_sort_asc")

    sort_map = {
        "Residual %":      "residual",
        "Actual Value":    "market_value_eur",
        "Predicted Value": "predicted_market_value_eur",
        "Player":          "player_name",
    }

    # Sort on numeric/raw columns before renaming, then build display df
    display = display.sort_values(
        sort_map[sort_by], ascending=sort_asc
    ).reset_index(drop=True)

    table_df = display[show_cols + ["player_id"]].rename(columns=rename_map)

    sel = st.dataframe(
        table_df.drop(columns=["player_id"]),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )

    # Resolve selected player
    selected_player_id = st.session_state.get("selected_player_id", None)

    if sel and sel.selection and sel.selection.get("rows"):
        row_idx = sel.selection["rows"][0]
        pid_val = table_df.iloc[row_idx]["player_id"]
        new_pid = str(pid_val)

        old_pid = str(st.session_state.get("prev_player_id", ""))
        if old_pid != new_pid and old_pid:
            for k in [k for k in st.session_state if k.startswith("wi_{}_".format(old_pid))]:
                del st.session_state[k]

        st.session_state["selected_player_id"] = new_pid
        st.session_state["prev_player_id"] = new_pid
        selected_player_id = new_pid

    # ── Player detail + What-If ───────────────────────────────
    if selected_player_id:
        match = filtered[filtered["player_id"].astype(str) == str(selected_player_id)]
        if match.empty:
            match = df[df["player_id"].astype(str) == str(selected_player_id)]
        if not match.empty:
            player = match.iloc[0].to_dict()
            league_key = player.get("league_key", selected_leagues[0])
            coeff = load_coefficients(league_key, season)
            render_player_detail(player, coeff, season)


def render_player_detail(player, coeff, season):
    name = str(player.get("player_name", "Unknown"))

    with st.expander("Detail — {}".format(name), expanded=True):
        base_eur = float(player.get("predicted_market_value_eur", 0) or 0)
        actual_eur = float(player.get("market_value_eur", 0) or 0)
        base_log = float(player.get("predicted_log_value", log(base_eur) if base_eur > 0 else 0))
        age_mean = coeff.get("age_mean", 27.0)
        pos = POSITION_TO_GROUP.get(str(player.get("position_tm", "")), "cm")
        valuation_label = str(player.get("valuation_label", "Fair Value"))
        residual = float(player.get("residual", 0) or 0)

        # Info grid
        league_key = player.get("league_key", "")
        league_df = load_results(league_key, season)
        league_vals = league_df["market_value_eur"].dropna()
        if len(league_vals) > 0:
            pct_rank = int((league_vals < actual_eur).sum() / len(league_vals) * 100)
        else:
            pct_rank = 0

        diff_str = "{:+.1f}%".format((actual_eur / base_eur - 1) * 100) if base_eur > 0 else "N/A"

        badge_color = VALUATION_COLORS.get(valuation_label, "#6b7280")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("**Player**")
            st.write(name)
            st.markdown("**Club**")
            st.write(str(player.get("club", "")))
            st.markdown("**League / Season**")
            st.write("{} / {}".format(LEAGUE_DISPLAY_NAMES.get(league_key, league_key), season))
        with g2:
            st.markdown("**Age**")
            st.write("{:.1f}".format(float(player.get("age", 0) or 0)))
            st.markdown("**Nationality**")
            st.write(str(player.get("nationality_group", "")))
            st.markdown("**Position**")
            st.write(str(player.get("position_tm", "")))
        with g3:
            st.markdown("**Actual Value**")
            st.write(fmt_eur(actual_eur))
            st.markdown("**Predicted Value**")
            st.write("{} ({})".format(fmt_eur(base_eur), diff_str))
            st.markdown("**Percentile (league)**")
            st.write("{}th".format(pct_rank))

        st.markdown(
            "<span style='background:{};color:white;padding:4px 12px;"
            "border-radius:12px;font-weight:bold'>{}</span>".format(
                badge_color, valuation_label
            ),
            unsafe_allow_html=True,
        )

        # Residual gauge
        clamped = max(-1.0, min(1.0, residual))
        gauge_pct = int((clamped + 1) / 2 * 100)
        bar_color = "#ef4444" if residual > 0.25 else ("#10b981" if residual < -0.25 else "#6b7280")
        st.markdown("**Residual gauge** (−1 = very undervalued → +1 = very overvalued)")
        st.markdown(
            "<div style='background:#e5e7eb;border-radius:6px;height:18px;position:relative'>"
            "<div style='background:{color};width:{pct}%;height:100%;border-radius:6px'></div>"
            "<div style='position:absolute;top:0;left:50%;width:2px;height:100%;background:#374151'></div>"
            "</div><p style='text-align:center;font-size:0.8em;margin:2px 0'>residual = {res:+.3f}</p>".format(
                color=bar_color, pct=gauge_pct, res=residual
            ),
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── What-If panel ─────────────────────────────────────
        st.subheader("What-If Prediction")

        pid = str(player["player_id"])
        coeff_dict = coeff.get("coefficients", {})
        defaults = get_wi_defaults(player)

        # Init session state on first appearance for this player
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # Reset button BEFORE sliders
        col_reset, col_info = st.columns([1, 3])
        with col_reset:
            if st.button("🔄 Reset", key="reset_{}".format(pid)):
                for key, val in defaults.items():
                    st.session_state[key] = val
                st.rerun()
        with col_info:
            st.caption("Base prediction: {} — adjust sliders to see impact".format(
                fmt_eur(base_eur)
            ))

        # ── Sliders ───────────────────────────────────────────
        is_gk = pos == "gk"
        is_def = pos in ("gk", "cb", "lb", "rb", "cdm")

        with st.expander("Demographics", expanded=True):
            st.slider("Age", 16, 42, step=1,
                      key="wi_{}_age".format(pid))

        if not is_def:
            with st.expander("Attacking"):
                st.slider("Goals per 90", 0.0, 2.0, step=0.01,
                          key="wi_{}_goals_per_90".format(pid))
                st.slider("Assists per 90", 0.0, 1.5, step=0.01,
                          key="wi_{}_assists_per_90".format(pid))
                st.slider("Shots per game", 0.0, 5.0, step=0.1,
                          key="wi_{}_shots_per_game".format(pid))
                st.slider("xG (season)", 0.0, 35.0, step=0.1,
                          key="wi_{}_xg".format(pid))
                st.slider("Key passes per game", 0.0, 3.0, step=0.1,
                          key="wi_{}_key_passes_per_game".format(pid))
                st.slider("Dribbles per game", 0.0, 5.0, step=0.1,
                          key="wi_{}_dribbles_per_game".format(pid))
                st.slider("Fouled per game", 0.0, 4.0, step=0.1,
                          key="wi_{}_fouled_per_game".format(pid))

        if not is_gk:
            with st.expander("Defensive"):
                st.slider("Tackles per game", 0.0, 5.0, step=0.1,
                          key="wi_{}_tackles_per_game".format(pid))
                st.slider("Interceptions per game", 0.0, 4.0, step=0.1,
                          key="wi_{}_interceptions_per_game".format(pid))
                st.slider("Clearances per game", 0.0, 8.0, step=0.1,
                          key="wi_{}_clearances_per_game".format(pid))
                st.slider("Blocks per game", 0.0, 3.0, step=0.1,
                          key="wi_{}_blocks_per_game".format(pid))
                st.slider("Aerials won per game", 0.0, 6.0, step=0.1,
                          key="wi_{}_aerials_won".format(pid))

        with st.expander("Passing"):
            st.slider("Avg passes per game", 0.0, 100.0, step=1.0,
                      key="wi_{}_avg_passes_per_game".format(pid))
            st.slider("Pass success %", 50.0, 100.0, step=0.1,
                      key="wi_{}_pass_success_pct".format(pid))
            st.slider("Crosses per game", 0.0, 5.0, step=0.1,
                      key="wi_{}_crosses_per_game".format(pid))
            st.slider("Long balls per game", 0.0, 8.0, step=0.1,
                      key="wi_{}_long_balls_per_game".format(pid))
            st.slider("Through balls per game", 0.0, 2.0, step=0.1,
                      key="wi_{}_through_balls_per_game".format(pid))

        with st.expander("Discipline"):
            st.slider("Yellow cards", 0, 15, step=1,
                      key="wi_{}_yellow_cards".format(pid))
            st.slider("Fouls per game", 0.0, 5.0, step=0.1,
                      key="wi_{}_fouls_per_game".format(pid))
            st.slider("Dribbled past per game", 0.0, 3.0, step=0.1,
                      key="wi_{}_dribbled_past_per_game".format(pid))

        if is_gk:
            with st.expander("Goalkeeper", expanded=True):
                st.slider("Save %", 50.0, 95.0, step=0.1,
                          key="wi_{}_gk_save_pct".format(pid))
                st.slider("Goals allowed per 90", 0.0, 3.0, step=0.1,
                          key="wi_{}_gk_ga_per_90".format(pid))
                st.slider("Clean sheets per 90", 0.0, 1.0, step=0.01,
                          key="wi_{}_gk_cs_per_90".format(pid))
                st.slider("Penalty save %", 0.0, 100.0, step=1.0,
                          key="wi_{}_gk_pk_save_pct".format(pid))

        with st.expander("Minutes & Match"):
            st.slider("Minutes played", 500, 3500, step=10,
                      key="wi_{}_minutes_played".format(pid))
            st.slider("xG difference", -10.0, 10.0, step=0.1,
                      key="wi_{}_xg_diff".format(pid))

        # ── Compute delta after all sliders rendered ──────────
        delta, contributions = compute_whatif_delta(
            pid, defaults, coeff_dict, pos, age_mean
        )

        try:
            new_log = base_log + delta
            new_eur = exp(new_log)
        except OverflowError:
            new_eur = base_eur
        pct_change = (new_eur / base_eur - 1) * 100 if base_eur > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Base Prediction", fmt_eur(base_eur))
        with m2:
            st.metric("What-If Prediction", fmt_eur(new_eur),
                      delta="{:+.1f}%".format(pct_change))
        with m3:
            st.metric("Log Change", "{:+.4f}".format(delta))

        if contributions:
            st.caption("Top drivers of change:")
            contrib_rows = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            contrib_df = pd.DataFrame([
                {
                    "Variable": k,
                    "Impact (log)": round(v, 4),
                    "Impact (%)": round((exp(v) - 1) * 100, 2),
                }
                for k, v in contrib_rows
            ])
            st.dataframe(contrib_df, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE 2 — MODEL EXPLORER
# ──────────────────────────────────────────────────────────────
def render_model_explorer(selected_leagues, season):
    st.header("Model Explorer")

    if len(selected_leagues) > 1:
        league_choice = st.selectbox(
            "Select league to explore",
            options=selected_leagues,
            format_func=lambda x: LEAGUE_DISPLAY_NAMES[x],
        )
        explore_leagues = selected_leagues
    else:
        league_choice = selected_leagues[0]
        explore_leagues = selected_leagues

    if len(explore_leagues) > 1:
        tabs = st.tabs([LEAGUE_DISPLAY_NAMES[lk] for lk in explore_leagues])
        for tab, lk in zip(tabs, explore_leagues):
            with tab:
                render_league_explorer(lk, season)
    else:
        render_league_explorer(explore_leagues[0], season)


def render_league_explorer(league_key, season):
    coeff = load_coefficients(league_key, season)
    df = load_results(league_key, season)

    if not coeff or df.empty:
        st.warning("No data for " + LEAGUE_DISPLAY_NAMES.get(league_key, league_key))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", "{:.4f}".format(coeff.get("r_squared", 0)))
    c2.metric("Adj R²", "{:.4f}".format(coeff.get("adj_r_squared", 0)))
    c3.metric("N", coeff.get("n_obs", 0))
    c4.metric("F-statistic", "{:.2f}".format(coeff.get("f_statistic", 0)))

    eq_tab, coef_tab, diag_tab = st.tabs(["The Equation", "Coefficient Plot", "Diagnostics"])

    coeff_dict = coeff.get("coefficients", {})
    pval_dict = coeff.get("pvalues", {})
    se_dict = coeff.get("std_errors", {})

    # ── The Equation ─────────────────────────────────────────
    with eq_tab:
        sorted_vars = sorted(
            [(k, v) for k, v in coeff_dict.items() if k != "const"],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top20 = sorted_vars[:20]

        lines = ["log(value) = {:.4f} (const)".format(coeff_dict.get("const", 0))]
        for var, c in top20:
            p = pval_dict.get(var, 1.0)
            stars = sig_stars(p)
            sign = "+" if c >= 0 else ""
            lines.append("  {} {:.4f} × {}  [{}]".format(sign, c, var, stars))

        if len(sorted_vars) > 20:
            lines.append("  ... and {} more variables".format(len(sorted_vars) - 20))

        st.code("\n".join(lines), language=None)

        st.markdown("---")
        st.subheader("Rerun Regression")

        group_options = {
            "Demographics (age)": ["age_gk", "age_cb", "age_lb", "age_rb",
                                   "age_cdm", "age_cm", "age_am", "age_lw", "age_rw", "age_st"],
            "Performance": list(STAT_TO_COEF.keys()),
            "Nationality": ["is_brazilian", "is_french", "is_english", "is_spanish",
                            "is_german", "is_argentine", "is_portuguese", "is_african",
                            "is_asian", "is_south_american_other"],
            "Club Tier": ["is_top4", "is_top6", "is_bottom6", "is_historic_top", "is_promoted"],
            "Discipline": ["yellow_cards", "fouls_per_game", "dribbled_past_per_game"],
        }

        sel_groups = []
        gcols = st.columns(len(group_options))
        for i, (grp, _) in enumerate(group_options.items()):
            if gcols[i].checkbox(grp, value=True, key="grp_{}_{}".format(league_key, i)):
                sel_groups.append(grp)

        if st.button("▶ Rerun Regression", key="rerun_{}".format(league_key)):
            include_cols = ["minutes_played"]
            for grp in sel_groups:
                include_cols.extend(group_options[grp])
            available = [c for c in include_cols if c in df.columns]
            X = df[available].fillna(0)
            stds = X.std()
            X = X[[c for c in X.columns if stds[c] > 0]]
            y = df["log_market_value"].dropna()
            X = X.loc[y.index]
            try:
                Xc = sm.add_constant(X, has_constant="add")
                res = sm.OLS(y, Xc).fit()
                st.success("Rerun complete — R² = {:.4f}, Adj R² = {:.4f}".format(
                    res.rsquared, res.rsquared_adj))
                top_rerun = sorted(
                    [(k, float(v)) for k, v in res.params.items() if k != "const"],
                    key=lambda x: abs(x[1]), reverse=True
                )[:15]
                st.dataframe(
                    pd.DataFrame(
                        [{"Variable": k, "Coef": round(v, 4),
                          "p-value": round(float(res.pvalues[k]), 4),
                          "Sig": sig_stars(float(res.pvalues[k]))}
                         for k, v in top_rerun]
                    ),
                    hide_index=True, use_container_width=True,
                )
            except Exception as e:
                st.error("Regression failed: " + str(e))

    # ── Coefficient Plot ──────────────────────────────────────
    with coef_tab:
        show_all = st.checkbox("Show all variables (incl. not significant)",
                               value=False, key="coef_all_{}".format(league_key))

        coef_rows = []
        for var, c in coeff_dict.items():
            if var == "const":
                continue
            p = pval_dict.get(var, 1.0)
            se = se_dict.get(var, 0.0)
            if not show_all and p >= 0.05:
                continue
            coef_rows.append({
                "variable": var,
                "coef": c,
                "se": se,
                "pvalue": p,
                "sig": p < 0.05,
            })

        if not coef_rows:
            st.info("No significant coefficients to display.")
        else:
            coef_df = pd.DataFrame(coef_rows)
            coef_df = coef_df.reindex(
                coef_df["coef"].abs().sort_values(ascending=False).index
            ).head(30)

            coef_df["color"] = coef_df["coef"].apply(
                lambda x: "#10b981" if x > 0 else "#ef4444"
            )
            coef_df["opacity"] = coef_df["sig"].apply(lambda s: 1.0 if s else 0.4)
            coef_df["ci_upper"] = coef_df["coef"] + 1.96 * coef_df["se"]
            coef_df["ci_lower"] = coef_df["coef"] - 1.96 * coef_df["se"]

            fig_coef = go.Figure()
            for _, row in coef_df.iterrows():
                fig_coef.add_trace(go.Bar(
                    x=[row["coef"]],
                    y=[row["variable"]],
                    orientation="h",
                    marker_color=row["color"],
                    opacity=row["opacity"],
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci_upper"] - row["coef"]],
                        arrayminus=[row["coef"] - row["ci_lower"]],
                    ),
                    showlegend=False,
                    name=row["variable"],
                ))

            fig_coef.add_vline(x=0, line_color="gray", line_width=1)
            fig_coef.update_layout(
                title="Coefficient Plot (sorted by |coef|)",
                xaxis_title="Coefficient",
                height=max(400, len(coef_df) * 22),
                margin=dict(l=200),
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    # ── Diagnostics ───────────────────────────────────────────
    with diag_tab:
        if "residual" not in df.columns:
            st.warning("No residuals in results.csv")
        else:
            dc1, dc2 = st.columns(2)

            with dc1:
                # Residual distribution
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df["residual"].dropna(),
                    nbinsx=40,
                    name="Residuals",
                    marker_color="#3b82f6",
                    opacity=0.75,
                ))
                mu = df["residual"].mean()
                sigma = df["residual"].std()
                x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                normal_y = (
                    (1 / (sigma * np.sqrt(2 * np.pi)))
                    * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
                    * len(df["residual"].dropna())
                    * (df["residual"].max() - df["residual"].min()) / 40
                )
                fig_hist.add_trace(go.Scatter(
                    x=x_range, y=normal_y,
                    mode="lines", name="Normal",
                    line=dict(color="#ef4444", width=2),
                ))
                fig_hist.update_layout(
                    title="Residual Distribution",
                    xaxis_title="Residual", yaxis_title="Count",
                    height=300,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with dc2:
                # Actual vs predicted (small)
                fig_sm = px.scatter(
                    df,
                    x="predicted_market_value_eur",
                    y="market_value_eur",
                    color="valuation_label",
                    color_discrete_map=VALUATION_COLORS,
                    log_x=True, log_y=True,
                    title="Actual vs Predicted",
                    labels={
                        "predicted_market_value_eur": "Predicted (€)",
                        "market_value_eur": "Actual (€)",
                    },
                )
                fig_sm.update_layout(height=300)
                st.plotly_chart(fig_sm, use_container_width=True)

            dc3, dc4 = st.columns(2)

            with dc3:
                # Residuals vs fitted
                fig_rv = px.scatter(
                    df,
                    x="predicted_log_value",
                    y="residual",
                    color="valuation_label",
                    color_discrete_map=VALUATION_COLORS,
                    title="Residuals vs Fitted",
                    labels={
                        "predicted_log_value": "Fitted (log)",
                        "residual": "Residual",
                    },
                )
                fig_rv.add_hline(y=0, line_color="gray", line_dash="dash")
                fig_rv.update_layout(height=300)
                st.plotly_chart(fig_rv, use_container_width=True)

            with dc4:
                st.markdown("**Model Stats**")
                summary_path = (PROC_DIR / league_key / season / "model_summary.txt")
                if summary_path.exists():
                    with open(summary_path) as f:
                        txt = f.read()
                    # Extract key stats
                    stats_to_find = [
                        "Omnibus:", "Prob(Omnibus):", "Jarque-Bera",
                        "Prob(JB):", "Cond. No.",
                    ]
                    lines_found = []
                    for line in txt.split("\n"):
                        if any(s in line for s in stats_to_find):
                            lines_found.append(line.strip())
                    if lines_found:
                        st.code("\n".join(lines_found), language=None)
                    else:
                        st.code(txt[:800], language=None)
                else:
                    st.info("model_summary.txt not found")


# ──────────────────────────────────────────────────────────────
# PAGE 3 — NATIONALITY & POSITION
# ──────────────────────────────────────────────────────────────
def render_nationality_position(df, selected_leagues, season):
    st.header("Nationality & Position Analysis")

    if df.empty:
        st.warning("No data loaded.")
        return

    # ── Club tier cards ────────────────────────────────────────
    st.subheader("Club Tier Premiums")

    tier_vars = [
        ("is_historic_top", "Historic Top Clubs"),
        ("is_promoted", "Promoted"),
        ("is_bottom6", "Bottom 6"),
        ("is_top4", "Current Top 4"),
    ]

    tc_cols = st.columns(4)
    for col, (var, label) in zip(tc_cols, tier_vars):
        coefs, pvals = [], []
        for lk in selected_leagues:
            c = load_coefficients(lk, season)
            if c and var in c.get("coefficients", {}):
                coefs.append(c["coefficients"][var])
                pvals.append(c["pvalues"].get(var, 1.0))

        if coefs:
            avg_coef = sum(coefs) / len(coefs)
            avg_p = sum(pvals) / len(pvals)
            prem = pct_premium(avg_coef)
            card_color = "#10b981" if avg_coef > 0 else "#ef4444"
            col.markdown(
                "<div style='background:{color};padding:12px;border-radius:8px;"
                "text-align:center;color:white'>"
                "<div style='font-size:1.3em;font-weight:bold'>{prem:+.1f}%</div>"
                "<div style='font-size:0.85em'>{label}</div>"
                "<div style='font-size:0.75em'>{stars}</div>"
                "</div>".format(
                    color=card_color, prem=prem,
                    label=label, stars=sig_stars(avg_p)
                ),
                unsafe_allow_html=True,
            )
        else:
            col.info(label + ": N/A")

    st.markdown("---")

    # ── Nationality section ────────────────────────────────────
    st.subheader("Nationality Analysis")

    # nationality_group string in data → OLS coefficient key
    # "Other European" is the baseline (always 0%)
    NAT_TO_COEF_KEY = {
        "Brazilian":              "is_brazilian",
        "French":                 "is_french",
        "English":                "is_english",
        "Spanish":                "is_spanish",
        "German":                 "is_german",
        "Argentine":              "is_argentine",
        "Portuguese":             "is_portuguese",
        "African":                "is_african",
        "Asian":                  "is_asian",
        "South American (other)": "is_south_american_other",
        "Other European":         None,
    }

    # Load coefficients for all selected leagues up front
    league_coeffs = {lk: load_coefficients(lk, season) for lk in selected_leagues}

    # Use actual nationality groups present in data (fixes Bug 3)
    nat_groups_in_data = sorted(df["nationality_group"].dropna().unique().tolist())

    nat_nc, nat_tc = st.columns([1, 1])

    # ── Bar chart (Bug 4 fix) ─────────────────────────────────
    with nat_nc:
        if len(selected_leagues) == 1:
            lk = selected_leagues[0]
            c = league_coeffs[lk]
            bar_rows = []
            for nat in nat_groups_in_data:
                coef_key = NAT_TO_COEF_KEY.get(nat)
                if coef_key is None:
                    coef_pct = 0.0
                else:
                    coef_v = c.get("coefficients", {}).get(coef_key, 0)
                    coef_pct = (np.exp(coef_v) - 1) * 100
                bar_rows.append({"Nationality": nat, "Coef %": round(coef_pct, 1)})
            bar_df = pd.DataFrame(bar_rows).sort_values("Coef %", ascending=True)
            fig_nat = px.bar(
                bar_df,
                x="Coef %",
                y="Nationality",
                orientation="h",
                color="Coef %",
                color_continuous_scale=["#ef4444", "#6b7280", "#10b981"],
                color_continuous_midpoint=0,
                title="Nationality Premium vs Baseline ({})".format(
                    LEAGUE_DISPLAY_NAMES[lk]
                ),
            )
            fig_nat.add_vline(x=0, line_color="gray", line_width=1)
            fig_nat.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_nat, use_container_width=True)
        else:
            multi_rows = []
            for lk in selected_leagues:
                c = league_coeffs[lk]
                for nat in nat_groups_in_data:
                    coef_key = NAT_TO_COEF_KEY.get(nat)
                    if coef_key is None:
                        coef_pct = 0.0
                    else:
                        coef_v = c.get("coefficients", {}).get(coef_key, 0)
                        coef_pct = (np.exp(coef_v) - 1) * 100
                    multi_rows.append({
                        "Nationality": nat,
                        "League": LEAGUE_DISPLAY_NAMES[lk],
                        "Coef %": round(coef_pct, 1),
                    })
            multi_df = pd.DataFrame(multi_rows)
            fig_multi = px.bar(
                multi_df,
                x="Coef %",
                y="Nationality",
                color="League",
                barmode="group",
                orientation="h",
                title="Nationality Premium by League",
            )
            fig_multi.add_vline(x=0, line_color="gray", line_width=1)
            fig_multi.update_layout(height=400)
            st.plotly_chart(fig_multi, use_container_width=True)

    # ── Table (Bug 1 + Bug 2 + Bug 3 fix) ────────────────────
    with nat_tc:
        table_rows = []
        for nat in nat_groups_in_data:
            nat_df_all = df[df["nationality_group"] == nat]
            n_players = len(nat_df_all)

            # Weighted median residual per league (Bug 2 fix)
            weighted_resid = 0.0
            total_w = 0
            for lk in selected_leagues:
                if "league_key" in nat_df_all.columns:
                    lk_nat = nat_df_all[nat_df_all["league_key"] == lk]
                else:
                    lk_nat = nat_df_all
                cnt = len(lk_nat)
                if cnt > 0:
                    weighted_resid += lk_nat["residual"].median() * cnt
                    total_w += cnt
            med_resid_pct = (weighted_resid / total_w * 100) if total_w > 0 else 0.0

            coef_key = NAT_TO_COEF_KEY.get(nat)

            row = {"Nationality": nat, "Players": n_players}

            # Bug 1 fix: single league → one coef column; multi → one per league
            if len(selected_leagues) == 1:
                lk = selected_leagues[0]
                c = league_coeffs[lk]
                if coef_key is None:
                    row["Coef %"] = "0.0% (baseline)"
                else:
                    coef_v = c.get("coefficients", {}).get(coef_key, 0)
                    row["Coef %"] = "{:+.1f}%".format((np.exp(coef_v) - 1) * 100)
            else:
                for lk in selected_leagues:
                    c = league_coeffs[lk]
                    col_name = LEAGUE_DISPLAY_NAMES[lk] + " Coef %"
                    if coef_key is None:
                        row[col_name] = "0.0% (baseline)"
                    else:
                        coef_v = c.get("coefficients", {}).get(coef_key, 0)
                        row[col_name] = "{:+.1f}%".format((np.exp(coef_v) - 1) * 100)

            row["Median Residual %"] = "{:+.1f}%".format(med_resid_pct)

            if n_players > 0:
                most_over = nat_df_all.loc[nat_df_all["residual"].idxmax(), "player_name"]
                most_under = nat_df_all.loc[nat_df_all["residual"].idxmin(), "player_name"]
            else:
                most_over = "-"
                most_under = "-"
            row["Most Overvalued"] = str(most_over)[:20]
            row["Most Undervalued"] = str(most_under)[:20]

            table_rows.append(row)

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    # ── Position section ────────────────────────────────────────
    st.subheader("Position Analysis")

    pos_data = []
    for pos_tm in df["position_tm"].dropna().unique():
        pos_players = df[df["position_tm"] == pos_tm]
        pos_data.append({
            "Position": pos_tm,
            "Players": len(pos_players),
            "Avg Value": pos_players["market_value_eur"].mean(),
            "Avg Residual %": pos_players["residual"].mean() * 100,
        })

    pos_df = pd.DataFrame(pos_data).sort_values("Avg Value", ascending=False)

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_pv = px.bar(
            pos_df, x="Position", y="Avg Value",
            title="Avg Market Value by Position",
            labels={"Avg Value": "Avg Market Value (€)"},
        )
        fig_pv.update_layout(height=300)
        st.plotly_chart(fig_pv, use_container_width=True)

    with pc2:
        pos_df["Color"] = pos_df["Avg Residual %"].apply(
            lambda x: "#10b981" if x > 0 else "#ef4444"
        )
        fig_pr = go.Figure()
        for _, row in pos_df.iterrows():
            fig_pr.add_trace(go.Bar(
                x=[row["Position"]],
                y=[row["Avg Residual %"]],
                marker_color=row["Color"],
                showlegend=False,
            ))
        fig_pr.update_layout(
            title="Avg Residual % by Position",
            yaxis_title="Avg Residual %",
            height=300,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    pos_display = pos_df.copy()
    pos_display["Avg Value"] = pos_display["Avg Value"].apply(fmt_eur)
    pos_display["Avg Residual %"] = pos_display["Avg Residual %"].apply(
        lambda x: "{:+.1f}%".format(x)
    )
    st.dataframe(
        pos_display[["Position", "Players", "Avg Value", "Avg Residual %"]],
        hide_index=True, use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────
# PAGE 4 — CROSS-LEAGUE
# ──────────────────────────────────────────────────────────────
def render_cross_league(season):
    st.header("Cross-League Analysis")

    all_leagues = list(LEAGUE_DISPLAY_NAMES.keys())
    available = [lk for lk in all_leagues
                 if (PROC_DIR / lk / season / "results.csv").exists()]

    if len(available) < 2:
        st.warning("Need at least 2 leagues with results.csv for cross-league analysis.")
        return

    # ── League Premium Regression ─────────────────────────────
    st.subheader("League Premium (Pooled Regression)")

    if st.button("▶ Run Cross-League Regression"):
        with st.spinner("Running pooled OLS…"):
            dfs = []
            for lk in available:
                d = load_results(lk, season)
                if d.empty:
                    continue
                d["league_key"] = lk
                dfs.append(d)

            pool = pd.concat(dfs, ignore_index=True)
            pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_market_value"])

            feature_cols = (
                ["minutes_played", "is_top4", "is_top6", "is_bottom6",
                 "is_historic_top", "is_promoted", "yellow_cards",
                 "pass_success_pct", "fouls_per_game",
                 "dribbled_past_per_game", "xg_diff",
                 "is_brazilian", "is_french", "is_english", "is_spanish",
                 "is_german", "is_argentine", "is_portuguese", "is_african",
                 "is_asian", "is_south_american_other"]
            )
            baseline = available[0]  # PL = baseline
            for lk in available[1:]:
                col_name = "is_{}".format(lk)
                pool[col_name] = (pool["league_key"] == lk).astype(int)
                feature_cols.append(col_name)

            avail_cols = [c for c in feature_cols if c in pool.columns]
            X = pool[avail_cols].fillna(0)
            stds = X.std()
            X = X[[c for c in X.columns if stds[c] > 0]]
            y = pool["log_market_value"]

            try:
                Xc = sm.add_constant(X, has_constant="add")
                res = sm.OLS(y, Xc).fit()

                st.success("R² = {:.4f}  Adj R² = {:.4f}  N = {}".format(
                    res.rsquared, res.rsquared_adj, int(res.nobs)
                ))

                league_rows = []
                for lk in available[1:]:
                    col_name = "is_{}".format(lk)
                    if col_name in res.params:
                        coef_v = float(res.params[col_name])
                        pv = float(res.pvalues[col_name])
                        league_rows.append({
                            "League": LEAGUE_DISPLAY_NAMES[lk],
                            "Coefficient": round(coef_v, 4),
                            "% Premium vs PL": "{:+.1f}%".format(pct_premium(coef_v)),
                            "p-value": round(pv, 4),
                            "Sig": sig_stars(pv),
                        })

                if league_rows:
                    st.dataframe(
                        pd.DataFrame(league_rows),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption(
                        "Interpretation: Controlling for all performance and club factors, "
                        "a positive coefficient means players in that league are valued more "
                        "than equivalent Premier League players."
                    )
            except Exception as e:
                st.error("Cross-league regression failed: " + str(e))

    st.markdown("---")

    # ── Coefficient Comparison Table ──────────────────────────
    st.subheader("Coefficient Comparison Across Leagues")

    all_coeff = {}
    for lk in available:
        c = load_coefficients(lk, season)
        all_coeff[lk] = c

    rows = []
    for var in KEY_VARIABLES:
        row = {"Variable": var}
        signs = []
        for lk in available:
            c = all_coeff.get(lk, {})
            coef_v = c.get("coefficients", {}).get(var, None)
            pv = c.get("pvalues", {}).get(var, 1.0)
            if coef_v is not None:
                row[LEAGUE_DISPLAY_NAMES[lk]] = round(float(coef_v), 4)
                if float(pv) < 0.05:
                    signs.append(1 if float(coef_v) > 0 else -1)
            else:
                row[LEAGUE_DISPLAY_NAMES[lk]] = None

        if signs:
            consistent = "✓" if len(set(signs)) == 1 else "✗"
        else:
            consistent = "—"
        row["Consistent"] = consistent
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Bubble chart (PL vs Bundesliga) ───────────────────────
    if "premier_league" in available and "bundesliga" in available:
        st.subheader("PL vs Bundesliga Coefficient Comparison")

        pl_c = all_coeff.get("premier_league", {})
        bl_c = all_coeff.get("bundesliga", {})

        bubble_rows = []
        for var in KEY_VARIABLES + list(STAT_TO_COEF.keys()):
            pl_coef = pl_c.get("coefficients", {}).get(var)
            bl_coef = bl_c.get("coefficients", {}).get(var)
            if pl_coef is None or bl_coef is None:
                continue
            pl_t = abs(float(pl_coef) / max(pl_c.get("std_errors", {}).get(var, 1e-9), 1e-9))
            bl_t = abs(float(bl_coef) / max(bl_c.get("std_errors", {}).get(var, 1e-9), 1e-9))
            mean_t = (pl_t + bl_t) / 2
            pl_p = pl_c.get("pvalues", {}).get(var, 1.0)
            bl_p = bl_c.get("pvalues", {}).get(var, 1.0)
            consistent = "Yes" if (float(pl_coef) * float(bl_coef) > 0) else "No"
            bubble_rows.append({
                "variable": var,
                "PL coef": float(pl_coef),
                "BL coef": float(bl_coef),
                "mean_t": mean_t,
                "consistent": consistent,
            })

        if bubble_rows:
            bub_df = pd.DataFrame(bubble_rows)
            mn_c = min(bub_df["PL coef"].min(), bub_df["BL coef"].min())
            mx_c = max(bub_df["PL coef"].max(), bub_df["BL coef"].max())

            fig_bub = px.scatter(
                bub_df,
                x="PL coef",
                y="BL coef",
                size="mean_t",
                color="consistent",
                text="variable",
                color_discrete_map={"Yes": "#10b981", "No": "#ef4444"},
                title="PL vs Bundesliga Coefficients",
                labels={"PL coef": "Premier League Coefficient",
                        "BL coef": "Bundesliga Coefficient"},
                size_max=30,
            )
            fig_bub.add_shape(
                type="line",
                x0=mn_c, y0=mn_c, x1=mx_c, y1=mx_c,
                line=dict(color="gray", dash="dash"),
            )
            fig_bub.update_traces(textposition="top center")
            fig_bub.update_layout(height=500)
            st.plotly_chart(fig_bub, use_container_width=True)

    st.markdown("---")

    # ── Key Insights ──────────────────────────────────────────
    st.subheader("Key Insights")

    insight_lines = []

    hist_prems = {}
    for lk in available:
        c = all_coeff.get(lk, {})
        v = c.get("coefficients", {}).get("is_historic_top")
        if v is not None:
            hist_prems[LEAGUE_DISPLAY_NAMES[lk]] = pct_premium(v)

    if hist_prems:
        min_lk = min(hist_prems, key=hist_prems.get)
        max_lk = max(hist_prems, key=hist_prems.get)
        insight_lines.append(
            "**Historic club prestige premium** is consistent across all leagues, ranging from "
            "{:+.1f}% ({}) to {:+.1f}% ({}).".format(
                hist_prems[min_lk], min_lk,
                hist_prems[max_lk], max_lk,
            )
        )

    prom_prems = {}
    for lk in available:
        c = all_coeff.get(lk, {})
        v = c.get("coefficients", {}).get("is_promoted")
        if v is not None:
            prom_prems[LEAGUE_DISPLAY_NAMES[lk]] = pct_premium(v)

    if prom_prems:
        avg_prom = sum(prom_prems.values()) / len(prom_prems)
        insight_lines.append(
            "**Promoted club discount**: On average, players at promoted clubs are valued "
            "{:+.1f}% vs their performance-equivalent peers.".format(avg_prom)
        )

    bottom6_vals = {}
    for lk in available:
        c = all_coeff.get(lk, {})
        v = c.get("coefficients", {}).get("is_bottom6")
        if v is not None:
            bottom6_vals[LEAGUE_DISPLAY_NAMES[lk]] = pct_premium(v)

    if bottom6_vals:
        pos_leagues = [lk for lk, p in bottom6_vals.items() if p > 0]
        neg_leagues = [lk for lk, p in bottom6_vals.items() if p < 0]
        if pos_leagues and neg_leagues:
            insight_lines.append(
                "**Bottom-6 effect** diverges: positive in {} but negative in {}.".format(
                    ", ".join(pos_leagues), ", ".join(neg_leagues)
                )
            )

    for line in insight_lines:
        st.markdown("• " + line)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # ── Sidebar part 1: league selection (no data I/O here) ──
    with st.sidebar:
        st.title("The Scout Society")
        st.caption("Transfer Market Intelligence")
        st.markdown("---")

        selected_leagues = st.multiselect(
            "Leagues",
            options=list(LEAGUE_DISPLAY_NAMES.keys()),
            default=["premier_league"],
            format_func=lambda x: LEAGUE_DISPLAY_NAMES[x],
        )
        if not selected_leagues:
            selected_leagues = ["premier_league"]

    season = "2024-25"

    # ── Load data outside any sidebar context ─────────────────
    raw_df = load_all_results(tuple(selected_leagues), season)

    # ── Sidebar part 2: filter widgets (depend on raw_df) ────
    with st.sidebar:
        all_positions = sorted(raw_df["position_tm"].dropna().unique().tolist()) if not raw_df.empty else []
        all_clubs = sorted(raw_df["club"].dropna().unique().tolist()) if not raw_df.empty else []
        all_nats = sorted(raw_df["nationality_group"].dropna().unique().tolist()) if not raw_df.empty else []

        position_filter = st.multiselect("Position", options=all_positions, default=[])
        club_filter = st.multiselect("Club", options=all_clubs, default=[])
        nat_filter = st.multiselect("Nationality Group", options=all_nats, default=[])
        valuation_filter = st.selectbox(
            "Valuation", ["All", "Overvalued", "Fair Value", "Undervalued"]
        )
        min_minutes = st.slider("Min Minutes Played", 500, 3500, 500, step=50)

    # ── Main area — tabs FIRST, nothing before this ───────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚽ Player Lookup",
        "📊 Model Explorer",
        "🌍 Nationality & Position",
        "🔀 Cross-League",
    ])

    with tab1:
        render_player_lookup(
            raw_df, selected_leagues, season,
            position_filter, club_filter, nat_filter,
            valuation_filter, min_minutes,
        )

    with tab2:
        render_model_explorer(selected_leagues, season)

    with tab3:
        filtered_for_page3 = apply_filters(
            raw_df.copy(), position_filter, club_filter,
            nat_filter, valuation_filter, min_minutes,
        )
        render_nationality_position(filtered_for_page3, selected_leagues, season)

    with tab4:
        render_cross_league(season)


main()
