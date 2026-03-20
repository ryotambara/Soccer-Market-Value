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
import statsmodels.api as sm
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS_PATH = os.path.join(_BASE, "data", "processed", "results.csv")
FEATURES_PATH = os.path.join(_BASE, "data", "processed", "features.csv")
COEF_PATH = os.path.join(_BASE, "data", "processed", "model_coefficients.json")

# ── Colour constants ──────────────────────────────────────────────────────────
AMBER = "#f0a500"
RED   = "#e05c5c"
BLUE  = "#5c9be0"
GREEN = "#4caf7d"

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_PATH, encoding="utf-8")
    if "percentile" not in df.columns:
        df["percentile"] = (
            df["residual"].rank(pct=True, ascending=False) * 100
        ).round(0).astype(int)
    if "valuation_label" not in df.columns:
        def _label(r):
            if r > 0.15:   return "overvalued"
            if r < -0.15:  return "undervalued"
            return "fairly valued"
        df["valuation_label"] = df["residual"].apply(_label)
    return df


@st.cache_data
def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH, encoding="utf-8")


@st.cache_data
def load_coefficients() -> dict:
    if not os.path.exists(COEF_PATH):
        return {}
    with open(COEF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_eur(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    m = v / 1_000_000
    if m >= 10:
        return f"€{m:.0f}M"
    if m >= 1:
        return f"€{m:.1f}M"
    return f"€{v/1000:.0f}k"


# ── Nationality / position group helpers ─────────────────────────────────────

_NAT_DUMMIES = [
    "is_brazilian", "is_french", "is_english", "is_spanish", "is_german",
    "is_argentinian", "is_portuguese", "is_african", "is_asian",
    "is_south_american_other",
]
_NAT_LABELS = {c: c.replace("is_", "").replace("_", " ").title() for c in _NAT_DUMMIES}
_NAT_LABELS["baseline"] = "Other Europe"

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


def infer_nat_group(row: pd.Series) -> str:
    for col in _NAT_DUMMIES:
        if col in row.index and row[col] == 1:
            return _NAT_LABELS[col]
    return "Other Europe"


def infer_pos_group(row: pd.Series) -> str:
    for col in _POS_DUMMIES:
        if col in row.index and row[col] == 1:
            return _POS_LABELS[col]
    return "Centre-Back"


# ── Page 1: Player Lookup ─────────────────────────────────────────────────────

def page_player_lookup(df: pd.DataFrame, coefs: dict) -> None:
    st.title("Player Lookup")

    # ── Sidebar filters ──
    with st.sidebar:
        st.subheader("Filters")
        search = st.text_input("Search player name", "")

        positions = ["All"] + sorted(df["position"].dropna().unique().tolist()) if "position" in df.columns else ["All"]
        pos_filter = st.multiselect("Position", options=positions[1:])

        all_clubs = sorted(df["club"].dropna().unique().tolist()) if "club" in df.columns else []
        selected_clubs = st.multiselect("Club", options=all_clubs, default=[], placeholder="All clubs")

        st.subheader("Prestige Adjustment")
        show_prestige = st.toggle(
            "Show Big 6 prestige premium",
            value=True,
            help=(
                "When ON: includes the historic Big 6 club premium in predicted values. "
                "When OFF: strips out the prestige coefficient to show what the player "
                "would be worth at a non-Big-6 club."
            ),
        )
        if not show_prestige:
            st.caption(
                "Predicted values adjusted to remove Big 6 prestige premium "
                "— shows performance-only value."
            )

        nats = sorted(df["nationality"].dropna().unique().tolist()) if "nationality" in df.columns else []
        nat_filter = st.multiselect("Nationality", options=nats)

        val_filter = st.selectbox("Valuation", ["All", "undervalued", "overvalued", "fairly valued"])

        res_min = float(df["residual"].min()) if "residual" in df.columns else -2.0
        res_max = float(df["residual"].max()) if "residual" in df.columns else 2.0
        res_range = st.slider("Residual range", res_min, res_max,
                              (res_min, res_max), step=0.01)

    # ── Apply filters ──
    filtered = df.copy()
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]
    if pos_filter:
        filtered = filtered[filtered["position"].isin(pos_filter)]
    if selected_clubs:
        filtered = filtered[filtered["club"].isin(selected_clubs)]
    if nat_filter:
        filtered = filtered[filtered["nationality"].isin(nat_filter)]
    if val_filter != "All":
        filtered = filtered[filtered["valuation_label"] == val_filter]
    filtered = filtered[
        (filtered["residual"] >= res_range[0]) &
        (filtered["residual"] <= res_range[1])
    ]

    # ── Prestige adjustment ──
    _BIG6 = {
        "Manchester City", "Man City", "Arsenal", "Arsenal FC",
        "Liverpool", "Liverpool FC", "Chelsea", "Chelsea FC",
        "Manchester United", "Man Utd", "Manchester United FC",
        "Tottenham Hotspur", "Tottenham",
    }
    if not show_prestige and coefs and "coefficients" in coefs:
        historic_coef = coefs["coefficients"].get("is_historic_top6", 0.0)
        if historic_coef != 0.0 and "predicted_log_value" in filtered.columns:
            filtered = filtered.copy()
            is_big6 = filtered["club"].isin(_BIG6)
            filtered.loc[is_big6, "predicted_log_value"] = (
                filtered.loc[is_big6, "predicted_log_value"] - historic_coef
            )
            filtered.loc[is_big6, "predicted_market_value_eur"] = np.exp(
                filtered.loc[is_big6, "predicted_log_value"]
            )

    # ── Leaderboard table ──
    st.subheader(f"Players ({len(filtered)})")

    display_cols = {
        "player_name": "Player", "club": "Club", "position": "Position",
        "age": "Age", "market_value_eur": "Actual Value",
        "predicted_market_value_eur": "Predicted Value",
        "residual": "Residual", "valuation_label": "Valuation",
        "percentile": "Percentile",
    }
    show_cols = [c for c in display_cols if c in filtered.columns]
    tbl = filtered[show_cols].rename(columns=display_cols).reset_index(drop=True)

    for col in ("Actual Value", "Predicted Value"):
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(fmt_eur)

    st.dataframe(tbl, width="stretch", height=400)

    # ── Player Detail ──
    st.divider()
    st.subheader("Player Detail")

    detail_search = st.text_input("Type a player name for full detail", key="detail_search")

    if detail_search:
        matches = df[df["player_name"].str.contains(detail_search, case=False, na=False)]
        if matches.empty:
            st.warning(f"No player found matching '{detail_search}'.")
        else:
            player = matches.iloc[0]
            pname = player.get("player_name", "")
            club = player.get("club", "")
            pos = player.get("position", "")
            age = player.get("age", "")
            nat = player.get("nationality", "")
            actual = player.get("market_value_eur")
            predicted = player.get("predicted_market_value_eur")
            residual = player.get("residual", 0.0)
            vlabel = player.get("valuation_label", "")
            pct = int(player.get("percentile", 0))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Player", pname)
                st.metric("Club", club)
                st.metric("Position", pos)
            with col2:
                st.metric("Age", age)
                st.metric("Nationality", nat)
                st.metric("Valuation", vlabel.upper() if vlabel else "")
            with col3:
                st.metric("Actual Value", fmt_eur(actual))
                st.metric("Predicted Value", fmt_eur(predicted),
                          delta=fmt_eur(actual - predicted) if actual and predicted else None)
                st.metric("Percentile", f"{pct}th")

            # Gauge
            gauge_val = max(0.0, min(1.0, (residual + 1.5) / 3.0))
            st.caption(f"Residual gauge (0 = max undervalued, 1 = max overvalued)")
            st.progress(gauge_val)

            # Stats grid
            st.divider()
            stat_cols_left = ["minutes_played", "goals", "assists",
                              "goals_per_90", "assists_per_90",
                              "contract_months_remaining", "team_league_position"]
            stat_cols_right = ["rating", "xg", "xg_diff",
                               "tackles_per_game", "key_passes_per_game",
                               "pass_success_pct", "shots_per_game"]

            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("Core stats")
                for c in stat_cols_left:
                    if c in player.index and pd.notna(player[c]):
                        st.write(f"**{c.replace('_', ' ').title()}**: {player[c]}")
            with sc2:
                st.caption("WhoScored stats")
                for c in stat_cols_right:
                    if c in player.index and pd.notna(player[c]):
                        st.write(f"**{c.replace('_', ' ').title()}**: {player[c]}")

            # ── What-if sliders ──
            if coefs and "coefficients" in coefs:
                st.divider()
                st.subheader("What-If Prediction")
                st.caption("Adjust values below to see how the model's predicted value changes.")

                wc1, wc2 = st.columns(2)
                with wc1:
                    wi_goals = st.slider("Goals per 90", 0.0, 1.5,
                                         float(player.get("goals_per_90", 0.0)), 0.05)
                    wi_assists = st.slider("Assists per 90", 0.0, 1.0,
                                           float(player.get("assists_per_90", 0.0)), 0.05)
                    wi_xg = st.slider("xG", 0.0, 50.0,
                                      float(player.get("xg", 0.0)), 0.5)
                with wc2:
                    wi_contract = st.slider("Contract months remaining", 0, 96,
                                            int(player.get("contract_months_remaining", 12)), 6)
                    wi_rating = st.slider("Rating", 5.0, 9.0,
                                          float(player.get("rating", 6.5)), 0.1)

                # Build feature vector for what-if
                coef_map = coefs["coefficients"]
                intercept = coefs.get("intercept", 0.0)

                # Start from original player's feature values
                log_pred = intercept
                for var, coef in coef_map.items():
                    if var in ("goals_per_90", "assists_per_90"):
                        continue  # replaced below
                    val = float(player.get(var, 0.0)) if var in player.index else 0.0
                    if pd.isna(val):
                        val = 0.0
                    log_pred += coef * val

                # Override slider variables
                pos_group = infer_pos_group(player)
                pos_prefix_map = {
                    "Striker": "striker", "Winger": "winger",
                    "Attacking Mid": "attmid", "Central Mid": "cm",
                    "CDM": "cdm", "Fullback": "fullback",
                    "Centre-Back": "cb", "Goalkeeper": "gk",
                }
                pp = pos_prefix_map.get(pos_group, "cb")

                for stat, wi_val in [("goals_per_90", wi_goals), ("assists_per_90", wi_assists)]:
                    for pfx in ["striker", "winger", "attmid", "cm", "cdm", "fullback", "cb", "gk"]:
                        col_name = f"{pfx}_{stat}"
                        if col_name in coef_map:
                            new_val = wi_val if pfx == pp else 0.0
                            log_pred += coef_map[col_name] * new_val

                if "rating" in coef_map:
                    log_pred += coef_map["rating"] * (wi_rating - float(player.get("rating", 6.5)))
                if "xg_diff" in coef_map:
                    orig_xg = float(player.get("xg", 0.0)) if pd.notna(player.get("xg")) else 0.0
                    log_pred += coef_map.get("xg_diff", 0.0) * (wi_xg - orig_xg)
                if "contract_months_remaining" in coef_map:
                    orig_cm = float(player.get("contract_months_remaining", 12))
                    log_pred += coef_map["contract_months_remaining"] * (wi_contract - orig_cm)

                wi_predicted = np.exp(log_pred)
                wi_delta = wi_predicted - (predicted if predicted else 0)

                st.metric("What-If Predicted Value", fmt_eur(wi_predicted),
                          delta=fmt_eur(wi_delta) if wi_delta != 0 else None)


# ── Page 2: Model Explorer ───────────────────────────────────────────────────

def page_model_explorer(features_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    st.title("Model Explorer")

    if features_df.empty:
        st.error(f"features.csv not found at {FEATURES_PATH}. Run the full pipeline first.")
        return

    num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"market_value_eur", "log_market_value", "predicted_log_value",
               "residual", "predicted_market_value_eur", "is_centre_back"}

    # ── Variable groups ──
    groups = {
        "Demographics": ["age", "age_squared", "contract_months_remaining"],
        "Performance — Global": [
            "minutes_played", "rating", "xg_diff", "pass_success_pct",
            "fouls_per_game", "dribbled_past_per_game", "yellow_cards",
        ],
        "Team & League": [
            "team_league_position", "is_top4", "is_top6", "is_bottom6",
            "is_historic_top6",
        ],
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
        "Tackles × Position": [c for c in num_cols if c.startswith("tackles_")],
        "Interceptions × Position": [c for c in num_cols if c.startswith("interceptions_")],
        "Clearances × Position": [c for c in num_cols if c.startswith("clearances_")],
        "Key Passes × Position": [c for c in num_cols if c.startswith("key_passes_")],
        "xG × Position": [c for c in num_cols if c.startswith("xg_") and "_pos" not in c
                          and any(c.endswith(p) for p in
                          ["_striker","_winger","_attmid","_cm","_cdm","_fullback","_cb","_gk"])],
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
        st.subheader("Variables")
        select_all = st.button("Select All")
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

    # ── Run regression ──
    st.subheader("Run Regression")
    run = st.button("▶ Run Regression", type="primary")

    if run and selected_vars:
        feat = features_df.copy()
        y_col = "log_market_value"
        if y_col not in feat.columns:
            st.error("log_market_value column missing from features.csv.")
            return

        y = feat[y_col].astype(float)
        X = feat[[v for v in selected_vars if v in feat.columns]].astype(float)

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        df_m = feat[mask].copy().reset_index(drop=True)

        X_const = sm.add_constant(X)
        res = sm.OLS(y, X_const).fit()

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{res.rsquared:.4f}")
        m2.metric("Adj R²", f"{res.rsquared_adj:.4f}")
        m3.metric("N", int(res.nobs))
        m4.metric("F-statistic", f"{res.fvalue:.1f}")

        # Coefficient table
        coef_df = pd.DataFrame({
            "Variable": res.params.index,
            "Coefficient": res.params.values,
            "Std Error": res.bse.values,
            "T-stat": res.tvalues.values,
            "P-value": res.pvalues.values,
        })
        coef_df["Sig"] = coef_df["P-value"].apply(
            lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        )
        coef_df = coef_df[coef_df["Variable"] != "const"]
        coef_df["abs_coef"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

        def _color_row(row):
            if row["P-value"] < 0.05:
                return ["background-color: #1a3a1a"] * len(row)
            elif row["P-value"] < 0.1:
                return ["background-color: #3a3210"] * len(row)
            return ["background-color: #1a1a1a"] * len(row)

        st.subheader("Coefficients")
        st.dataframe(
            coef_df.style.apply(_color_row, axis=1).format({
                "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
                "T-stat": "{:.3f}", "P-value": "{:.4f}",
            }),
            width="stretch",
            height=400,
        )

        # Charts
        predicted_log = res.fittedvalues
        residuals_run = (y - predicted_log).values
        predicted_eur = np.exp(predicted_log.values)
        actual_eur = np.exp(y.values)

        def _vlabel(r):
            if r > 0.15:   return "overvalued"
            if r < -0.15:  return "undervalued"
            return "fairly valued"

        chart_df = pd.DataFrame({
            "residual": residuals_run,
            "actual_eur": actual_eur,
            "predicted_eur": predicted_eur,
            "valuation": [_vlabel(r) for r in residuals_run],
        })

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_hist = px.histogram(
                chart_df, x="residual", nbins=40,
                title="Residual Distribution",
                color_discrete_sequence=[AMBER],
            )
            fig_hist.add_vline(x=0, line_color="white", line_dash="dash")
            fig_hist.update_layout(
                paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig_hist, width="stretch")

        with ch2:
            color_map = {"overvalued": AMBER, "undervalued": BLUE, "fairly valued": GREEN}
            fig_scat = px.scatter(
                chart_df, x="predicted_eur", y="actual_eur",
                color="valuation", color_discrete_map=color_map,
                title="Actual vs Predicted",
                labels={"predicted_eur": "Predicted (€)", "actual_eur": "Actual (€)"},
                opacity=0.7,
            )
            max_val = max(chart_df["actual_eur"].max(), chart_df["predicted_eur"].max())
            fig_scat.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="white", dash="dash"),
            )
            fig_scat.update_layout(
                paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig_scat, width="stretch")

        # Top under/overvalued
        if "player_name" in df_m.columns:
            chart_df["player_name"] = df_m["player_name"].values
            chart_df["club"] = df_m["club"].values if "club" in df_m.columns else ""
            chart_df = chart_df.sort_values("residual", ascending=False)

            tu1, tu2 = st.columns(2)
            with tu1:
                st.subheader("Top 10 Overvalued")
                ov = chart_df.head(10)[["player_name", "club", "residual"]].reset_index(drop=True)
                st.dataframe(ov, width="stretch")
            with tu2:
                st.subheader("Top 10 Undervalued")
                uv = chart_df.tail(10)[["player_name", "club", "residual"]].iloc[::-1].reset_index(drop=True)
                st.dataframe(uv, width="stretch")

    elif run and not selected_vars:
        st.warning("Select at least one variable to run the regression.")


# ── Page 3: Nationality & Position Analysis ───────────────────────────────────

def _dummy_to_pct(coef: float) -> float:
    """Convert a log-scale dummy coefficient to % market premium/discount."""
    return round((np.exp(coef) - 1.0) * 100, 1)


def _auto_insight(group_name: str, grp_df: pd.DataFrame, player_col: str,
                  model_pct: float) -> str:
    direction = "premium" if model_pct >= 0 else "discount"
    extreme_ov = grp_df.loc[grp_df["residual"].idxmax(), player_col]
    extreme_uv = grp_df.loc[grp_df["residual"].idxmin(), player_col]
    return (
        f"The model estimates a **{abs(model_pct):.0f}% market {direction}** for "
        f"{group_name} players (all else equal). "
        f"Most over-priced individual: **{extreme_ov}** | "
        f"Most under-priced: **{extreme_uv}**."
    )


def page_nat_pos(df: pd.DataFrame, coefs: dict) -> None:
    st.title("Nationality & Position Analysis")

    player_col = "player_name" if "player_name" in df.columns else "player"
    coef_map = coefs.get("coefficients", {}) if coefs else {}

    df = df.copy()
    df["_nat_group"] = df.apply(infer_nat_group, axis=1)
    df["_pos_group"] = df.apply(infer_pos_group, axis=1)

    # Map from group label → dummy column name
    _nat_dummy_for = {v: k for k, v in _NAT_LABELS.items() if k != "baseline"}
    _pos_dummy_for = {v: k for k, v in _POS_LABELS.items() if k != "baseline"}

    tab_nat, tab_pos = st.tabs(["🌍 By Nationality", "⚽ By Position"])

    # ── Nationality tab ──
    with tab_nat:
        st.caption(
            "**Market Premium %** = the model's estimated % by which the market "
            "over-pays (+) or under-pays (−) for a player from this nationality, "
            "holding all other factors constant. "
            "Baseline = Other Europe (0%)."
        )
        nat_agg = []
        for grp_name, grp in df.groupby("_nat_group"):
            if grp.empty:
                continue
            dummy = _nat_dummy_for.get(grp_name)
            raw_coef = coef_map.get(dummy, 0.0) if dummy else 0.0
            mkt_pct = _dummy_to_pct(raw_coef)
            nat_agg.append({
                "Nationality": grp_name,
                "Players": len(grp),
                "Market Premium %": mkt_pct,
                "Model Coef (log)": round(raw_coef, 4),
                "Avg Actual €M": round(grp["market_value_eur"].mean() / 1e6, 2),
                "Avg Predicted €M": round(grp["predicted_market_value_eur"].mean() / 1e6, 2),
                "Most Overvalued": grp.loc[grp["residual"].idxmax(), player_col],
                "Most Undervalued": grp.loc[grp["residual"].idxmin(), player_col],
                "_pct": mkt_pct,
            })
        nat_df = pd.DataFrame(nat_agg).sort_values("Market Premium %", ascending=False)

        fig_nat = px.bar(
            nat_df, x="Nationality", y="Market Premium %",
            title="Model-Estimated Market Premium/Discount by Nationality",
            color="Market Premium %",
            color_continuous_scale=[[0, BLUE], [0.5, "#888"], [1, AMBER]],
            labels={"Market Premium %": "Market Premium (%)"},
        )
        fig_nat.update_layout(
            paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
            xaxis_tickangle=-30,
        )
        fig_nat.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
        st.plotly_chart(fig_nat, width="stretch")

        show_nat = nat_df.drop(columns="_pct").reset_index(drop=True)
        st.dataframe(show_nat, width="stretch")

        # Insight boxes
        for _, row in nat_df.iterrows():
            grp = df[df["_nat_group"] == row["Nationality"]]
            if len(grp) >= 3:
                st.info(_auto_insight(row["Nationality"], grp, player_col, row["_pct"]))

    # ── Position tab ──
    with tab_pos:
        st.caption(
            "**Market Premium %** = the model's estimated % by which the market "
            "over-pays (+) or under-pays (−) for a player in this position, "
            "holding all other factors constant. "
            "Baseline = Centre-Back (0%)."
        )

        # Big 6 prestige note
        _historic_coef = coef_map.get("is_historic_top6")
        _historic_pval = (coefs.get("pvalues", {}) or {}).get("is_historic_top6") if coefs else None
        if _historic_coef is not None:
            _prem_pct = _dummy_to_pct(_historic_coef)
            _pval_str = f"p={_historic_pval:.3f}" if _historic_pval is not None else "p=n/a"
            st.info(
                f"**Big 6 prestige premium: {_prem_pct:+.1f}%** ({_pval_str})  "
                f"— the model estimates players at historic Big 6 clubs "
                f"(Man City, Arsenal, Liverpool, Chelsea, Man Utd, Spurs) "
                f"command this premium over equivalent players elsewhere, "
                f"independent of current league position."
            )
        pos_agg = []
        for grp_name, grp in df.groupby("_pos_group"):
            if grp.empty:
                continue
            dummy = _pos_dummy_for.get(grp_name)
            raw_coef = coef_map.get(dummy, 0.0) if dummy else 0.0
            mkt_pct = _dummy_to_pct(raw_coef)
            pos_agg.append({
                "Position": grp_name,
                "Players": len(grp),
                "Market Premium %": mkt_pct,
                "Model Coef (log)": round(raw_coef, 4),
                "Avg Actual €M": round(grp["market_value_eur"].mean() / 1e6, 2),
                "Avg Predicted €M": round(grp["predicted_market_value_eur"].mean() / 1e6, 2),
                "Most Overvalued": grp.loc[grp["residual"].idxmax(), player_col],
                "Most Undervalued": grp.loc[grp["residual"].idxmin(), player_col],
                "_pct": mkt_pct,
            })
        pos_df = pd.DataFrame(pos_agg).sort_values("Market Premium %", ascending=False)

        fig_pos = px.bar(
            pos_df, x="Position", y="Market Premium %",
            title="Model-Estimated Market Premium/Discount by Position",
            color="Market Premium %",
            color_continuous_scale=[[0, BLUE], [0.5, "#888"], [1, AMBER]],
        )
        fig_pos.update_layout(
            paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
        )
        fig_pos.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
        st.plotly_chart(fig_pos, width="stretch")

        show_pos = pos_df.drop(columns="_pct").reset_index(drop=True)
        st.dataframe(show_pos, width="stretch")

        # Scatter: xg vs residual
        if "xg" in df.columns:
            fig_xg = px.scatter(
                df, x="xg", y="residual",
                color="_pos_group",
                title="xG vs Individual Residual by Position",
                hover_data=[player_col],
                opacity=0.7,
            )
            fig_xg.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
            fig_xg.update_layout(
                paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig_xg, width="stretch")

        # Scatter: rating vs residual
        if "rating" in df.columns:
            fig_rat = px.scatter(
                df, x="rating", y="residual",
                color="_pos_group",
                title="Rating vs Individual Residual by Position",
                hover_data=[player_col],
                opacity=0.7,
            )
            fig_rat.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
            fig_rat.update_layout(
                paper_bgcolor="#0d1420", plot_bgcolor="#0d1420",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig_rat, width="stretch")

        for _, row in pos_df.iterrows():
            grp = df[df["_pos_group"] == row["Position"]]
            if len(grp) >= 3:
                st.info(_auto_insight(row["Position"], grp, player_col, row["_pct"]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="PitchIQ — Transfer Market Intelligence",
        page_icon="⚽",
        layout="wide",
    )

    # Load data
    results_df = load_results()
    features_df = load_features()
    coefs = load_coefficients()

    if results_df.empty:
        st.error(
            f"results.csv not found at `{RESULTS_PATH}`. "
            "Run the full pipeline (merge → clean → features → regression) first."
        )
        return

    # Sidebar navigation
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
        st.caption(f"{len(results_df)} players loaded")

    if page == "⚽ Player Lookup":
        page_player_lookup(results_df, coefs)
    elif page == "📊 Model Explorer":
        page_model_explorer(features_df, results_df)
    else:
        page_nat_pos(results_df, coefs)


if __name__ == "__main__":
    main()
