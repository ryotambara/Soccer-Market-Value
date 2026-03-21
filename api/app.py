"""
api/app.py

Flask API for player valuation lookup.
Endpoint: GET /player?name=Salah

Returns JSON with actual vs predicted market value, residual,
valuation label, and percentile rank.

Run: python api/app.py
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz

app = Flask(__name__)
CORS(app)

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(_BASE_DIR, "data", "processed", "premier_league", "2025-26", "results.csv")

# Global dataframe loaded at startup
players_df: pd.DataFrame = pd.DataFrame()


def load_data():
    """Load results CSV into memory. Called once at startup."""
    global players_df
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results file not found at {RESULTS_PATH}")
        print("Run the full pipeline first: regression.py must complete successfully.")
        return

    players_df = pd.read_csv(RESULTS_PATH, encoding="utf-8")
    print(f"Loaded {len(players_df)} players from {RESULTS_PATH}")

    # Compute percentile rank by residual (lower residual = more undervalued = higher percentile)
    players_df["percentile"] = (
        players_df["residual"].rank(pct=True, ascending=False) * 100
    ).round(0).astype(int)

    print("Player data ready. API is live.")


def get_valuation_label(residual: float) -> str:
    """Classify player as undervalued / overvalued / fairly valued.
    residual = log_market_value - predicted_log_value
    Positive → actual > predicted → overvalued
    Negative → actual < predicted → undervalued
    """
    if residual > 0.15:
        return "overvalued"
    elif residual < -0.15:
        return "undervalued"
    else:
        return "fairly valued"


def find_player(name_query: str) -> dict | None:
    """
    Find a player using fuzzy name matching.
    Returns the best-matching player row as a dict, or None.
    """
    if players_df.empty:
        return None

    query = name_query.strip().lower()
    player_names = players_df["player_name"].tolist()

    result = process.extractOne(
        query,
        [n.lower() for n in player_names],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=60,
    )

    if result is None:
        return None

    _, score, idx = result
    row = players_df.iloc[idx]
    return row.to_dict(), score


@app.route("/player", methods=["GET"])
def player_lookup():
    """
    GET /player?name=Salah

    Returns:
    {
      "player": "Mohamed Salah",
      "club": "Liverpool",
      "actual_value_eur": 50000000,
      "predicted_value_eur": 61000000,
      "residual": 0.20,
      "valuation_label": "undervalued",
      "percentile": 87
    }
    """
    name_param = request.args.get("name", "").strip()

    if not name_param:
        return jsonify({"error": "Missing 'name' query parameter. Usage: /player?name=Salah"}), 400

    if players_df.empty:
        return jsonify({"error": "Player data not loaded. Ensure the pipeline has been run."}), 503

    match = find_player(name_param)

    if match is None:
        return jsonify({
            "error": f"No player found matching '{name_param}'.",
            "hint": "Try a more complete name or check the spelling."
        }), 404

    row, match_score = match

    actual_value = row.get("market_value_eur", None)
    predicted_value = row.get("predicted_market_value_eur", None)
    residual = row.get("residual", None)
    percentile = int(row.get("percentile", 0))

    response = {
        "player": str(row.get("player_name", "")),
        "club": str(row.get("club", "")),
        "position": str(row.get("position", "")),
        "age": int(row["age"]) if pd.notna(row.get("age")) else None,
        "actual_value_eur": int(actual_value) if pd.notna(actual_value) else None,
        "predicted_value_eur": int(predicted_value) if pd.notna(predicted_value) else None,
        "residual": round(float(residual), 4) if pd.notna(residual) else None,
        "valuation_label": get_valuation_label(residual) if pd.notna(residual) else "unknown",
        "percentile": percentile,
        "match_confidence": match_score,
    }

    return jsonify(response)


@app.route("/players", methods=["GET"])
def list_players():
    """
    GET /players?limit=50&sort=residual

    Returns a list of all players with their valuations.
    Optional query params:
      - limit: max number to return (default 100)
      - sort: 'residual' (default), 'name', 'value'
      - order: 'desc' (default) or 'asc'
      - position: filter by position group
    """
    if players_df.empty:
        return jsonify({"error": "Player data not loaded."}), 503

    df = players_df.copy()

    # Filters
    position_filter = request.args.get("position", "").strip().lower()
    if position_filter:
        df = df[df["position"].str.lower().str.contains(position_filter, na=False)]

    # Sort
    sort_by = request.args.get("sort", "residual")
    order = request.args.get("order", "desc")
    ascending = order == "asc"

    sort_col_map = {
        "residual": "residual",
        "name": "player_name",
        "value": "market_value_eur",
        "predicted": "predicted_market_value_eur",
        "percentile": "percentile",
    }
    sort_col = sort_col_map.get(sort_by, "residual")
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)

    # Limit
    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100
    df = df.head(limit)

    output_cols = ["player_name", "club", "position", "age",
                   "market_value_eur", "predicted_market_value_eur",
                   "residual", "valuation_label" if "valuation_label" in df.columns else "residual",
                   "percentile"]
    available = [c for c in output_cols if c in df.columns]

    # Add valuation label on the fly if not in df
    records = []
    for _, row in df[available].iterrows():
        r = row.to_dict()
        if "valuation_label" not in r and "residual" in r:
            res = r["residual"]
            r["valuation_label"] = get_valuation_label(res) if pd.notna(res) else "unknown"
        # Convert numpy types to native Python for JSON serialisation
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, float) and np.isnan(v):
                r[k] = None
        records.append(r)

    return jsonify({
        "count": len(records),
        "players": records,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "players_loaded": len(players_df),
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Football Transfer Value API",
        "endpoints": {
            "GET /player?name=<name>": "Lookup a specific player",
            "GET /players?limit=100&sort=residual": "List all players",
            "GET /health": "Service health check",
        }
    })


if __name__ == "__main__":
    load_data()
    app.run(host="0.0.0.0", port=5001, debug=False)
