"""
scraper/transfermarkt_positions.py

Lightweight scraper: fetches player positions from Transfermarkt squad pages.
Outputs player_name, club, position_tm, player_id — no market value, no contracts.

Change LEAGUE at the top to run a different league.

Run: python scraper/transfermarkt_positions.py
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

# ── Change this to run a different league ─────────────────────────────────────
LEAGUE = "bundesliga"
# Options: "premier_league" | "bundesliga" | "la_liga" | "serie_a" | "liga_portugal"
# ──────────────────────────────────────────────────────────────────────────────

LEAGUES = {
    "premier_league": {
        "slug":   "premier-league/startseite/wettbewerb/GB1",
        "season": "2025",
    },
    "bundesliga": {
        "slug":   "bundesliga/startseite/wettbewerb/L1",
        "season": "2025",
    },
    "la_liga": {
        "slug":   "primera-division/startseite/wettbewerb/ES1",
        "season": "2025",
    },
    "serie_a": {
        "slug":   "serie-a/startseite/wettbewerb/IT1",
        "season": "2025",
    },
    "liga_portugal": {
        "slug":   "liga-portugal/startseite/wettbewerb/PO1",
        "season": "2025",
    },
}

BASE_URL = "https://www.transfermarkt.com"

# Reused verbatim from scraper/transfermarkt.py
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
    "DNT": "1",
}

# Exact position strings as Transfermarkt writes them
KNOWN_POSITIONS = {
    "Goalkeeper",
    "Centre-Back",
    "Left-Back",
    "Right-Back",
    "Left Wing-Back",
    "Right Wing-Back",
    "Defensive Midfield",
    "Central Midfield",
    "Attacking Midfield",
    "Left Midfield",
    "Right Midfield",
    "Left Winger",
    "Right Winger",
    "Second Striker",
    "Centre-Forward",
}

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _output_path(league_key):
    return os.path.join(
        _BASE_DIR, "data", "raw", league_key, "2025-26", "transfermarkt_positions.csv"
    )


def get_club_urls(session, league_url):
    """Fetch the league page and return list of {name, url} dicts."""
    print(f"Fetching league page: {league_url}")
    resp = session.get(league_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    clubs = []
    seen = set()

    table = soup.find("table", {"class": "items"})
    if table:
        for row in table.find_all("tr", class_=["odd", "even"]):
            name_cell = row.find("td", class_="hauptlink")
            link = None
            if name_cell:
                link = name_cell.find("a", href=re.compile(r"/startseite/verein/\d+"))
            if not link:
                link = row.find("a", href=re.compile(r"/startseite/verein/\d+"))
            if link:
                url = BASE_URL + link["href"]
                name = link.get_text(strip=True) or link.get("title", "").strip()
                if url not in seen and name:
                    clubs.append({"name": name, "url": url})
                    seen.add(url)
    else:
        for a in soup.select("td.hauptlink a[href*='/startseite/verein/']"):
            url = BASE_URL + a["href"]
            name = a.get_text(strip=True) or a.get("title", "").strip()
            if url not in seen and name:
                clubs.append({"name": name, "url": url})
                seen.add(url)

    print(f"  Found {len(clubs)} clubs.")
    return clubs


def scrape_club_players(session, club_name, club_url):
    """Fetch a squad page and return list of player dicts."""
    try:
        resp = session.get(club_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.Timeout:
        print(f"  WARNING: Timeout fetching {club_name} — skipping.")
        return []
    except requests.HTTPError as e:
        print(f"  WARNING: HTTP {e.response.status_code} for {club_name} — skipping.")
        return []
    except requests.RequestException as e:
        print(f"  WARNING: Request error for {club_name}: {e} — skipping.")
        return []

    soup = BeautifulSoup(resp.content, "html.parser")

    # Resolve club name from page title if needed
    resolved_name = club_name.strip()
    if not resolved_name:
        title = soup.find("title")
        if title:
            resolved_name = title.get_text(strip=True).split(" - ")[0].strip()

    squad_table = soup.find("table", {"class": "items"})
    if not squad_table:
        print(f"  WARNING: No squad table for {resolved_name} — skipping.")
        return []

    rows = squad_table.find_all("tr", class_=["odd", "even"])
    players = []

    for row in rows:
        try:
            name_cell = row.find("td", class_="hauptlink")
            if not name_cell:
                continue
            name_link = name_cell.find("a")
            if not name_link:
                continue
            player_name = name_link.get_text(strip=True)

            player_href = name_link.get("href", "")
            pid_match = re.search(r"profil/spieler/(\d+)", player_href)
            if not pid_match:
                pid_match = re.search(r"/spieler/(\d+)$", player_href)
            player_id = int(pid_match.group(1)) if pid_match else None

            # Position: find the cell whose text exactly matches a known TM position
            position_tm = ""
            for cell in row.find_all("td"):
                text = cell.get_text(strip=True)
                if text in KNOWN_POSITIONS:
                    position_tm = text
                    break

            players.append({
                "player_name": player_name,
                "club":        resolved_name,
                "position_tm": position_tm,
                "player_id":   player_id,
            })
        except Exception as e:
            print(f"  ERROR processing row: {e}")
            continue

    return players


def main():
    cfg = LEAGUES[LEAGUE]
    league_url = f"{BASE_URL}/{cfg['slug']}/saison_id/{cfg['season']}"
    out_path = _output_path(LEAGUE)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 60)
    print(f"Transfermarkt Positions Scraper — {LEAGUE}  [2025-26]")
    print(f"League URL: {league_url}")
    print(f"Output:     {out_path}")
    print("=" * 60)

    session = requests.Session()
    clubs = get_club_urls(session, league_url)
    if not clubs:
        print("ERROR: No clubs found. Check the league URL.")
        return

    all_players = []
    for i, club in enumerate(clubs, 1):
        print(f"\n[{i}/{len(clubs)}] {club['name']}")
        players = scrape_club_players(session, club["name"], club["url"])
        print(f"  {len(players)} players found")
        all_players.extend(players)
        time.sleep(2)

    df = pd.DataFrame(all_players)
    df = df.drop_duplicates(subset=["player_name", "club"])
    df = df[["player_name", "club", "position_tm", "player_id"]]

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Done. Saved {len(df)} players to {out_path}")
    if not df.empty:
        print("\nPosition distribution:")
        print(df["position_tm"].value_counts().to_string())


if __name__ == "__main__":
    main()
