"""
scraper/transfermarkt.py

Scrapes Premier League player data from Transfermarkt.

Two modes — set MODE at the top of this file:

  MODE = "2025-26"  current season, scrapes live market values
  MODE = "2024-25"  last season, fetches historical market value closest
                    to 2025-06-01 from each player's profile page

Output files:
  data/raw/transfermarkt_2025-26.csv
  data/raw/transfermarkt_2024-25.csv

Run: python scraper/transfermarkt.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from typing import Optional, List, Tuple
from datetime import datetime, date
from dateutil import parser as dateutil_parser

# ── Change this to switch seasons ─────────────────────────────────────────
MODE = "2024-25"   # "2025-26" | "2024-25"
# ──────────────────────────────────────────────────────────────────────────

BASE_URL = "https://www.transfermarkt.com"
_contract_debug_count = 0  # prints raw date string for first 3 players

# Target date for 2024-25 historical market value lookup
_TARGET_DATE_2425 = date(2025, 6, 1)
_MV_WINDOW_DAYS   = 366   # accept values within 12 months of target

# Transfermarkt encodes the season year as the start year (2024 = 2024-25)
_SAISON = "2025" if MODE == "2025-26" else "2024"

LEAGUE_URL = (
    f"https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
    f"/saison_id/{_SAISON}"
)

_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(_RAW_DIR, f"transfermarkt_{MODE}.csv")

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


def parse_market_value(value_str: str) -> Optional[float]:
    """Convert '€50m', '€500k', etc. to a float in euros."""
    if not value_str or value_str.strip() in ("-", "", "?"):
        return None
    value_str = value_str.replace("€", "").replace(",", "").strip()
    try:
        if "m" in value_str.lower():
            return float(value_str.lower().replace("m", "")) * 1_000_000
        elif "k" in value_str.lower():
            return float(value_str.lower().replace("k", "")) * 1_000
        else:
            return float(value_str)
    except (ValueError, AttributeError):
        return None


def contract_months_remaining(expiry_str: str) -> Optional[int]:
    """Convert a contract expiry date string to integer months remaining from today."""
    if not expiry_str or expiry_str.strip() in ("-", "", "?"):
        return None
    formats = ["%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%b %Y", "%d.%m.%Y"]
    for fmt in formats:
        try:
            expiry_date = datetime.strptime(expiry_str.strip(), fmt).date()
            today = date.today()
            months = (expiry_date.year - today.year) * 12 + (expiry_date.month - today.month)
            return max(months, 0)
        except ValueError:
            continue
    return None


def _extract_contract_months(soup: BeautifulSoup, player_name: str) -> Optional[int]:
    """
    Parse contract months remaining from a profile page soup object.
    (Logic extracted from the original get_contract_expiry.)
    """
    page_text = soup.get_text(" ", strip=True)

    date_patterns = [
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{2}\.\d{2}\.\d{4}\b',
    ]

    today = date.today()
    future_dates = []  # list of (date_obj, raw_string)

    for pattern in date_patterns:
        for raw in re.findall(pattern, page_text) if "(" not in pattern else [
            m.group() for m in re.finditer(pattern, page_text)
        ]:
            if isinstance(raw, tuple):
                raw = " ".join(raw).strip()
            raw = raw.strip()
            try:
                parsed = dateutil_parser.parse(raw, dayfirst=False).date()
            except (ValueError, OverflowError):
                continue
            if parsed > today:
                future_dates.append((parsed, raw))

    for m in re.finditer(date_patterns[0], page_text):
        raw = m.group().strip()
        try:
            parsed = dateutil_parser.parse(raw, dayfirst=False).date()
        except (ValueError, OverflowError):
            continue
        if parsed > today and (parsed, raw) not in future_dates:
            future_dates.append((parsed, raw))

    if not future_dates:
        return None

    june_30 = [item for item in future_dates if item[0].month == 6 and item[0].day == 30]
    chosen_date, chosen_raw = (
        min(june_30, key=lambda x: x[0]) if june_30
        else min(future_dates, key=lambda x: x[0])
    )

    months = (
        (chosen_date.year - today.year) * 12
        + (chosen_date.month - today.month)
    )
    result = max(months, 0)

    global _contract_debug_count
    if _contract_debug_count < 3:
        print(f"    [CONTRACT DEBUG] {player_name}: raw='{chosen_raw}' → {result} months")
        _contract_debug_count += 1

    return result


def _extract_historical_mv(soup: BeautifulSoup, player_name: str) -> Optional[float]:
    """
    Extract the market value closest to 2025-06-01 from a profile page soup.

    Approach 1 — Highcharts script tag: looks for [[timestamp_ms, value], ...]
                 arrays embedded as JavaScript in <script> tags.
    Approach 2 — marktwert table: parses any table with class containing
                 "marktwert" for date + value rows.

    Returns value in EUR as float, or None if nothing found within 12 months
    of the target date (2025-06-01).
    """
    target = _TARGET_DATE_2425
    best_value = None
    best_delta = float("inf")

    # ── Approach 1: Highcharts [[timestamp, value], ...] in script tags ──────
    for script in soup.find_all("script"):
        text = script.string or ""
        if not text:
            continue
        # Only bother with scripts that contain market-value-like content
        if "series" not in text and "data" not in text:
            continue

        # Match [timestamp_ms, value] pairs — timestamp is 13 digits,
        # value is a plain integer (euros, no decimals on TM)
        pairs = re.findall(r'\[(\d{10,13}),\s*(\d+)\]', text)
        for ts_str, val_str in pairs:
            ts = int(ts_str)
            # Timestamps > 10^10 are milliseconds; otherwise seconds
            ts_sec = ts / 1000.0 if ts > 10_000_000_000 else float(ts)
            try:
                val_date = datetime.utcfromtimestamp(ts_sec).date()
            except (ValueError, OSError, OverflowError):
                continue
            delta = abs((val_date - target).days)
            if delta < best_delta:
                best_delta = delta
                best_value = float(val_str)

    if best_value is not None and best_delta <= _MV_WINDOW_DAYS:
        return best_value

    # Reset for approach 2
    best_value = None
    best_delta = float("inf")

    # ── Approach 2: marktwert table (sometimes present on profile page) ───────
    mv_table = soup.find("table", class_=re.compile(r"marktwert", re.I))
    if mv_table:
        for row in mv_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            # Date is usually in the first cell
            date_text = cells[0].get_text(strip=True)
            try:
                row_date = dateutil_parser.parse(date_text, dayfirst=True).date()
            except (ValueError, OverflowError):
                continue
            # Value may be in any cell containing '€'
            val = None
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                if "€" in cell_text:
                    val = parse_market_value(cell_text)
                    if val is not None:
                        break
            if val is None:
                continue
            delta = abs((row_date - target).days)
            if delta < best_delta:
                best_delta = delta
                best_value = val

    if best_value is not None and best_delta <= _MV_WINDOW_DAYS:
        return best_value

    return None


def get_profile_data(
    session: requests.Session,
    player_id: int,
    player_name: str,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Fetch the player's profile page once and return:
      (contract_months_remaining, market_value_eur)

    In MODE == "2025-26": market_value_eur is always None
      (live value comes from the squad page instead).
    In MODE == "2024-25": market_value_eur is the historical value
      closest to 2025-06-01, or None if not found.

    URL: https://www.transfermarkt.com/spieler/profil/spieler/{player_id}
    """
    url = f"{BASE_URL}/spieler/profil/spieler/{player_id}"
    time.sleep(random.uniform(1, 2))

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    WARNING: Profile fetch failed for {player_name}: {e}")
        return None, None

    soup = BeautifulSoup(resp.content, "html.parser")

    contract_months = _extract_contract_months(soup, player_name)

    market_value = None
    if MODE == "2024-25":
        market_value = _extract_historical_mv(soup, player_name)
        if market_value is not None:
            print(f"    Market value (Jun 2025): €{market_value / 1_000_000:.1f}m")
        else:
            print(f"    WARNING: No historical value found for {player_name}")

    return contract_months, market_value


def get_contract_expiry(
    session: requests.Session,
    player_id: int,
    player_name: str,
) -> Optional[int]:
    """Thin wrapper — fetches profile page and returns contract months only."""
    contract_months, _ = get_profile_data(session, player_id, player_name)
    return contract_months


def get_club_urls(session: requests.Session) -> List[dict]:
    """Fetch the league page and return a list of {name, url} dicts for each club."""
    print(f"Fetching league page ({MODE}): {LEAGUE_URL}")
    response = session.get(LEAGUE_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    clubs: List[dict] = []
    seen_urls: set = set()

    def _extract_club(link_tag) -> Tuple[str, str]:
        name = link_tag.get_text(strip=True) or link_tag.get("title", "").strip()
        url = BASE_URL + link_tag["href"]
        return name, url

    table = soup.find("table", {"class": "items"})
    if not table:
        # Fallback: scan all hauptlink anchors pointing to club pages
        for a in soup.select("td.hauptlink a[href*='/startseite/verein/']"):
            name, url = _extract_club(a)
            if url not in seen_urls and name:
                clubs.append({"name": name, "url": url})
                seen_urls.add(url)
        return clubs

    for row in table.find_all("tr", class_=["odd", "even"]):
        # hauptlink td holds the club name — prefer it over any other link
        name_cell = row.find("td", class_="hauptlink")
        link = None
        if name_cell:
            link = name_cell.find("a", href=re.compile(r"/startseite/verein/\d+"))
        if not link:
            link = row.find("a", href=re.compile(r"/startseite/verein/\d+"))
        if link:
            name, url = _extract_club(link)
            if url not in seen_urls and name:
                clubs.append({"name": name, "url": url})
                seen_urls.add(url)

    print(f"  Found {len(clubs)} clubs.")
    return clubs


def scrape_club_players(
    session: requests.Session,
    club_name: str,
    club_url: str,
) -> List[dict]:
    """Scrape all players from a club squad page. Returns a list of player dicts."""
    print(f"  Scraping club: {club_name} — {club_url}")
    time.sleep(random.uniform(2, 3))

    # Retry up to 3 times on timeout / network error
    response = None
    for attempt in range(1, 4):
        try:
            response = session.get(club_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"    Attempt {attempt}/3 failed for {club_name}: {e}")
            if attempt < 3:
                print("    Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"    ERROR: All retries exhausted for {club_name}. Skipping.")
                return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Derive club name from the page itself if the league-page link text was empty
    derived_club_name = club_name.strip() if club_name else ""
    if not derived_club_name:
        title_tag = soup.find("title")
        if title_tag:
            # "Manchester City - Club profile | Transfermarkt" → "Manchester City"
            derived_club_name = title_tag.get_text(strip=True).split(" - ")[0].strip()
    if not derived_club_name:
        h1 = soup.find("h1")
        if h1:
            derived_club_name = h1.get_text(strip=True).split("\n")[0].strip()

    print(f"    Club name resolved: '{derived_club_name}'")

    squad_table = soup.find("table", {"class": "items"})
    if not squad_table:
        print(f"    WARNING: No squad table found for {club_name}")
        return []

    rows = squad_table.find_all("tr", class_=["odd", "even"])
    print(f"    Found {len(rows)} player rows.")

    players = []
    for row in rows:
        try:
            player: dict = {}

            # ── Name + player ID ──
            name_cell = row.find("td", {"class": "hauptlink"})
            if not name_cell:
                continue
            name_link = name_cell.find("a")
            if not name_link:
                continue
            player["player_name"] = name_link.get_text(strip=True)

            # Player ID from the profile URL: /slug/profil/spieler/12345
            player_href = name_link.get("href", "")
            pid_match = re.search(r"profil/spieler/(\d+)", player_href)
            if not pid_match:
                pid_match = re.search(r"/spieler/(\d+)$", player_href)
            player_id = int(pid_match.group(1)) if pid_match else None

            # ── Club ──
            player["club"] = derived_club_name

            cells = row.find_all("td")

            # ── Position ──
            known_positions = [
                "Centre-Forward", "Second Striker", "Left Winger", "Right Winger",
                "Left Midfield", "Right Midfield", "Attacking Midfield",
                "Central Midfield", "Defensive Midfield", "Left-Back", "Right-Back",
                "Left Wing-Back", "Right Wing-Back", "Goalkeeper", "Centre-Back",
            ]
            position = ""
            for cell in cells:
                text = cell.get_text(strip=True)
                if text in known_positions:
                    position = text
                    break
            player["position"] = position

            # ── Age — from parenthesised number in the DOB cell ──
            # DOB cells look like "Feb 25, 1999 (27)" — find all (N) and
            # pick the first one where 15 <= N <= 45
            age = None
            for cell in cells:
                text = cell.get_text(strip=True)
                for match in re.findall(r'\((\d+)\)', text):
                    candidate = int(match)
                    if 15 <= candidate <= 45:
                        age = candidate
                        break
                if age is not None:
                    break
            player["age"] = age

            # ── Nationality — flag image title/alt ──
            nationality = ""
            flag_img = row.find("img", {"class": "flaggenrahmen"})
            if flag_img:
                nationality = flag_img.get("title", "") or flag_img.get("alt", "")
            player["nationality"] = nationality

            # ── Market value + contract — fetched from player profile page ──
            if player_id is not None:
                contract_months, hist_mv = get_profile_data(
                    session, player_id, player["player_name"]
                )
            else:
                contract_months, hist_mv = None, None

            player["contract_months_remaining"] = contract_months

            if MODE == "2024-25":
                # Use historical value extracted from the profile page
                player["market_value_eur"] = hist_mv
            else:
                # MODE == "2025-26": scrape live value from the squad page
                market_value_raw = ""
                for cell in cells:
                    text = cell.get_text(strip=True)
                    if "€" in text:
                        market_value_raw = text
                player["market_value_eur"] = parse_market_value(market_value_raw)

            players.append(player)

            # Progress counter every 10 players
            done = len(players)
            total = len(rows)
            if done % 10 == 0:
                print(f"  Contract data: {done}/{total} players done")

        except Exception as e:
            print(f"    ERROR processing row: {e}")
            continue

    return players


def debug_profile_page(session: requests.Session) -> None:
    """
    Fetch Josko Gvardiol's profile page (ID 314558) and print diagnostic
    info to help debug why historical market value extraction returns nothing.
    Exits after printing — remove the call from main() once fixed.
    """
    url = "https://www.transfermarkt.com/spieler/profil/spieler/314558"
    print("=" * 60)
    print("DEBUG: Fetching Gvardiol profile page")
    print(f"  URL: {url}")
    print("=" * 60)

    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FETCH FAILED: {e}")
        raise SystemExit(1)

    page_source = resp.text
    soup = BeautifulSoup(resp.content, "html.parser")
    print(f"  Page fetched OK — {len(page_source)} chars\n")

    # ── 1. Script tags containing market-value-related keywords ──────────────
    keywords = ["market", "value", "serie", "datum", "marktwert", "highcharts"]
    print("-" * 60)
    print("1. Script tags containing keywords:", keywords)
    matching_scripts = []
    for script in soup.find_all("script"):
        text = script.string or ""
        if any(kw in text.lower() for kw in keywords):
            matching_scripts.append(text)

    print(f"   Found {len(matching_scripts)} matching script tag(s).\n")
    for i, text in enumerate(matching_scripts, 1):
        print(f"   --- Script #{i} (first 500 chars) ---")
        print(text[:500])
        print()

    # ── 2. Elements with market-value-related class names ────────────────────
    print("-" * 60)
    print("2. Elements with market-value-related classes:")
    target_classes = ["marktwert-liste", "market-value", "tm-player-market-value"]
    for cls in target_classes:
        els = soup.find_all(class_=re.compile(cls, re.I))
        print(f"   class~='{cls}': {len(els)} element(s)")
        for el in els[:3]:
            print(f"     [{el.name}] {el.get_text(strip=True)[:200]}")
    print()

    # ── 3. All table elements and their class names ───────────────────────────
    print("-" * 60)
    print("3. All <table> elements and their classes:")
    tables = soup.find_all("table")
    print(f"   Found {len(tables)} table(s).")
    for i, tbl in enumerate(tables, 1):
        cls = tbl.get("class", [])
        print(f"   Table #{i}: class={cls}")
    print()

    # ── 4. Raw [timestamp13digit, value] pairs in page source ─────────────────
    print("-" * 60)
    print("4. re.findall(r'\\[\\d{13},\\d+\\]', page_source) — first 10:")
    pairs_13 = re.findall(r'\[\d{13},\d+\]', page_source)
    if pairs_13:
        for p in pairs_13[:10]:
            print(f"   {p}")
    else:
        print("   (none found)")
    print()

    # ── 4b. Also try 10-digit timestamps ──────────────────────────────────────
    print("4b. re.findall(r'\\[\\d{10},\\d+\\]', page_source) — first 10:")
    pairs_10 = re.findall(r'\[\d{10},\d+\]', page_source)
    if pairs_10:
        for p in pairs_10[:10]:
            print(f"   {p}")
    else:
        print("   (none found)")
    print()

    # ── 4c. The broader pattern used by _extract_historical_mv ───────────────
    print("4c. re.findall(r'\\[(\\d{10,13}),\\s*(\\d+)\\]', page_source) — first 10:")
    pairs_broad = re.findall(r'\[(\d{10,13}),\s*(\d+)\]', page_source)
    if pairs_broad:
        for ts, val in pairs_broad[:10]:
            print(f"   [{ts}, {val}]")
    else:
        print("   (none found)")
    print()

    # ── 5. "y": value pairs ───────────────────────────────────────────────────
    print("-" * 60)
    print('5. re.findall(r\'"y":\\d+\', page_source) — first 10:')
    y_vals = re.findall(r'"y":\d+', page_source)
    if y_vals:
        for v in y_vals[:10]:
            print(f"   {v}")
    else:
        print("   (none found)")
    print()

    # ── Bonus: look for any large integer that looks like a euro amount ────────
    print("-" * 60)
    print("Bonus: any 5-8 digit integers near 'value' or 'marktwert' in scripts:")
    for i, text in enumerate(matching_scripts, 1):
        amounts = re.findall(r'\b(\d{5,8})\b', text)
        if amounts:
            print(f"   Script #{i}: {amounts[:20]}")
    print()

    print("=" * 60)
    print("DEBUG COMPLETE — exiting.")
    print("=" * 60)
    raise SystemExit(0)


def main() -> None:
    os.makedirs(_RAW_DIR, exist_ok=True)

    session = requests.Session()
    all_players: List[dict] = []

    print("=" * 60)
    print(f"Transfermarkt Scraper — Premier League  [{MODE}]")
    print("=" * 60)

    debug_profile_page(session)   # ← remove after debugging

    clubs = get_club_urls(session)
    if not clubs:
        print("ERROR: Could not find any clubs. Check the league URL.")
        return

    for i, club in enumerate(clubs, 1):
        print(f"\n[{i}/{len(clubs)}] {club['name']}")
        players = scrape_club_players(session, club["name"], club["url"])
        all_players.extend(players)
        print(f"  Running total: {len(all_players)} players")
        time.sleep(random.uniform(2, 3))

    df = pd.DataFrame(all_players)
    df = df.drop_duplicates(subset=["player_name", "club"])

    # Drop players with no market value
    before = len(df)
    df = df.dropna(subset=["market_value_eur"])
    dropped = before - len(df)
    if dropped:
        print(f"\nDropped {dropped} players with no market value.")

    # Enforce column order
    base_cols = ["player_name", "club", "position", "age", "nationality",
                 "market_value_eur", "contract_months_remaining"]
    df = df[[c for c in base_cols if c in df.columns]]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Done. Saved {len(df)} players to {OUTPUT_PATH}")

    sample_cols = ["player_name", "club", "age", "market_value_eur",
                   "contract_months_remaining"]
    available = [c for c in sample_cols if c in df.columns]
    print(f"\nSample (first 5 rows):")
    print(df[available].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
