"""
scraper/transfermarkt.py

Scrapes player data from Transfermarkt for any supported league.

Two knobs at the top of this file:

  MODE   — season to scrape
  LEAGUE — league to scrape

Output files:
  data/raw/transfermarkt_{league}_{mode}.csv
  e.g. transfermarkt_pl_2025-26.csv
       transfermarkt_bundesliga_2025-26.csv

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

# ── Change these to switch season / league ────────────────────────────────
MODE   = "2025-26"    # "2025-26" | "2024-25"
LEAGUE = "bundesliga" # "premier-league" | "bundesliga"
# ──────────────────────────────────────────────────────────────────────────

LEAGUE_CONFIG = {
    "premier-league": {
        "url_slug":      "premier-league/startseite/wettbewerb/GB1",
        "output_suffix": "pl",
        "name":          "Premier League",
    },
    "bundesliga": {
        "url_slug":      "bundesliga/startseite/wettbewerb/L1",
        "output_suffix": "bundesliga",
        "name":          "Bundesliga",
    },
}

BASE_URL = "https://www.transfermarkt.com"

# Transfermarkt encodes the season year as the start year (2024 = 2024-25)
_SAISON = "2025" if MODE == "2025-26" else "2024"

_league_cfg = LEAGUE_CONFIG[LEAGUE]
LEAGUE_URL  = f"{BASE_URL}/{_league_cfg['url_slug']}/saison_id/{_SAISON}"

_RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(
    _RAW_DIR, f"transfermarkt_{_league_cfg['output_suffix']}_{MODE}.csv"
)

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
    Parse contract months remaining from a Transfermarkt profile page.

    Three strategies, tried in order:
      1. Regex on full page text for "Contract expires: DD/MM/YYYY" (and German equiv)
      2. Table row scan: label cell contains "contract"/"vertrag", value cell has date
      3. DOM sibling scan: element whose text contains "expires" → next sibling date
    """
    text = soup.get_text(" ", strip=True)
    now  = datetime.today()

    def _months(expiry: datetime) -> int:
        return max(0, (expiry.year - now.year) * 12 + (expiry.month - now.month))

    def _try_parse(date_str: str, fmts: List[str]) -> Optional[datetime]:
        for fmt in fmts:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    # ── Strategy 1: "Contract expires" / "Vertragsende" in page text ─────────
    label_patterns = [
        (r'Contract expires[:\s]+(\d{1,2}/\d{2}/\d{4})', ['%d/%m/%Y']),
        (r'Contract expires[:\s]+(\d{1,2}/\d{4})',        ['%m/%Y']),
        (r'Vertragsende[:\s]+(\d{1,2}\.\d{2}\.\d{4})',    ['%d.%m.%Y']),
        (r'Vertragsende[:\s]+(\d{2}/\d{4})',               ['%m/%Y']),
    ]
    for pattern, fmts in label_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            expiry = _try_parse(m.group(1), fmts)
            if expiry:
                result = _months(expiry)
                print(f"    [CONTRACT] {player_name}: '{m.group(0)}' → {result} months")
                return result

    # ── Strategy 2: <tr> scan — label cell + value cell ──────────────────────
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()
        if not any(kw in label for kw in ("contract", "vertrag", "expires")):
            continue
        value = cells[1].get_text(strip=True)
        m = re.search(r'(\d{1,2})[/.](\d{2})[/.](\d{4})', value)
        if m:
            try:
                expiry = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                result = _months(expiry)
                print(f"    [CONTRACT] {player_name}: row '{value}' → {result} months")
                return result
            except ValueError:
                continue

    # ── Strategy 3: element containing "expires" → next sibling date ─────────
    for el in soup.find_all(["span", "td", "div", "li"]):
        if "expires" not in el.get_text(strip=True).lower():
            continue
        for sibling in list(el.next_siblings)[:5]:
            if not hasattr(sibling, "get_text"):
                continue
            val = sibling.get_text(strip=True)
            m = re.search(r'(\d{1,2})/(\d{2})/(\d{4})', val)
            if m:
                try:
                    expiry = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                    result = _months(expiry)
                    print(f"    [CONTRACT] {player_name}: sibling '{val}' → {result} months")
                    return result
                except ValueError:
                    continue

    return None


def get_contract_expiry(
    session: requests.Session,
    player_id: int,
    player_name: str,
) -> Optional[int]:
    """Fetch the player's profile page and return contract months remaining."""
    url = f"{BASE_URL}/spieler/profil/spieler/{player_id}"
    time.sleep(random.uniform(1, 2))

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    WARNING: Profile fetch failed for {player_name}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    return _extract_contract_months(soup, player_name)


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

            # ── Market value — live from squad page ──
            market_value_raw = ""
            for cell in cells:
                text = cell.get_text(strip=True)
                if "€" in text:
                    market_value_raw = text
            player["market_value_eur"] = parse_market_value(market_value_raw)

            # ── Contract expiry ───────────────────────────────────────
            # In 2025-26 mode the contract date is JS-rendered and not
            # reliably parseable from static HTML.  Skip the profile
            # fetch entirely; clean.py will fill with the median.
            if MODE == "2025-26":
                player["contract_months_remaining"] = None
            else:
                player["contract_months_remaining"] = (
                    get_contract_expiry(session, player_id, player["player_name"])
                    if player_id is not None else None
                )

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


def debug_contract_parsing(session: requests.Session) -> None:
    """
    Fetch Upamecano's profile page and print diagnostic info to fix
    contract date parsing. Exits after printing.
    """
    url = f"{BASE_URL}/spieler/profil/spieler/346532"
    print("=" * 60)
    print("DEBUG: Contract parsing — Upamecano (346532)")
    print(f"  URL: {url}")
    print("=" * 60)

    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FETCH FAILED: {e}")
        raise SystemExit(1)

    soup = BeautifulSoup(resp.content, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # 1. First 3000 chars of page text
    print("\n--- 1. Page text (first 3000 chars) ---")
    print(page_text[:3000])

    # 2. <td> elements containing "Contract", "contract", or "Vertrag"
    print("\n--- 2. <td> elements with contract/Vertrag text ---")
    found = False
    for td in soup.find_all("td"):
        text = td.get_text(strip=True)
        if any(kw in text for kw in ("Contract", "contract", "Vertrag", "vertrag")):
            print(f"  [{td.get('class', '')}] {text[:200]}")
            found = True
    if not found:
        print("  (none found)")

    # 3. Elements with class containing "contract" or "vertrag"
    print("\n--- 3. Elements with class~=contract/vertrag ---")
    found = False
    for el in soup.find_all(class_=re.compile(r"contract|vertrag", re.I)):
        print(f"  <{el.name} class={el.get('class')}> {el.get_text(strip=True)[:200]}")
        found = True
    if not found:
        print("  (none found)")

    # 4. Date patterns in full page text
    print("\n--- 4. Date patterns found in page text ---")
    dates = re.findall(
        r'\d{1,2}/\d{4}|\w+ \d{4}|\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}',
        page_text,
    )
    print("  Date patterns:", dates[:20])

    # 5. Run current _extract_contract_months and show result
    print("\n--- 5. _extract_contract_months result ---")
    result = _extract_contract_months(soup, "Upamecano")
    print(f"  contract_months_remaining = {result}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE — exiting.")
    print("=" * 60)
    raise SystemExit(0)


def main() -> None:
    os.makedirs(_RAW_DIR, exist_ok=True)

    session = requests.Session()
    all_players: List[dict] = []

    print("=" * 60)
    print(f"Transfermarkt Scraper — {_league_cfg['name']}  [{MODE}]")
    print(f"League URL: {LEAGUE_URL}")
    print(f"Output:     {OUTPUT_PATH}")
    print("=" * 60)

    if MODE == "2025-26":
        print("Note: contract months not scraped (JS-rendered) — "
              "pipeline will fill with median.")

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
