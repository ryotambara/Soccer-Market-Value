"""
scraper/whoscored.py

Scrapes Premier League 2024-25 player statistics from WhoScored
using Selenium with Chrome (non-headless to avoid bot detection).

Run: python scraper/whoscored.py

Output: data/raw/whoscored_raw.csv
"""

import os
import time
import random
import pandas as pd
from typing import Optional, List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager

# ── Paths ───────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH  = os.path.join(_BASE_DIR, "data", "raw", "premier_league", "2024-25", "whoscored_raw.csv")

# ── Target URL ───────────────────────────────────────────────────────────────
STATS_URL = (
    "https://www.whoscored.com/Regions/252/Tournaments/2/Seasons/9618"
    "/Stages/22076/PlayerStatistics"
)

TABLE_BODY_ID = "player-table-statistics-body"
NEXT_BTN_ID   = "next"

# ── Column definitions per tab ───────────────────────────────────────────────
# Each entry: (tab_text_fragment, columns_to_extract)
# Columns are extracted positionally from <td> elements in each row.
# Position 0 is always the expand icon; 1=player name; 2=team; etc.

SUMMARY_COLS = [
    "player_name",       # td 1  (first anchor inside)
    "team",              # td 2
    "age",               # td 3
    "position",          # td 4
    "apps",              # td 5
    "mins",              # td 6
    "goals",             # td 7
    "assists",           # td 8
    "shots_per_game",    # td 9
    "key_passes_per_game",   # td 10
    "tackles_per_game",      # td 11
    "interceptions_per_game",# td 12
    "aerials_won_per_game",  # td 13
    "dribbles_per_game",     # td 14
    "pass_success_pct",      # td 15
    "rating",                # td 16 (last)
]

DEFENSIVE_COLS = [
    "player_name",            # td 1
    "team",                   # td 2
    "tackles_per_game",       # td 3
    "interceptions_per_game", # td 4
    "fouls_per_game",         # td 5
    "aerials_won_per_game",   # td 6
    "clearances_per_game",    # td 7
    "blocked_per_game",       # td 8
]

PASSING_COLS = [
    "player_name",           # td 1
    "team",                  # td 2
    "key_passes_per_game",   # td 3
    "pass_success_pct",      # td 4
    "dribbles_per_game",     # td 5
]

# ── Driver setup ─────────────────────────────────────────────────────────────

def make_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # Do NOT add --headless — WhoScored blocks headless browsers
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


# ── Cloudflare check ─────────────────────────────────────────────────────────

def check_cloudflare(driver: webdriver.Chrome) -> bool:
    title = driver.title.lower()
    if "just a moment" in title or "cloudflare" in title:
        print("  WARNING: Cloudflare challenge detected. Waiting 10s...")
        time.sleep(10)
        return True
    return False


# ── Table scraping ────────────────────────────────────────────────────────────

def wait_for_table(driver: webdriver.Chrome, timeout: int = 15) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, TABLE_BODY_ID))
        )
        return True
    except TimeoutException:
        print(f"  WARNING: Table did not load within {timeout}s.")
        return False


def extract_row(row_el, col_names: List[str]) -> Optional[Dict[str, str]]:
    """
    Extract text values from a table row element.
    col_names maps positionally to td[1], td[2], ... (td[0] is the expand icon).
    Player name is extracted from the first <a> inside td[1].
    """
    try:
        tds = row_el.find_elements(By.TAG_NAME, "td")
        # Need at least enough cells
        if len(tds) < len(col_names) + 1:
            return None

        record = {}
        for i, col in enumerate(col_names):
            td = tds[i + 1]  # offset by 1 to skip expand icon td
            if col == "player_name":
                # Player name is inside an <a> tag
                try:
                    record[col] = td.find_element(By.TAG_NAME, "a").text.strip()
                except NoSuchElementException:
                    record[col] = td.text.strip()
            else:
                record[col] = td.text.strip()

        return record if record.get("player_name") else None

    except StaleElementReferenceException:
        return None
    except Exception:
        return None


def scrape_current_tab(
    driver: webdriver.Chrome,
    col_names: List[str],
    tab_label: str,
) -> List[Dict[str, str]]:
    """Scrape all pages of the currently active tab."""
    all_rows = []
    page = 1

    while True:
        if not wait_for_table(driver):
            print(f"  Skipping remainder of '{tab_label}' tab — table not found.")
            break

        if check_cloudflare(driver):
            if not wait_for_table(driver, timeout=20):
                break

        # Scroll slightly to trigger any lazy loading
        driver.execute_script("window.scrollTo(0, 300)")
        time.sleep(1)

        try:
            tbody = driver.find_element(By.ID, TABLE_BODY_ID)
            rows = tbody.find_elements(By.TAG_NAME, "tr")
        except NoSuchElementException:
            print(f"  WARNING: Table body missing on page {page}.")
            break

        page_records = []
        for row_el in rows:
            record = extract_row(row_el, col_names)
            if record:
                page_records.append(record)

        all_rows.extend(page_records)
        print(f"  [{tab_label}] Page {page}: scraped {len(page_records)} players "
              f"(total so far: {len(all_rows)})")

        # Try to click Next
        try:
            next_btn = driver.find_element(By.ID, NEXT_BTN_ID)
            btn_class = next_btn.get_attribute("class") or ""
            if "disabled" in btn_class or not next_btn.is_enabled():
                print(f"  [{tab_label}] Reached last page ({page}).")
                break
            next_btn.click()
            page += 1
            time.sleep(random.uniform(2, 4))
        except NoSuchElementException:
            # Try alternate selector
            try:
                next_btn = driver.find_element(
                    By.CSS_SELECTOR, "a.next, li.next a, .paginate_button.next"
                )
                btn_class = next_btn.get_attribute("class") or ""
                if "disabled" in btn_class:
                    break
                next_btn.click()
                page += 1
                time.sleep(random.uniform(2, 4))
            except NoSuchElementException:
                print(f"  [{tab_label}] No next button found — assuming last page.")
                break

    return all_rows


def click_tab(driver: webdriver.Chrome, tab_text: str) -> bool:
    """
    Find and click a stats tab by matching its visible text.
    Tab links are <li> elements with class containing 'stats-type-choice'
    or similar navigation elements.
    """
    time.sleep(random.uniform(3, 5))

    selectors = [
        "li.stats-type-choice a",
        "ul.stat-type li a",
        "div.tabs li a",
        "ul.tabs li a",
        "#tablist li a",
        "li.tab a",
    ]

    for selector in selectors:
        try:
            tab_links = driver.find_elements(By.CSS_SELECTOR, selector)
            for link in tab_links:
                if tab_text.lower() in link.text.strip().lower():
                    driver.execute_script("window.scrollTo(0, 0)")
                    time.sleep(0.5)
                    link.click()
                    print(f"  Clicked tab: '{link.text.strip()}'")
                    time.sleep(random.uniform(3, 5))
                    return True
        except Exception:
            continue

    # Fallback: try finding by partial link text
    try:
        link = driver.find_element(By.PARTIAL_LINK_TEXT, tab_text)
        link.click()
        print(f"  Clicked tab via partial text: '{tab_text}'")
        time.sleep(random.uniform(3, 5))
        return True
    except NoSuchElementException:
        pass

    print(f"  WARNING: Could not find tab '{tab_text}'. Skipping.")
    return False


# ── Merge helper ──────────────────────────────────────────────────────────────

def merge_tabs(
    summary: List[Dict],
    defensive: List[Dict],
    passing: List[Dict],
) -> pd.DataFrame:
    df_sum = pd.DataFrame(summary)
    df_def = pd.DataFrame(defensive) if defensive else pd.DataFrame()
    df_pas = pd.DataFrame(passing)  if passing   else pd.DataFrame()

    # Defensive: keep only columns not already in summary
    def_extra = [c for c in DEFENSIVE_COLS
                 if c not in SUMMARY_COLS and c != "player_name" and c != "team"]
    pas_extra = [c for c in PASSING_COLS
                 if c not in SUMMARY_COLS and c != "player_name" and c != "team"]

    merged = df_sum
    if not df_def.empty and def_extra:
        keep = ["player_name", "team"] + [c for c in def_extra if c in df_def.columns]
        merged = merged.merge(df_def[keep], on=["player_name", "team"], how="left",
                              suffixes=("", "_def"))

    if not df_pas.empty and pas_extra:
        keep = ["player_name", "team"] + [c for c in pas_extra if c in df_pas.columns]
        merged = merged.merge(df_pas[keep], on=["player_name", "team"], how="left",
                              suffixes=("", "_pas"))

    # Final column order
    final_cols = [
        "player_name", "team", "age", "position", "apps", "mins",
        "goals", "assists", "shots_per_game", "key_passes_per_game",
        "tackles_per_game", "interceptions_per_game", "aerials_won_per_game",
        "dribbles_per_game", "pass_success_pct", "clearances_per_game",
        "blocked_per_game", "fouls_per_game", "rating",
    ]
    present = [c for c in final_cols if c in merged.columns]
    return merged[present]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=" * 60)
    print("WhoScored Scraper — Premier League 2024-25")
    print("=" * 60)

    driver = make_driver()

    try:
        print(f"\nLoading: {STATS_URL}")
        driver.get(STATS_URL)
        time.sleep(5)

        check_cloudflare(driver)

        # ── DEBUG: inspect page structure before any scraping ────────────
        print("\n" + "=" * 60)
        print("DEBUG — page structure inspection")
        print("=" * 60)

        print(f"\n[1] Page title: {driver.title!r}")

        print("\n[2] First 3000 chars of page source:")
        print(driver.page_source[:3000])

        print("\n[3] First 30 elements with IDs:")
        id_els = driver.find_elements(By.XPATH, "//*[@id]")
        for el in id_els[:30]:
            print(f"  id={el.get_attribute('id')!r}  tag={el.tag_name}")

        print("\n[4] Table body selector probe:")
        selectors = [
            "#player-table-statistics-body",
            "#statistics-table-summary tbody",
            ".player-table tbody",
            "table.grid tbody",
            "#top-player-stats-summary tbody",
        ]
        found_tbody = None
        for sel in selectors:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            print(f"  {sel}: {len(els)} elements found")
            if els and found_tbody is None:
                found_tbody = els[0]

        print("\n[5] First 500 chars of first tbody found:")
        if found_tbody:
            print(found_tbody.get_attribute("innerHTML")[:500])
        else:
            print("  (no tbody found with any selector)")

        print("\n" + "=" * 60)
        print("DEBUG complete — exiting without scraping.")
        print("=" * 60)

    finally:
        driver.quit()
        print("\nBrowser closed.")

    # (scraping disabled during debug — remove debug block to re-enable)
    return
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
