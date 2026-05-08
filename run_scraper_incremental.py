import subprocess
import sys
import os
import time
import sqlite3
import logging
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from config import DB_PATH

FED_BASE = "https://www.federalreserve.gov"

def fetch_url(url):
    env = os.environ.copy()
    for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
        env.pop(k, None)
    env['no_proxy'] = '*'
    env['NO_PROXY'] = '*'
    result = subprocess.run(
        ['curl.exe', '-s', '-k', '-L', '--max-time', '15', url],
        capture_output=True, timeout=25, env=env
    )
    if result.returncode == 0 and result.stdout:
        return result.stdout.decode('utf-8', errors='replace').lstrip('\ufeff')
    return ""

def extract_links_calendar(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if re.match(r'/newsevents/pressreleases/monetary\d{8}a\.htm$', href):
            dm = re.search(r'monetary(\d{4})(\d{2})(\d{2})a\.htm', href)
            if dm:
                rd = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                links.append({'release_date': rd, 'url': urljoin(FED_BASE, href)})
    return links

def extract_links_historical(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if text != 'Statement':
            continue
        if re.match(r'/newsevents/pressreleases/monetary\d{8}a\.htm$', href):
            dm = re.search(r'monetary(\d{4})(\d{2})(\d{2})a\.htm', href)
            if dm:
                rd = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                links.append({'release_date': rd, 'url': urljoin(FED_BASE, href)})
        elif re.match(r'/newsevents/press/monetary/\d{8}a\.htm$', href):
            dm = re.search(r'/(\d{4})(\d{2})(\d{2})a\.htm', href)
            if dm:
                rd = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
                links.append({'release_date': rd, 'url': urljoin(FED_BASE, href)})
    return links

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    content_div = (
        soup.select_one('.fomc-statement')
        or soup.select_one('#article')
        or soup.select_one('.col-md-8')
        or soup.select_one('article')
    )
    if not content_div:
        content_div = soup.find('div', class_=re.compile(r'content|article|statement', re.I))
    if not content_div:
        return ""
    for tag in content_div.select('script, style, .article__time, nav, footer, header'):
        tag.decompose()
    text = content_div.get_text(separator='\n', strip=True)
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(Voting|Vote|Implementation|Home|Board|Federal Reserve|Feedback|Accessibility|Last Update|Disclaimer|Share)', line, re.I):
            continue
        if len(line) < 10 and not re.match(r'^\d{4}$', line):
            continue
        lines.append(line)
    return '\n'.join(lines)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT release_date FROM nlp_fomc_raw")
existing = set(r[0] for r in cursor.fetchall())
conn.close()
logger.info(f"Already in DB: {len(existing)} records")

all_links = []

logger.info("Fetching calendar page...")
html = fetch_url(f"{FED_BASE}/monetarypolicy/fomccalendars.htm")
links = extract_links_calendar(html)
logger.info(f"Calendar: {len(links)} links")
all_links.extend(links)

for year in range(2014, 2027):
    logger.info(f"Fetching historical {year}...")
    html = fetch_url(f"{FED_BASE}/monetarypolicy/fomchistorical{year}.htm")
    if html:
        links = extract_links_historical(html)
        logger.info(f"  {year}: {len(links)} links")
        all_links.extend(links)
    time.sleep(0.3)

seen = set()
unique = []
for l in all_links:
    if l['release_date'] not in seen:
        seen.add(l['release_date'])
        unique.append(l)
unique.sort(key=lambda x: x['release_date'])

to_fetch = [l for l in unique if l['release_date'] not in existing]
logger.info(f"Total unique: {len(unique)}, Need to fetch: {len(to_fetch)}")

saved = 0
failed = 0
for idx, link in enumerate(to_fetch, 1):
    rd = link['release_date']
    url = link['url']
    logger.info(f"  [{idx}/{len(to_fetch)}] {rd}")
    html = fetch_url(url)
    text = extract_text(html)
    if text and len(text) >= 200:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO nlp_fomc_raw (release_date, statement_url, statement_text, fetch_status) VALUES (?, ?, ?, 'fetched')",
            (rd, url, text)
        )
        conn.commit()
        conn.close()
        saved += 1
    else:
        logger.warning(f"    FAILED: text_len={len(text) if text else 0}")
        failed += 1
    time.sleep(0.2)

logger.info(f"Done! Saved={saved}, Failed={failed}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*), MIN(release_date), MAX(release_date) FROM nlp_fomc_raw")
row = cursor.fetchone()
conn.close()
logger.info(f"DB total: {row[0]} records, {row[1]} ~ {row[2]}")
