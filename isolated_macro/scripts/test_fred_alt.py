import os, sys, requests, json, sqlite3, time
import numpy as np
import pandas as pd
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()
from config import DB_PATH

api_key = os.environ.get('FRED_API_KEY')
proxy = os.environ.get('HTTPS_PROXY', os.environ.get('HTTP_PROXY', ''))
proxies = {'http': proxy, 'https': proxy} if proxy else None

print("=" * 60)
print("  Testing multiple methods to get BAMLH0A0HYM2 historical data")
print("=" * 60)

# Method 1: FRED CSV download
print("\n[Method 1] FRED CSV download URL...")
csv_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
try:
    resp = requests.get(csv_url, timeout=30, proxies=proxies)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200 and len(resp.text) > 50:
        lines = resp.text.strip().split('\n')
        print(f"  Total lines: {len(lines)}")
        print(f"  Header: {lines[0]}")
        print(f"  First 5 data rows:")
        for line in lines[1:6]:
            print(f"    {line}")
        print(f"  Last 5 data rows:")
        for line in lines[-5:]:
            print(f"    {line}")
    else:
        print(f"  Response too short or error: {resp.text[:200]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

# Method 2: FRED API with vintage dates (ALFRED)
print("\n[Method 2] FRED API with ALFRED vintage dates...")
url = 'https://api.stlouisfed.org/fred/series/observations'
for vintage_start, vintage_end in [
    ('2006-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01'),
    ('2015-01-01', '2020-01-01'),
]:
    params = {
        'series_id': 'BAMLH0A0HYM2',
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '2006-01-01',
        'observation_end': '2017-12-31',
        'realtime_start': vintage_start,
        'realtime_end': vintage_end,
    }
    try:
        resp = requests.get(url, params=params, timeout=30, proxies=proxies)
        data = resp.json()
        obs = data.get('observations', [])
        print(f"  Vintage {vintage_start}~{vintage_end}: {len(obs)} observations")
        if obs:
            print(f"    First: date={obs[0].get('date')}, value={obs[0].get('value')}")
            print(f"    Last: date={obs[-1].get('date')}, value={obs[-1].get('value')}")
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {e}")
    time.sleep(0.5)

# Method 3: Try alternative series HYS (Moody's Baa-High Yield Spread)
print("\n[Method 3] Try alternative series...")
alt_series = ['BAMLH0A0HYM2', 'BAMLH0A1AABBB', 'BAMLH0A3BBBC', 'HYS', 'BAA10YM', 'AAA10YM']
url_info = 'https://api.stlouisfed.org/fred/series'
for sid in alt_series:
    try:
        resp = requests.get(url_info, params={
            'series_id': sid, 'api_key': api_key, 'file_type': 'json'
        }, timeout=15, proxies=proxies)
        if resp.status_code == 200:
            data = resp.json()
            ser = data.get('seriess', [{}])[0]
            obs_start = ser.get('observation_start', 'N/A')
            obs_end = ser.get('observation_end', 'N/A')
            title = ser.get('title', 'N/A')
            print(f"  {sid}: start={obs_start}, end={obs_end}, title={title[:60]}")
        else:
            print(f"  {sid}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  {sid}: {type(e).__name__}")
    time.sleep(0.3)
