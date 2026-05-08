import os, sys, requests, json
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('FRED_API_KEY')
proxy = os.environ.get('HTTPS_PROXY', os.environ.get('HTTP_PROXY', ''))
proxies = {'http': proxy, 'https': proxy} if proxy else None

print(f"API Key: {api_key[:8]}..." if api_key else "NO API KEY!")
print(f"Proxy: {proxy}")

url = 'https://api.stlouisfed.org/fred/series/observations'
params = {
    'series_id': 'BAMLH0A0HYM2',
    'api_key': api_key,
    'file_type': 'json',
    'observation_start': '2006-01-01',
    'observation_end': '2017-12-31',
    'sort_order': 'asc',
}

print(f"\nTest 1: Standard FRED API request...")
try:
    resp = requests.get(url, params=params, timeout=30, proxies=proxies)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        obs = data.get('observations', [])
        print(f"  Observations: {len(obs)}")
        if obs:
            print(f"  First: {obs[0]}")
            print(f"  Last: {obs[-1]}")
    else:
        print(f"  Error: {resp.text[:500]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print(f"\nTest 2: Without proxy...")
try:
    resp = requests.get(url, params=params, timeout=30, proxies={'http': None, 'https': None})
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        obs = data.get('observations', [])
        print(f"  Observations: {len(obs)}")
        if obs:
            print(f"  First: {obs[0]}")
    else:
        print(f"  Error: {resp.text[:500]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print(f"\nTest 3: FRED series info...")
info_url = 'https://api.stlouisfed.org/fred/series'
info_params = {
    'series_id': 'BAMLH0A0HYM2',
    'api_key': api_key,
    'file_type': 'json',
}
try:
    resp = requests.get(info_url, params=info_params, timeout=30, proxies=proxies)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        ser = data.get('seriess', [{}])[0]
        print(f"  Title: {ser.get('title')}")
        print(f"  Observation start: {ser.get('observation_start')}")
        print(f"  Observation end: {ser.get('observation_end')}")
        print(f"  Frequency: {ser.get('frequency')}")
        print(f"  Seasonal: {ser.get('seasonal_adjustment')}")
    else:
        print(f"  Error: {resp.text[:500]}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")
