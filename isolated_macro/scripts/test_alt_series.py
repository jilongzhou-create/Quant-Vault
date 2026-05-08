import os, sys, requests, json, time
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('FRED_API_KEY')
proxy = os.environ.get('HTTPS_PROXY', os.environ.get('HTTP_PROXY', ''))
proxies = {'http': proxy, 'https': proxy} if proxy else None

url_info = 'https://api.stlouisfed.org/fred/series'
url_obs = 'https://api.stlouisfed.org/fred/series/observations'

alt_series = ['BAA10YM', 'AAA10YM', 'BAMLH0A0HYM2', 'BAMLH0A1AABBB', 'BAMLH0A3BBBC', 'BAMLH0A2BBB']

print("Checking alternative credit spread series on FRED:")
print("-" * 80)
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
            freq = ser.get('frequency', 'N/A')
            print(f"  {sid}: start={obs_start}, end={obs_end}, freq={freq}")
            print(f"    title={title[:70]}")
        else:
            print(f"  {sid}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  {sid}: {type(e).__name__}")
    time.sleep(0.3)

print("\n" + "=" * 80)
print("Testing BAA10YM data availability for 2008 GFC period...")
params = {
    'series_id': 'BAA10YM',
    'api_key': api_key,
    'file_type': 'json',
    'observation_start': '2006-01-01',
    'observation_end': '2010-12-31',
    'sort_order': 'asc',
}
try:
    resp = requests.get(url_obs, params=params, timeout=30, proxies=proxies)
    if resp.status_code == 200:
        data = resp.json()
        obs = data.get('observations', [])
        print(f"  BAA10YM 2006-2010: {len(obs)} observations")
        if obs:
            valid_obs = [o for o in obs if o.get('value', '.') != '.']
            print(f"  Valid observations: {len(valid_obs)}")
            if valid_obs:
                print(f"  First: date={valid_obs[0]['date']}, value={valid_obs[0]['value']}")
                print(f"  Last: date={valid_obs[-1]['date']}, value={valid_obs[-1]['value']}")
                peak_obs = max(valid_obs, key=lambda x: float(x['value']))
                print(f"  Peak: date={peak_obs['date']}, value={peak_obs['value']}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print("\nTesting BAA10YM full range for IS period (2007-2019)...")
params['observation_start'] = '2007-01-01'
params['observation_end'] = '2019-12-31'
try:
    resp = requests.get(url_obs, params=params, timeout=30, proxies=proxies)
    if resp.status_code == 200:
        data = resp.json()
        obs = data.get('observations', [])
        valid_obs = [o for o in obs if o.get('value', '.') != '.']
        print(f"  BAA10YM 2007-2019: {len(valid_obs)} valid observations")
        if valid_obs:
            vals = [float(o['value']) for o in valid_obs]
            print(f"  Value range: {min(vals):.2f} ~ {max(vals):.2f}")
            print(f"  Mean: {sum(vals)/len(vals):.2f}")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")
