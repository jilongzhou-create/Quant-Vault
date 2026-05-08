import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
import requests

api_key = os.environ.get('FRED_API_KEY')
proxy = os.environ.get('HTTP_PROXY', os.environ.get('http_proxy', ''))
proxies = {'http': proxy, 'https': proxy} if proxy else None

url = 'https://api.stlouisfed.org/fred/series/observations'
params = {
    'series_id': 'BAMLH0A0HYM2',
    'api_key': api_key,
    'file_type': 'json',
    'observation_start': '1996-12-31',
    'observation_end': '2018-01-01',
    'sort_order': 'asc',
    'realtime_start': '1996-12-31',
    'realtime_end': '2018-01-01',
}
print('Requesting FRED API with ALFRED realtime period...')
try:
    resp = requests.get(url, params=params, timeout=60, proxies=proxies)
    print('Status:', resp.status_code)
    data = resp.json()
    obs = data.get('observations', [])
    print('Total observations:', len(obs))
    if obs:
        print('First 10:')
        for o in obs[:10]:
            print('  date=%s realtime=%s value=%s' % (o.get('date',''), o.get('realtime_start',''), o.get('value','')))
        print('...')
        print('Last 5:')
        for o in obs[-5:]:
            print('  date=%s realtime=%s value=%s' % (o.get('date',''), o.get('realtime_start',''), o.get('value','')))
except Exception as e:
    print('Error:', type(e).__name__, e)
