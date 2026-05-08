#!/usr/bin/env python3
"""
FMP (Financial Modeling Prep) API 测试脚本
验证 OHLCV 和财报数据的拉取，特别是 fillingDate/acceptedDate
"""

import os
import sys
import json
import requests
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: Please install python-dotenv")
    sys.exit(1)

def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_separator("环境与配置加载")
    
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY")
    
    if not api_key:
        print("ERROR: FMP_API_KEY not found in .env")
        sys.exit(1)
    
    print(f"FMP_API_KEY loaded (first 10 chars): {api_key[:10]}...")
    
    symbol = "TSLA"
    
    print_separator("量价测试 (OHLCV)")
    
    ohlcv_url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&apikey={api_key}"
    
    try:
        print(f"Request URL: {ohlcv_url}")
        response = requests.get(ohlcv_url, timeout=30)
        
        if response.status_code != 200:
            print(f"ERROR: HTTP status code: {response.status_code}")
            print(f"Response:\n{response.text}")
        else:
            data = response.json()
            
            if not isinstance(data, list):
                print(f"ERROR: Response is not a list")
                print(f"Full response:\n{json.dumps(data, indent=2)}")
            else:
                records = data[:5]
                print(f"SUCCESS: Retrieved {len(data)} records")
                print(f"First 5 records:")
                
                df = pd.DataFrame(records)
                cols_to_show = ["date", "open", "high", "low", "close", "volume"]
                print(df[cols_to_show].to_string(index=False))
                
    except Exception as e:
        print(f"ERROR: OHLCV request failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("财报测试 (Fundamentals)")
    
    financials_url = f"https://financialmodelingprep.com/stable/income-statement?symbol={symbol}&period=quarter&limit=4&apikey={api_key}"
    
    try:
        print(f"Request URL: {financials_url}")
        response = requests.get(financials_url, timeout=30)
        
        if response.status_code != 200:
            print(f"ERROR: HTTP status code: {response.status_code}")
            print(f"Response:\n{response.text}")
        else:
            data = response.json()
            
            if not isinstance(data, list):
                print(f"ERROR: Response is not a list")
                print(f"Full response:\n{json.dumps(data, indent=2)}")
            else:
                print(f"SUCCESS: Retrieved {len(data)} quarterly income statements")
                print(f"\nKey date information:")
                print(f"{'fiscalYear':<14} {'period':<10} {'date':<14} {'fillingDate':<19} {'acceptedDate':<19}")
                print("-" * 80)
                
                for item in data:
                    fiscal_year = item.get("fiscalYear", "N/A")
                    period = item.get("period", "N/A")
                    date_str = item.get("date", "N/A")
                    filling_date = item.get("filingDate", "N/A")
                    accepted_date = item.get("acceptedDate", "N/A")
                    
                    print(f"{fiscal_year:<14} {period:<10} {date_str:<14} {filling_date:<19} {accepted_date:<19}")
                
                print(f"\nFull first record (to see all fields):")
                print(json.dumps(data[0], indent=2) if data else "No data")
                
    except Exception as e:
        print(f"ERROR: Financials request failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("测试完成")

if __name__ == "__main__":
    main()
