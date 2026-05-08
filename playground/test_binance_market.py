#!/usr/bin/env python3
"""
临时测试脚本：直连币安获取BTC市场数据
"""

import ccxt
import time
from datetime import datetime

def test_binance_market_data():
    """
    测试从币安获取BTC市场数据
    """
    print("=" * 60)
    print("测试币安市场数据获取 (直连模式)")
    print("=" * 60)
    
    try:
        # 初始化币安交易所实例
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        
        # 禁用代理（确保直连）
        if 'proxies' in exchange.options:
            del exchange.options['proxies']
        
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"交易所: {exchange.name}")
        
        # 测试连接
        print("\n测试连接...")
        start_time = time.time()
        exchange.load_markets()
        end_time = time.time()
        print(f"✅ 连接成功，耗时: {end_time - start_time:.2f} 秒")
        
        # 获取BTC/USDT的最新行情
        print("\n获取BTC/USDT最新行情...")
        start_time = time.time()
        ticker = exchange.fetch_ticker('BTC/USDT')
        end_time = time.time()
        
        print(f"✅ 行情获取成功，耗时: {end_time - start_time:.2f} 秒")
        print("\n=== BTC/USDT 市场数据 ===")
        print(f"最新价格: {ticker['last']:.2f} USDT")
        print(f"24h 最高价: {ticker['high']:.2f} USDT")
        print(f"24h 最低价: {ticker['low']:.2f} USDT")
        print(f"24h 交易量: {ticker['quoteVolume']:.2f} USDT")
        print(f"24h 变化: {ticker['percentage']:.2f}%")
        print(f"时间戳: {datetime.fromtimestamp(ticker['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取K线数据
        print("\n获取最近24小时K线数据...")
        start_time = time.time()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=24)
        end_time = time.time()
        
        print(f"✅ K线数据获取成功，耗时: {end_time - start_time:.2f} 秒")
        print("\n=== 最近1小时K线 ===")
        latest_ohlcv = ohlcv[-1]
        print(f"时间: {datetime.fromtimestamp(latest_ohlcv[0]/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"开盘: {latest_ohlcv[1]:.2f}")
        print(f"最高: {latest_ohlcv[2]:.2f}")
        print(f"最低: {latest_ohlcv[3]:.2f}")
        print(f"收盘: {latest_ohlcv[4]:.2f}")
        print(f"成交量: {latest_ohlcv[5]:.2f}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binance_market_data()
