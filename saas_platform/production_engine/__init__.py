"""
云端实盘生产大脑 (Cloud Production Engine)

每日定时运行，完成两个核心动作：
  1. 增量拉取最新行情与因子数据 → Supabase
  2. 热加载策略代码，计算信号与模拟净值 → Supabase
"""
