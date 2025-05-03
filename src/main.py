try:
    import akshare as ak
    # Increase the default timeout for requests
    import requests
    # Remove the following two lines:
    # requests.adapters.DEFAULT_RETRIES = 5
    # requests.DEFAULT_TIMEOUT = 30  # Increase from default 15 seconds
except ModuleNotFoundError:
    raise ModuleNotFoundError("akshare module is not installed. Please install it using 'pip install akshare openpyxl' before running this script.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import matplotlib as mpl

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # For macOS
mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略 AKShare 可能产生的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 参数设置
benchmark_code = '000300' # AKShare 通常不需要 .SH 后缀
stock_num = 10
max_drawdown = 0.10 # 最大回撤容忍度 (基于峰值计算)
start_date = '20210101'
end_date = '20240101'
start_date_dt = datetime.strptime(start_date, '%Y%m%d')
end_date_dt = datetime.strptime(end_date, '%Y%m%d')

# 获取指数行情作为基准 (使用 AKShare)
try:
    # Add retry logic for network operations
    def retry_function(func, max_tries=3, delay=5):
        import time
        tries = 0
        while tries < max_tries:
            try:
                return func()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                tries += 1
                if tries == max_tries:
                    raise
                print(f"Network error: {e}. Retrying in {delay} seconds... (Attempt {tries}/{max_tries})")
                time.sleep(delay)
    
    # Then use it like:
    # Use local test data for development
    import os
    
    # Check if we have cached data
    cache_file = f"cache_{benchmark_code}_{start_date}_{end_date}.csv"
    try:
        if os.path.exists(cache_file):
            print(f"Using cached benchmark data from {cache_file}")
            bench_df = pd.read_csv(cache_file)
            bench_df['date'] = pd.to_datetime(bench_df['trade_date'])
            bench_df.set_index('date', inplace=True)
        else:
            # Fetch from API with retry logic
            bench_df = retry_function(lambda: ak.index_zh_a_hist(symbol=benchmark_code, period="daily", 
                                                               start_date=start_date, end_date=end_date))
            if bench_df is None or bench_df.empty:
                raise ValueError("AKShare 指数行情数据获取失败或为空")
            # Save to cache
            bench_df.rename(columns={'日期': 'trade_date', '收盘': 'close'}, inplace=True)
            bench_df.to_csv(cache_file, index=False)
            bench_df['date'] = pd.to_datetime(bench_df['trade_date'])
            bench_df.set_index('date', inplace=True)
    except Exception as e:
        raise ValueError(f"获取指数行情数据时出错: {e}")
    
    bench_df.rename(columns={'日期': 'trade_date', '收盘': 'close'}, inplace=True)
    bench_df['date'] = pd.to_datetime(bench_df['trade_date'])
    bench_df.set_index('date', inplace=True)
    bench_df = bench_df.sort_index() # 确保按日期排序
    
    # --- 使用静态股票列表代替动态选股 ---
    print("使用静态股票列表进行回测...")
    # 常见的大市值股票作为示例
    static_selected_codes = ['600519', '601318', '600036', '601166', '600276', 
                             '000858', '601888', '600900', '601398', '600887']
    print(f"使用静态股票列表: {static_selected_codes}")
    
    # 简化的策略框架
    # 使用月末作为调仓日
    # --- 多因子动态选股模型 ---
    print("使用多因子模型进行动态选股回测...")
    
    # 定义因子及其重命名映射
    factor_map = {
        '净资产收益率ROE(加权)': 'roe',
        '净利润同比增长率': 'netprofit_yoy',
        '营业总收入同比增长率': 'revenue_yoy',
        '市盈率(动)': 'pe'
    }
    factor_cols = list(factor_map.keys())
    
    rebalance_dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='M')
    net_value_history = []
    max_value_history = []
    value = 1.
    current_max_value = 1.0
    triggered = False
    
    for current_date in rebalance_dates:
        current_date_str = current_date.strftime('%Y%m%d')
        try:
            # Get CSI 500 constituents
            hs500_df = ak.index_stock_cons(symbol="000905")
            hs500_codes = hs500_df['品种代码'].tolist()
            # Get latest financial indicators
            all_indicators = []
            for code in hs500_codes:
                try:
                    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2019")
                    if not df.empty and '披露日期' in df.columns:
                        df['披露日期'] = pd.to_datetime(df['披露日期'], errors='coerce')
                        # 只保留披露日早于等于当前调仓日的数据
                        df_valid = df[df['披露日期'] <= current_date]
                        if not df_valid.empty:
                            latest = df_valid.sort_values('披露日期').iloc[-1].copy()
                            latest['股票代码'] = code
                            all_indicators.append(latest)
                except Exception as e:
                    print(f"获取{code}财务数据失败: {e}")
            indicators = pd.DataFrame(all_indicators)
            print(f"Columns in indicators: {indicators.columns.tolist()}")  # Debug print

            # Try to find the correct code column
            code_col = None
            for candidate in ['股票代码', '证券代码', '代码']:
                if candidate in indicators.columns:
                    code_col = candidate
                    break
            if code_col is None:
                print(f"{current_date_str}: No valid stock code column found in indicators, skipping selection.")
                selected = []
                continue

            # Only keep data before rebalance date
            if '披露日期' in indicators.columns:
                indicators['披露日期'] = pd.to_datetime(indicators['披露日期'], errors='coerce')
                indicators = indicators[indicators['披露日期'] <= current_date]
                indicators = indicators.sort_values('披露日期')
                indicators = indicators.drop_duplicates(code_col, keep='last')
            # Only keep CSI 500 stocks
            indicators = indicators[indicators[code_col].isin(hs500_codes)]
            # Check available factors
            available_factors = [col for col in factor_cols if col in indicators.columns]
            if len(available_factors) < 2:
                print(f"{current_date_str}: Not enough financial factors, skipping selection.")
                selected = []
            else:
                for col in available_factors:
                    indicators[factor_map[col]] = (indicators[col] - indicators[col].mean()) / indicators[col].std()
                indicators['score'] = indicators[[factor_map[col] for col in available_factors]].sum(axis=1)
                selected = indicators.sort_values('score', ascending=False).head(stock_num)[code_col].tolist()
        except Exception as e:
            print(f"{current_date_str}: 财务数据获取失败，原因：{e}")
            selected = []
    
        # 计算未来一个月收益
        rets = []
        for code in selected:
            try:
                price_df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                              start_date=current_date_str, 
                                              end_date=(current_date + pd.Timedelta(days=30)).strftime('%Y%m%d'), 
                                              adjust="qfq")
                if price_df is not None and len(price_df) > 1:
                    price_df = price_df.sort_values('日期')
                    pct = price_df.iloc[-1]['收盘'] / price_df.iloc[0]['收盘'] - 1
                    rets.append(pct)
            except Exception as e:
                print(f"获取 {code} 价格数据失败: {e}")
                continue
    
        pct = np.mean(rets) if rets else 0
        print(f"{current_date_str}: 选出 {len(selected)} 只股票, 平均收益率: {pct:.2%}")
    
        value *= (1 + pct)
        current_max_value = max(value, current_max_value)
        drawdown = 1 - value / current_max_value
        if drawdown >= max_drawdown:
            triggered = True
    
        net_value_history.append(value)
        max_value_history.append(current_max_value)
    
    # 创建结果 DataFrame
    if rebalance_dates.empty:
        print("错误：没有生成有效的 rebalance_dates，无法创建结果。")
    elif len(rebalance_dates) != len(net_value_history):
        print(f"错误：日期列表 ({len(rebalance_dates)}) 和净值历史 ({len(net_value_history)}) 长度不匹配。")
    else:
        results = pd.DataFrame({
            'net_value': net_value_history,
            'max_value': max_value_history
        }, index=pd.to_datetime(rebalance_dates))
    
        # 计算标准回撤序列
        results['drawdown'] = 1 - results['net_value'] / results['max_value']
    
        # 绘图
        plt.figure(figsize=(12, 5))
        results['net_value'].plot(title="净值曲线 (多因子动态选股)")
        plt.ylabel("Net Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("net_value_multifactor.png")
        print("多因子动态选股净值曲线图已保存为 net_value_multifactor.png")
    
        plt.figure(figsize=(12, 5))
        results['drawdown'].plot(title="回撤曲线 (多因子动态选股)")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("drawdown_multifactor.png")
        print("多因子动态选股回撤曲线图已保存为 drawdown_multifactor.png")
    
        print("\n最终净值:", results['net_value'].iloc[-1] if not results.empty else "N/A")
        print("最大回撤:", results['drawdown'].max() if not results.empty else "N/A")
except Exception as e:
    print(f"程序执行过程中出错: {e}")
