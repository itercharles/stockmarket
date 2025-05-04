import akshare as ak
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import matplotlib as mpl
import os

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # For macOS
mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略 AKShare 可能产生的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 参数设置
benchmark_code = '000300' # AKShare 通常不需要 .SH 后缀
stock_num = 10
max_drawdown = 0.10 # 最大回撤容忍度 (基于峰值计算)
start_date = '20230101'
end_date = '20240101'
start_date_dt = datetime.strptime(start_date, '%Y%m%d')
end_date_dt = datetime.strptime(end_date, '%Y%m%d')

# 获取指数行情作为基准 (使用 AKShare)
def get_benchmark_data(benchmark_code, start_date, end_date):
    cache_file = f"cache_{benchmark_code}_{start_date}_{end_date}.csv"
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
    try:
        if os.path.exists(cache_file):
            print(f"Using cached benchmark data from {cache_file}")
            bench_df = pd.read_csv(cache_file)
        else:
            bench_df = retry_function(lambda: ak.index_zh_a_hist(symbol=benchmark_code, period="daily", 
                                                                 start_date=start_date, end_date=end_date))
            if bench_df is None or bench_df.empty:
                raise ValueError("AKShare 指数行情数据获取失败或为空")
            bench_df.rename(columns={'日期': 'trade_date', '收盘': 'close'}, inplace=True)
            bench_df.to_csv(cache_file, index=False)
        bench_df.rename(columns={'日期': 'trade_date', '收盘': 'close'}, inplace=True)
        bench_df['date'] = pd.to_datetime(bench_df['trade_date'])
        bench_df.set_index('date', inplace=True)
        return bench_df.sort_index()
    except Exception as e:
        raise ValueError(f"获取指数行情数据时出错: {e}")

def get_all_financial_data(symbol, start_year, cache_file):
    if os.path.exists(cache_file):
        print("加载本地缓存的财务数据...")
        return pd.read_pickle(cache_file)
    print("批量抓取财务数据...")
    all_data = []
    cons_df = ak.index_stock_cons(symbol=symbol)
    codes = cons_df['品种代码'].tolist()
    for code in codes:
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
            if not df.empty:
                df['股票代码'] = code
                all_data.append(df)
        except Exception as e:
            print(f"获取{code}财务数据失败: {e}")
    all_financial_data = pd.concat(all_data, ignore_index=True)
    all_financial_data.to_pickle(cache_file)
    print("财务数据已缓存到本地。")
    return all_financial_data

def select_stocks(current_date, codes, all_financial_data, factor_map, factor_cols, stock_num):
    # 只保留目标股票的最新财报
    df = all_financial_data[all_financial_data['股票代码'].isin(codes)].copy()
    if df.empty or '披露日期' not in df.columns:
        return []
    df['披露日期'] = pd.to_datetime(df['披露日期'], errors='coerce')
    df = df[df['披露日期'] <= current_date]
    if df.empty:
        return []
    # 按股票分组，取每只股票最新一条
    df = df.sort_values(['股票代码', '披露日期']).drop_duplicates('股票代码', keep='last')
    available_factors = [col for col in factor_cols if col in df.columns]
    if len(available_factors) < 2:
        print(f"{current_date.strftime('%Y%m%d')}: Not enough financial factors, skipping selection.")
        return []
    for col in available_factors:
        df[factor_map[col]] = (df[col] - df[col].mean()) / df[col].std()
    df['score'] = df[[factor_map[col] for col in available_factors]].sum(axis=1)
    return df.sort_values('score', ascending=False).head(stock_num)['股票代码'].tolist()

def calc_portfolio_return(selected, current_date, days=30):
    rets = []
    for code in selected:
        try:
            price_df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                          start_date=current_date.strftime('%Y%m%d'), 
                                          end_date=(current_date + pd.Timedelta(days=days)).strftime('%Y%m%d'), 
                                          adjust="qfq")
            if price_df is not None and len(price_df) > 1:
                price_df = price_df.sort_values('日期')
                pct = price_df.iloc[-1]['收盘'] / price_df.iloc[0]['收盘'] - 1
                rets.append(pct)
        except Exception as e:
            print(f"获取 {code} 价格数据失败: {e}")
    return np.mean(rets) if rets else 0

def main():
    bench_df = get_benchmark_data(benchmark_code, start_date, end_date)
    factor_map = {
        '加权净资产收益率(%)': '加权净资产收益率(%)',
        '净利润增长率(%)': '净利润增长率(%)',
        '主营业务收入增长率(%)': '主营业务收入增长率(%)'
        
    }
    factor_cols = list(factor_map.keys())
    rebalance_dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='M')
    net_value_history = []
    max_value_history = []
    value = 1.
    current_max_value = 1.0
    triggered = False
    financial_cache_file = "financial_cache_000016_2019.pkl"
    all_financial_data = get_all_financial_data("000016", "2019", financial_cache_file)
    cons_df = ak.index_stock_cons(symbol="000016")
    codes = cons_df['品种代码'].tolist()
    for current_date in rebalance_dates:
        selected = select_stocks(current_date, codes, all_financial_data, factor_map, factor_cols, stock_num)
        pct = calc_portfolio_return(selected, current_date)
        print(f"{current_date.strftime('%Y%m%d')}: 选出 {len(selected)} 只股票, 平均收益率: {pct:.2%}")
        value *= (1 + pct)
        current_max_value = max(value, current_max_value)
        drawdown = 1 - value / current_max_value
        if drawdown >= max_drawdown:
            triggered = True
        net_value_history.append(value)
        max_value_history.append(current_max_value)
    if rebalance_dates.empty:
        print("错误：没有生成有效的 rebalance_dates，无法创建结果。")
    elif len(rebalance_dates) != len(net_value_history):
        print(f"错误：日期列表 ({len(rebalance_dates)}) 和净值历史 ({len(net_value_history)}) 长度不匹配。")
    else:
        results = pd.DataFrame({
            'net_value': net_value_history,
            'max_value': max_value_history
        }, index=pd.to_datetime(rebalance_dates))
        results['drawdown'] = 1 - results['net_value'] / results['max_value']
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行过程中出错: {e}")