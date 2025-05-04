import akshare as ak
import pandas as pd
from datetime import timedelta
import time

def get_factor_data(date_str):
    date = pd.to_datetime(date_str)
    start = date - timedelta(days=30)
    end = date
    date_range = pd.date_range(start, end, freq='B')

    # 获取沪深300成分股
    hs300 = ak.index_stock_cons(symbol="000300")
    stock_list = hs300[['品种代码']].rename(columns={'品种代码': 'stock_code'})
    stock_list['stock_code'] = stock_list['stock_code'].apply(lambda x: x.zfill(6))
    stock_list = stock_list.head(10)  # 只取前10只股票进行测试

    results = []
    for _, row in stock_list.iterrows():
        code = row['stock_code']
        try:
            indi_df = ak.stock_a_indicator_lg(symbol=code)
            if indi_df.empty:
                continue
            latest_row = indi_df.sort_values(by='trade_date').iloc[-1]
            pe_ttm = latest_row['pe_ttm']
            pb = latest_row['pb']
            roe = latest_row.get('roe') or 0  # 没有ROE则为0

            # 获取动量（30日涨跌幅）
            suffix = 'sh' if code.startswith('6') else 'sz'
            full_code = suffix + code
            hist_df = ak.stock_zh_a_hist(symbol=full_code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
            print(hist_df)
            if hist_df is None or hist_df.empty or '日期' not in hist_df.columns:
                print(f"Skip {code}: no valid price data")
                continue
            hist_df['date'] = pd.to_datetime(hist_df['日期'])
            hist_df.set_index('date', inplace=True)
            hist_df = hist_df.reindex(date_range).ffill()
            momentum = hist_df['收盘'][-1] / hist_df['收盘'][0] - 1

            results.append({
                'stock_code': full_code,
                'pe_ttm': pe_ttm,
                'pb': pb,
                'roe': roe,
                'momentum': momentum
            })

            time.sleep(0.2)  # 限速，防止请求过快

        except Exception as e:
            print(f"跳过 {code}: {e}")
            continue

    return pd.DataFrame(results)