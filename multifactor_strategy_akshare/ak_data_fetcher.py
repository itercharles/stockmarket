import akshare as ak
import pandas as pd
from datetime import timedelta

def get_factor_data(date_str):
    date = pd.to_datetime(date_str)
    stock_df = ak.stock_a_lg_indicator_em()
    start = date - timedelta(days=30)
    end = date
    date_range = pd.date_range(start, end, freq='B')

    def get_momentum(code):
        try:
            df = ak.stock_zh_a_hist(symbol=code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
            df['date'] = pd.to_datetime(df['日期'])
            df.set_index('date', inplace=True)
            df = df.reindex(date_range).ffill()
            return df['收盘'][-1] / df['收盘'][0] - 1
        except:
            return None

    stock_df = stock_df.rename(columns={
        '代码': 'stock_code',
        '市盈率': 'pe_ttm',
        '市净率': 'pb',
        '净资产收益率': 'roe'
    })

    stock_df['stock_code'] = stock_df['stock_code'].apply(lambda x: ('sh' if x.startswith('6') else 'sz') + x)
    stock_df['momentum'] = stock_df['stock_code'].apply(get_momentum)

    return stock_df.dropna(subset=['pe_ttm', 'pb', 'roe', 'momentum'])[['stock_code', 'pe_ttm', 'pb', 'roe', 'momentum']]