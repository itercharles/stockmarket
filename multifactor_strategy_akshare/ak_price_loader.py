import akshare as ak
import pandas as pd

def get_price_data(stocks, start, end):
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    price_dict = {}

    for stock in stocks:
        try:
            df = ak.stock_zh_a_hist(symbol=stock, start_date=start_str, end_date=end_str)
            df['date'] = pd.to_datetime(df['日期'])
            df.set_index('date', inplace=True)
            price_dict[stock] = df['收盘']
        except:
            continue

    return pd.DataFrame(price_dict)