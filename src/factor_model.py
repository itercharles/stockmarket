import pandas as pd

class MultiFactorModel:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher

    def compute_factors(self, df):
        df = df.copy()
        df['pe_rank'] = df['pe_ttm'].rank(ascending=True)
        df['pb_rank'] = df['pb'].rank(ascending=True)
        df['roe_rank'] = df['roe'].rank(ascending=False)
        df['momentum_rank'] = df['momentum'].rank(ascending=False)

        df['score'] = (
            df['pe_rank'] * 0.25 +
            df['pb_rank'] * 0.25 +
            df['roe_rank'] * 0.25 +
            df['momentum_rank'] * 0.25
        )
        return df.sort_values('score').head(10)

    def select_stocks(self, date):
        df = self.data_fetcher(date)
        selected = self.compute_factors(df)
        return selected['stock_code'].tolist()