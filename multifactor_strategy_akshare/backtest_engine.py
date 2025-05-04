import pandas as pd

class BacktestEngine:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def run_backtest(self, start_date, end_date, frequency='M'):
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        all_returns = []

        while current_date <= end_date:
            stocks = self.model.select_stocks(current_date.strftime('%Y-%m-%d'))
            prices = self.data_loader(stocks, current_date, current_date + pd.DateOffset(months=1))

            if not prices.empty:
                returns = prices.pct_change().mean(axis=1)
                all_returns.append(returns)

            current_date += pd.DateOffset(months=1)

        result = pd.concat(all_returns).sort_index()
        result = (1 + result).cumprod()
        return result