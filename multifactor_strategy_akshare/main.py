from factor_model import MultiFactorModel
from backtest_engine import BacktestEngine
from ak_data_fetcher import get_factor_data
from ak_price_loader import get_price_data

model = MultiFactorModel(get_factor_data)
engine = BacktestEngine(model, get_price_data)

result = engine.run_backtest('2023-01-01', '2023-12-31')
result.plot(title='多因子选股策略（AKShare数据）回测收益')