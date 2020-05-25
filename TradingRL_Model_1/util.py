import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime as dt
import pandas_datareader.data as pdr


def get_data(stock_intervals=10, col='Adj Close'):
    """
    Returns a 3 x n_step array
    """
    end_date = dt.date(2019, 12, 29)
    start_date = end_date - dt.timedelta(365 * stock_intervals)
    msft = pdr.get_data_yahoo('MSFT', start_date, end_date)
    ibm = pdr.get_data_yahoo('IBM', start_date, end_date)
    qcom = pdr.get_data_yahoo('QCOM', start_date, end_date)

    #return np.array([msft[col].values,
    #                 ibm[col].values,
    #                 qcom[col].values])
    return np.array([msft[col].values])


def get_scalar(env):
    """
    Takes a env and returns a scaler for its observation space
    """
    low = [0] * (env.n_stock * 2 + 1)

    high = []
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    # 3 is a magic number...
    max_cash = env.init_invest * 3
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)

    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
