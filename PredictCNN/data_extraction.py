import os
import datetime as dt
import pandas_datareader.data as pdr
from indicators import calculate_technical_indicators


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_data(ticker, total_dates, intervals):
    maybe_make_dir("data")
    data = pdr.get_data_yahoo(ticker, dt.date.today() - dt.timedelta(total_dates), dt.date.today())
    calculate_technical_indicators(data, intervals)
    data.dropna(inplace=True)
    data.to_csv("data/" + ticker + ".csv", header=True)
    return data
