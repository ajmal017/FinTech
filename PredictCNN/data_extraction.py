import os
import datetime as dt
import pandas_datareader.data as pdr
from indicators import calculate_technical_indicators
from os import path
import pandas as pd


def get_data_file(ticker, data_folder):
    return data_folder + "/" + ticker + ".csv"


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_data(ticker, total_dates, intervals, data_folder="data"):
    maybe_make_dir(data_folder)
    data_file = get_data_file(ticker, data_folder)
    if path.exists(data_file):
        data = pd.read_csv(data_file)
        print(data)
    else:
        data = pdr.get_data_yahoo(ticker, dt.date.today() - dt.timedelta(total_dates), dt.date.today())
        calculate_technical_indicators(data, intervals)
        data.dropna(inplace=True)
        data.to_csv(data_file, header=True)
    return data
