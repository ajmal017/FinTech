
import argparse
import requests
import quandl
import warnings
import os

import pandas as pd
import numpy as np
import bs4 as bs
import tensorflow as tf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.others import *
from ta.volatility import *


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def macd(close, fast=12, slow=26, signal=9):
    ma_fast = close.ewm(span=fast, min_periods=fast).mean()
    ma_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_ = ma_fast - ma_slow
    ma_signal = macd_.ewm(span=signal, min_periods=signal).mean()
    return (macd_, ma_signal)

def RSI(close, interval=14):
    """
    Momentum indicator
    As per https://www.investopedia.com/terms/r/rsi.asp
    RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
    RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )
    E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
    http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Verified!
    """
    delta = close - close.shift(1)
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta
    avg_gain = gain.rolling(interval).sum() / interval
    avg_loss = loss.rolling(interval).sum() / interval
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def IBR(close, low, high):
    return (close - low) / (high - low)

def WilliamR(close, low, high, interval=14):
    """
    Momentum indicator
    Using TA Libraray
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    """
    return wr(high, low, close, interval, fillna=True)

def MFI(close, low, high, vol, interval=14):
    """
    Momentum type indicator
    """
    return money_flow_index(high, low, close, vol, n=interval, fillna=True)

def calculate_roc(series, period):
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

def ROC(close, interval=14):
    """
    Momentum oscillator
    As per implement https://www.investopedia.com/terms/p/pricerateofchange.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    ROC = (close_price_n - close_price_(n-1) )/close_price_(n-1) * 100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """
    # for 12 day period, 13th day price - 1st day price
    return close.rolling(interval + 1).apply(calculate_roc, args=(interval,), raw=False)

def CMF(close, low, high, vol, interval=21):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    return chaikin_money_flow(high, low, close, vol, interval, fillna=True)

def calculate_cmo(series, period):
    # num_gains = (series >= 0).sum()
    # num_losses = (series < 0).sum()
    sum_gains = series[series >= 0].sum()
    sum_losses = np.abs(series[series < 0].sum())
    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    return np.round(cmo, 3)

def CMO(close, interval=20):
    """
    Chande Momentum Oscillator
    As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
    CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
    range = +100 to -100
    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated
    return: None (adds the result in a column)
    """
    diff = close.diff()[1:]  # skip na
    return diff.rolling(interval).apply(calculate_cmo, args=(interval,), raw=False)

def SMA(close, interval=9):
    """
    Momentum indicator
    """
    return close.rolling(interval).mean()

def EMA(close, interval=9):
    """
    Momentum indicator
    """
    return close.ewm(span=interval, min_periods=interval-1).mean()

def wavg(rolling_prices, period):
    weights = pd.Series(range(1, period + 1))
    return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

def WMA(close, interval=9, hma_step=0):
    """
    Momentum indicator
    """
    return close.rolling(interval).apply(wavg, args=(interval,), raw=False)

def TRIX(close, interval=15):
    """
    Shows the percent rate of change of a triple exponentially smoothed moving average.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    return trix(close, interval, fillna=True)

def CCI(close, low, high, interval=20):
    """
    Commodity Channel Index (CCI)
    CCI measures the difference between a security’s price change and its average
    price change. High positive readings indicate that prices are well above their
    average, which is a show of strength. Low negative readings indicate that
    prices are well below their average, which is a show of weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """
    return cci(high, low, close, interval, fillna=True)

def DPO(close, interval=20):
    """
    Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    """
    return dpo(close, n=interval)

def KST(close, interval=20):
    """
    KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst  """
    return kst(close, interval)

def DMI(close, low, high, interval=14):
    """
    Average Directional Movement Index (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI) are
    derived from smoothed averages of these differences, and measure trend direction
    over time. These two indicators are often referred to collectively as the
    Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    """
    return adx(high, low, close, n=interval, fillna=True)

def BB_MAV(close, interval=20):
    """
    Bollinger Bands (BB)
    N-period simple moving average (MA).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    """
    return bollinger_mavg(close, n=interval, fillna=True)

def FI(close, vol, interval=13):
    """
    Force Index (FI)
    It illustrates how strong the actual buying or selling pressure is.
    High positive values mean there is a strong rising trend, and low
    values signify a strong downward trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    """
    return force_index(close, vol, n=interval, fillna=True)

def EOM(low, high, vol, interval=14):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    dm = (high + low) / 2 - (high.shift(1) - low.shift(1)) / 2
    br = vol / (high - low)
    evm = dm / br
    return evm.rolling(interval).mean()

def add_techincal_indicators(input_data):

    close = input_data['close']
    high = input_data['high']
    low = input_data['low']
    vol = input_data['volume']

    input_data['macd'], input_data['signal'] = macd(close)
    input_data['rsi'] = RSI(close=close)
    input_data['ibr'] = IBR(close=close, low=low, high=high)
    input_data['willamr'] = WilliamR(close=close, low=low, high=high)
    input_data['mfi'] = MFI(close=close, low=low, high=high, vol=vol)
    input_data['roc_12'] = ROC(close=close, interval=12)
    input_data['roc_25'] = ROC(close=close, interval=25)
    input_data['cmf'] = CMF(close=close, low=low, high=high, vol=vol)
    input_data['cmo'] = CMO(close=close)
    input_data['sma'] = SMA(close=close)
    input_data['ema'] = EMA(close=close)
    input_data['wma'] = WMA(close=close)
    input_data['trix'] = TRIX(close=close)
    input_data['cci'] = CCI(close=close, low=low, high=high)
    input_data['dpo'] = DPO(close=close)
    input_data['kst'] = KST(close=close)
    input_data['dmi'] = DMI(close=close, low=low, high=high)
    input_data['bb'] = BB_MAV(close=close)
    input_data['fi'] = FI(close=close, vol=vol)
    input_data['eom'] = EOM(low=low, high=high, vol=vol)
    input_data.dropna(inplace=True)
    return input_data

def get_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.rstrip('\r\n'))
    return tickers

def build_train_test_data(inputs, training_days):
    x_data, y_data = [], []
    for i in range(training_days, len(inputs)):
        x_data.append(inputs[i - training_days:i, :])
        y_data.append(inputs[i, 0:5])
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], inputs.shape[1]))
    return (x_data, y_data)

def build_predict_data(inputs, training_days):
    x_data = []
    x_data.append(inputs[-training_days:, :])
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], inputs.shape[1]))
    return x_data

def build_model(x_train, y_train, verbose=False):
    """ create and fit the LSTM network """
    uints = 200
    model = Sequential()
    model.add(CuDNNLSTM(units=uints,
                    return_sequences=True,
                    input_shape=(x_train.shape[1], x_train.shape[2])))

    model.add(CuDNNLSTM(units=uints))
    model.add(Dense(y_train.shape[1]))


    model.compile(loss='mean_squared_error', optimizer='adam')
    if verbose:
        model.summary()

    return model

def train_predict(ticker,
                  input_data,
                  data_folder,
                  training_days=60,
                  test_days=252,
                  epochs=50,
                  batch_size=32,
                  verbose=0):
    """
    Build a train and test data set from input data
    Build a model
    Train a model
    Calculate the accuracy
    Predict tomorrow trend
    """
    start = dt.datetime.now()
    #display(input_data)
    input_values = input_data.values.reshape(-1, input_data.shape[1])
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = input_scaler.fit_transform(input_values)

    train_size = len(input_data) - test_days

    #converting dataset into x_train and y_train
    x_train, y_train = build_train_test_data(scaled_data[:train_size], training_days)
    X_test, _ = build_train_test_data(scaled_data[train_size - training_days:], training_days)
    X_predict = build_predict_data(scaled_data, training_days)

    # build model with train data
    model = build_model(x_train, y_train)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    result_data = model.predict(X_test)
    result_pred = model.predict(X_predict)

    np.savetxt(data_folder + "/" + ticker + "_input.out", scaled_data, delimiter=",")
    np.savetxt(data_folder + "/" + ticker + "_result.out", result_data, delimiter=",")
    np.savetxt(data_folder + "/" + ticker + "_predict.out", result_pred, delimiter=",")

    end = dt.datetime.now()
    print("\ttrain predict   exec time:{0:6.3f}".format((end - start).total_seconds()))

    return

def main(start, end):
    print ("Start [{0.3d}] -> End [{1:3d}]".format(start, end))
    data_folder = '/content/drive/My Drive/FintechData/results/'
    maybe_make_dir(data_folder)

    end_date = dt.date.today()
    data_folder += str(end_date)
    maybe_make_dir(data_folder)

    data_folder += "/" + str(start) + "-" + str(end)
    maybe_make_dir(data_folder)

    tickers = get_sp500_tickers()[start:end]
    start_date = end_date - dt.timedelta(365 * 20)

    for i, ticker in enumerate(tickers):
        start = dt.datetime.now()
        print ("[{0:3d}]:{1}".format(i, ticker))

        data = quandl.get_table('WIKI/PRICES',
                                qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'] },
                                ticker = ticker,
                                date = { 'gte': str(start_date), 'lte': str(end_date) },
                                api_key='iaSAm2syxYNWCMMENDkJ')
        data.rename(columns={'adj_open':'open',
                            'adj_high':'high',
                            'adj_low':'low',
                            'adj_close':'close',
                            'adj_volume':'volume'},
                    inplace=True)
        data = data.iloc[::-1]
        if len(data) > 3000:
            data = add_techincal_indicators(data)
            end = dt.datetime.now()
            print("\tdata extraction exec time:{0:6.3f}".format((end - start).total_seconds()))
            train_predict(ticker, data.drop(columns=['ticker', 'date']), data_folder, epochs=50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='start index of tickers in S&P List')
    parser.add_argument('-e', '--end', type=int, default=50,
                        help='last index of tickers in S&P List')

    args = parser.parse_args()
    main(args.start, args.end)