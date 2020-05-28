
import argparse
import requests
import warnings
import os

import bs4 as bs
import datetime as dt
import pandas_datareader.data as pdr

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, LSTM
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.volatility import *

# from analysis import build_ticker_result

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def calc_macd(close, fast=12, slow=26, signal=9):
    ma_fast = close.ewm(span=fast, min_periods=fast).mean()
    ma_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd = ma_fast - ma_slow
    ma_signal = macd.ewm(span=signal, min_periods=signal).mean()
    return macd, ma_signal


def calc_rsi(close, interval=14):
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
    return 100 - (100 / (1 + rs))


def calc_ibr(close, low, high):
    return (close - low) / (high - low)


def calc_william_r(close, low, high, interval=14):
    """
    Momentum indicator
    Using TA Library
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    """
    return wr(high, low, close, interval, fillna=True)


def calc_mfi(close, low, high, vol, interval=14):
    """
    Momentum type indicator
    """
    return money_flow_index(high, low, close, vol, n=interval, fillna=True)


def calculate_roc(series, _):
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100


def calc_roc(close, interval=14):
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


def calc_cmf(close, low, high, vol, interval=21):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    return chaikin_money_flow(high, low, close, vol, interval, fillna=True)


def calculate_cmo(series, _):
    # num_gains = (series >= 0).sum()
    # num_losses = (series < 0).sum()
    sum_gains = series[series >= 0].sum()
    sum_losses = np.abs(series[series < 0].sum())
    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    return np.round(cmo, 3)


def calc_cmo(close, interval=20):
    """
    Change Momentum Oscillator
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


def calc_sma(close, interval=9):
    """
    Momentum indicator
    """
    return close.rolling(interval).mean()


def calc_ema(close, interval=9):
    """
    Momentum indicator
    """
    return close.ewm(span=interval, min_periods=interval-1).mean()


def calc_wavg(rolling_prices, period):
    weights = pd.Series(range(1, period + 1))
    return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()


def calc_wma(close, interval=9):
    """
    Momentum indicator
    """
    return close.rolling(interval).apply(calc_wavg, args=(interval,), raw=False)


def calc_trix(close, interval=15):
    """
    Shows the percent rate of change of a triple exponentially smoothed moving average.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    return trix(close, interval, fillna=True)


def calc_cci(close, low, high, interval=20):
    """
    Commodity Channel Index (CCI)
    CCI measures the difference between a security’s price change and its average
    price change. High positive readings indicate that prices are well above their
    average, which is a show of strength. Low negative readings indicate that
    prices are well below their average, which is a show of weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """
    return cci(high, low, close, interval, fillna=True)


def calc_dpo(close, interval=20):
    """
    Defended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    """
    return dpo(close, n=interval)


def calc_kst(close, interval=20):
    """
    KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst  """
    return kst(close, interval)


def calc_dmi(close, low, high, interval=14):
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


def calc_bb_mav(close, interval=20):
    """
    Bollinger Bands (BB)
    N-period simple moving average (MA).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    """
    return bollinger_mavg(close, n=interval, fillna=True)


def calc_fi(close, vol, interval=13):
    """
    Force Index (FI)
    It illustrates how strong the actual buying or selling pressure is.
    High positive values mean there is a strong rising trend, and low
    values signify a strong downward trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    """
    return force_index(close, vol, n=interval, fillna=True)


def calc_eom(low, high, vol, interval=14):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    dm = (high + low) / 2 - (high.shift(1) - low.shift(1)) / 2
    br = vol / (high - low)
    evm = dm / br
    return evm.rolling(interval).mean()


def add_technical_indicators(input_data):

    close = input_data['close']
    high = input_data['high']
    low = input_data['low']
    vol = input_data['volume']

    input_data['macd'], input_data['signal'] = calc_macd(close)
    input_data['rsi'] = calc_rsi(close=close)
    input_data['ibr'] = calc_ibr(close=close, low=low, high=high)
    input_data['willamr'] = calc_william_r(close=close, low=low, high=high)
    input_data['mfi'] = calc_mfi(close=close, low=low, high=high, vol=vol)
    input_data['roc_12'] = calc_roc(close=close, interval=12)
    input_data['roc_25'] = calc_roc(close=close, interval=25)
    input_data['cmf'] = calc_cmf(close=close, low=low, high=high, vol=vol)
    input_data['cmo'] = calc_cmo(close=close)
    input_data['sma'] = calc_sma(close=close)
    input_data['ema'] = calc_ema(close=close)
    input_data['wma'] = calc_wma(close=close)
    input_data['trix'] = calc_trix(close=close)
    input_data['cci'] = calc_cci(close=close, low=low, high=high)
    input_data['dpo'] = calc_dpo(close=close)
    input_data['kst'] = calc_kst(close=close)
    input_data['dmi'] = calc_dmi(close=close, low=low, high=high)
    input_data['bb'] = calc_bb_mav(close=close)
    input_data['fi'] = calc_fi(close=close, vol=vol)
    input_data['eom'] = calc_eom(low=low, high=high, vol=vol)
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
    for i in range(training_days, len(inputs)-1):
        x_data.append(inputs[i - training_days:i, :])
        y_data.append(np.concatenate((inputs[i, 0:5], inputs[i+1, 0:5])))
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], inputs.shape[1]))
    return x_data, y_data


def build_predict_data(inputs, training_days):
    x_data = []
    x_data.append(inputs[-training_days:, :])
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], inputs.shape[1]))
    return x_data


def build_model(x_train, y_train, verbose=False):
    """ create and fit the LSTM network """
    units = 200
    model = Sequential()
    model.add(CuDNNLSTM(units=units,
                        return_sequences=True,
                        input_shape=(x_train.shape[1], x_train.shape[2])))

    model.add(CuDNNLSTM(units=units))
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
    price_columns = ['high', 'low', 'open', 'close']
    input_df = input_data.reset_index(drop=True)
    input_values = input_df.values.reshape(-1, input_df.shape[1])

    price_data = input_data[price_columns]
    min_price_value = np.min(price_data.min().values)
    max_price_value = np.max(price_data.max().values)
    price_data = (price_data - min_price_value) / (max_price_value - min_price_value)
    scaled_price = price_data.values.reshape(-1, len(price_columns))

    indicator_data = input_df.drop(columns=price_columns)
    indicator_values = indicator_data.values.reshape(-1, indicator_data.shape[1])
    input_scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_indicator = input_scalar.fit_transform(indicator_values)
    scaled_data = np.concatenate((scaled_price, scaled_indicator), axis=1)

    train_size = len(input_data) - test_days

    # converting data set into x_train and y_train
    x_train, y_train = build_train_test_data(scaled_data[:train_size], training_days)
    x_test, _ = build_train_test_data(scaled_data[train_size - training_days:], training_days)
    x_predict = build_predict_data(scaled_data, training_days)

    # build model with train data
    model = build_model(x_train, y_train)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    result_data = model.predict(x_test)
    result_predict = model.predict(x_predict)

    np.savetxt(data_folder + "/" + ticker + "_market.out", input_values[-test_days*2:], delimiter=",")
    np.savetxt(data_folder + "/" + ticker + "_input.out", scaled_data[-test_days*2:], delimiter=",")
    np.savetxt(data_folder + "/" + ticker + "_result.out", result_data, delimiter=",")
    np.savetxt(data_folder + "/" + ticker + "_predict.out", result_predict, delimiter=",")

    return


def train(start, end):
    print("Start [{0:3d}] -> End [{1:3d}]".format(start, end))
    data_folder = 'results/'
    maybe_make_dir(data_folder)

    end_date = dt.date.today()
    data_folder += str(end_date)
    maybe_make_dir(data_folder)

    tickers = get_sp500_tickers()[start:end]
    start_date = end_date - dt.timedelta(365 * 20)

    for i, ticker in enumerate(tickers):
        start_time = dt.datetime.now()

        #try:
        data = pdr.get_data_yahoo(ticker, start_date, end_date)
        data.rename(columns={'Open': 'open',
                             'High': 'high',
                             'Low': 'low',
                             'Close': 'close',
                             'Volume': 'volume'},
                    inplace=True)
        data.drop(columns=['Adj Close'], inplace=True)
        if len(data) > 3000:
            data = add_technical_indicators(data)
            train_predict(ticker, data, data_folder, epochs=50, verbose=0)
            end_time = dt.datetime.now()
            print("[{0:3d}]:{1}\texec time:{2:6.3f}".
                  format(i + start, ticker.rjust(5, " "), (end_time - start_time).total_seconds()))
        else:
            print("[{0:3d}]:{1}".format(i + start, ticker.rjust(5, " ")))
        #except:
        #    print("Error: [{0:3d}]:{1}".format(i + start, ticker.rjust(5, " ")))
        #    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='start index of tickers in S&P List')
    parser.add_argument('-e', '--end', type=int, default=50,
                        help='last index of tickers in S&P List')
    parser.add_argument('-m', '--mode', type=str, default='a',
                        help='Execution mode, t as train, a as analysis')
    parser.add_argument('-t', '--ticker', type=str, default='MMM',
                        help='Ticker which is analysed')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    # if args.mode == 'a':
    #    build_ticker_result(args.ticker, 'results/2020-05-28/')
    #else:
    train(args.start, args.end)
