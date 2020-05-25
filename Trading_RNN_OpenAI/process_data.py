import talib
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr


def get_data(ticker='AAPL', span=15, test_size_percentage=20):
    end_date = dt.date(2019, 12, 29)
    start_date = end_date - dt.timedelta(365 * span)
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    test_size = int(test_size_percentage/100 * data.shape[0])
    train_size = data.shape[0] - 252
    data.columns = map(str.lower, data.columns)
    return data[:train_size], data[train_size:]

def get_data_from_file(ticker='AAPL', interval='5min'):
    data = pd.read_csv("~/FintechData/" + ticker + "_" + interval + ".txt", date_parser=True)
    data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
    train_data = data[data['date'] < '2019-01-01'][::6]
    end_date = dt.date(2020, 1, 1)
    start_date = dt.date(2019, 1, 1)
    test_data = pdr.get_data_yahoo(ticker, start_date, end_date)
    test_data.columns = map(str.lower, test_data.columns)
    return train_data, test_data

class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.open = df['open'].astype('float')
        self.close = df['close'].astype('float')
        self.high = df['high'].astype('float')
        self.low = df['low'].astype('float')
        self.volume = df['volume'].astype('float')

    def add_bar_features(self):
        # stationary candle
        self.df['bar_hc'] = self.high - self.close
        self.df['bar_ho'] = self.high - self.open
        self.df['bar_hl'] = self.high - self.low
        self.df['bar_cl'] = self.close - self.low
        self.df['bar_ol'] = self.open - self.low
        self.df['bar_co'] = self.close - self.open
        self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1)
        return self.df

    def add_mv_avg_features(self):
        self.df['sma5'] = talib.SMA(self.close, 5)
        self.df['sma20'] = talib.SMA(self.close, 20)
        self.df['sma120'] = talib.SMA(self.close, 120)
        self.df['ema12'] = talib.SMA(self.close, 5)
        self.df['ema26'] = talib.SMA(self.close, 26)
        return self.df

    def add_adj_features(self):
        self.df['adj_open'] = self.df['open'] / self.close
        self.df['adj_high'] = self.df['high'] / self.close
        self.df['adj_low'] = self.df['low'] / self.close
        self.df['adj_close'] = self.df['close'] / self.close
        return self.df

    # note! this is not a complete list
    # additional indicator can help in some scenario but usually acts as a noise
    def add_ta_features(self):
        obv = talib.OBV(self.close, self.volume)
        obv_mv_avg = talib.MA(obv, timeperiod=10)
        obv_mv_avg[np.isnan(obv_mv_avg)] = obv[np.isnan(obv_mv_avg)]
        difference = obv - obv_mv_avg

        self.df['obv'] = obv
        self.df['obv_signal'] = difference
        self.df['obv_cheat'] = np.gradient(difference)

        upper, middle, lower = talib.BBANDS(self.close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        self.df['dn'] = lower
        self.df['mavg'] = middle
        self.df['up'] = upper
        self.df['pctB'] = (self.close - self.df.dn) / (self.df.up - self.df.dn)
        rsi14 = talib.RSI(self.close, 14)
        self.df['rsi14'] = rsi14

        macd, macdsignal, macdhist = talib.MACD(self.close, 12, 26, 9)
        self.df['macd'] = macd
        self.df['signal'] = macdsignal

        # addtional info
        self.df['adx'] = talib.ADX(self.high, self.low, self.close, timeperiod=14)
        self.df['cci'] = talib.CCI(self.high, self.low, self.close, timeperiod=14)

        # maximum profit
        self.df['plus_di'] = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=14)

        # lower_bound
        self.df['lower_bound'] = self.df['open'] - self.df['low'] + 1

        # ATR
        self.df['atr'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)

        # STOCH momentum
        self.df = ta.stochastic_oscillator_k(self.df)
        self.df = ta.stochastic_oscillator_d(self.df, n=10)

        # TRIX
        self.df['trix'] = talib.TRIX(self.close, timeperiod=5)
        self.df['trix_signal'] = ta.moving_average(self.df['trix'], n=3)
        self.df['trix_hist'] = self.df['trix'] - self.df['trix_signal']

        # MFI
        self.df['mfi14'] = money_flow_index(self.df, 14)
