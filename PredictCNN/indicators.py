from tqdm import tqdm

from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.volatility import *


def calculate_rsi(data, intervals):
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
    # print("Calculating RSI")
    for i in tqdm(intervals):
        delta = data['Close'] - data['Close'].shift(1)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(i).sum() / i
        avg_loss = loss.rolling(i).sum() / i
        rs = avg_gain / avg_loss
        data['rsi_' + str(i)] = 100 - (100 / (1 + rs))


def calculate_ibr(data):
    return (data['Close'] - data['Low']) / (data['High'] - data['Low'])


def calculate_williamr(data, intervals):
    """
    Momentum indicator
    Using TA Libraray
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    """
    # print("Calculating WilliamR")
    for i in tqdm(intervals):
        data['wr_' + str(i)] = wr(data['High'], data['Low'], data['Close'], i, fillna=True)


def calculate_mfi(data, intervals):
    """
    Momentum type indicator
    """
    # print("Calculating MFI")
    for i in tqdm(intervals):
        data['mfi_' + str(i)] = money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'],
                                                 n=i,
                                                 fillna=True)


def calculate_roc_internal(series, _):
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100


def calculate_roc(data, intervals):
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
    # print("Calculating ROC")
    for period in tqdm(intervals):
        data['roc_' + str(period)] = np.nan
        # for 12 day period, 13th day price - 1st day price
        res = data['Close'].rolling(period + 1).apply(calculate_roc_internal, args=(period,), raw=False)
        data['roc_' + str(period)] = res


def calculate_cmf(data, intervals):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    # print("Calculating CMF")
    for i in tqdm(intervals):
        data['cmf_' + str(i)] = chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'],
                                                   i,
                                                   fillna=True)


def calculate_cmo_internal(series, _):
    # num_gains = (series >= 0).sum()
    # num_losses = (series < 0).sum()
    sum_gains = series[series >= 0].sum()
    sum_losses = np.abs(series[series < 0].sum())
    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    return np.round(cmo, 3)


def calculate_cmo(data, intervals):
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
    diff = data["Close"].diff()[1:]  # skip na
    # print("Calculating CMO")
    for period in tqdm(intervals):
        data['cmo_' + str(period)] = np.nan
        res = diff.rolling(period).apply(calculate_cmo_internal, args=(period,), raw=False)
        data['cmo_' + str(period)][1:] = res


def calculate_sma(data, intervals, col_name='Close'):
    """
    Momentum indicator
    """
    # print("Calculating SMA " + col_name)
    for i in tqdm(intervals):
        data[col_name + '_sma_' + str(i)] = data[col_name].rolling(i).mean()


def calculate_ema(data, intervals):
    """
    Momentum indicator
    """
    # print("Calculating EMA")
    for i in tqdm(intervals):
        data['ema_' + str(i)] = data['Close'].ewm(span=i, min_periods=i-1).mean()


def wavg(rolling_prices, period):
    weights = pd.Series(range(1, period + 1))
    return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()


def calculate_wma(data, intervals, hma_step=0):
    """
    Momentum indicator
    """
    temp_col_count_dict = {}
    for i in tqdm(intervals, disable=(hma_step != 0)):
        res = data['Close'].rolling(i).apply(wavg, args=(i,), raw=False)
        # print("interval {} has unique values {}".format(i, res.unique()))
        if hma_step == 0:
            data['wma_' + str(i)] = res
        elif hma_step == 1:
            if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
            else:
                temp_col_count_dict['hma_wma_' + str(i)] = 0
                # after halving the periods and rounding, there may be two intervals with same value e.g.
                # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
                data['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
        elif hma_step == 3:
            import re
            expr = r"^hma_[0-9]{1}"
            columns = list(data.columns)
            # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
            data['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res


def calculate_trix(data, intervals):
    """
    Shows the percent rate of change of a triple exponentially smoothed moving average.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    # print("Calculating TRIX")
    for i in tqdm(intervals):
        data['trix_' + str(i)] = trix(data['Close'], i, fillna=True)


def calculate_cci(data, intervals):
    """
    Commodity Channel Index (CCI)
    CCI measures the difference between a security’s price change and its average
    price change. High positive readings indicate that prices are well above their
    average, which is a show of strength. Low negative readings indicate that
    prices are well below their average, which is a show of weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """
    # print("Calculating CCI")
    for i in tqdm(intervals):
        data['cci_' + str(i)] = cci(data['High'], data['Low'], data['Close'], i, fillna=True)


def calculate_dpo(data, intervals):
    """
    Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    """
    # print("Calculating DPO")
    for i in tqdm(intervals):
        data['dpo_' + str(i)] = dpo(data['Close'], n=i)


def calculate_kst(data, intervals):
    """
    KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst  """
    # print("Calculating KST")
    for i in tqdm(intervals):
        data['kst_' + str(i)] = kst(data['Close'], i)


def calculate_dmi(data, intervals):
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
    # print("Calculating DMI")
    for i in tqdm(intervals):
        data['dmi_'+str(i)] = adx(data['High'], data['Low'], data['Close'], n=i, fillna=True)


def calculate_bb_mav(data, intervals):
    """
    Bollinger Bands (BB)
    N-period simple moving average (MA).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    """
    # print("Calculating Bollinger Band MAV")
    for i in tqdm(intervals):
        data['bb_' + str(i)] = bollinger_mavg(data['Close'], n=i, fillna=True)


def calculate_fi(data, intervals):
    """
    Force Index (FI)
    It illustrates how strong the actual buying or selling pressure is.
    High positive values mean there is a strong rising trend, and low
    values signify a strong downward trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    """
    # print("Calculating Force Index")
    for i in tqdm(intervals):
        data['fi_' + str(i)] = force_index(data['Close'], data['Volume'], n=i, fillna=True)


def calculate_eom(ohlc, ndays=20):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    # print("Calculating EOM")
    dm = (ohlc['High'] + ohlc['Low']) / 2 - (ohlc['High'].shift(1) - ohlc['Low'].shift(1)) / 2
    br = ohlc['Volume'] / (ohlc['High'] - ohlc['Low'])
    evm = dm / br
    for i in tqdm(ndays):
        ohlc['eom_' + str(i)] = evm.rolling(i).mean()


def calculate_technical_indicators(data, intervals):
    # Momentum Indicators
    calculate_rsi(data, intervals)
    calculate_williamr(data, intervals)
    calculate_mfi(data, intervals)

    # calculate_MACD(data, intervals)  # ready to use +3
    # calculate_PPO(data, intervals)  #ready to use +1
    calculate_roc(data, intervals)  # Took long
    calculate_cmf(data, intervals)  # volume EMA
    calculate_cmo(data, intervals)  # Took long
    calculate_sma(data, intervals)
    calculate_sma(data, intervals, 'Open')
    calculate_ema(data, intervals)
    calculate_wma(data, intervals)
    """
    calculate_HMA(data, intervals)
    """
    # Trending
    calculate_trix(data, intervals)
    calculate_cci(data, intervals)
    calculate_dpo(data, intervals)  # Trend oscillator
    calculate_kst(data, intervals)
    calculate_dmi(data, intervals)
    # volatility
    calculate_bb_mav(data, intervals)
    # calculate_PSI(data, intervals)  # can't find formula
    calculate_fi(data, intervals)  # volume
    calculate_eom(data, intervals)
