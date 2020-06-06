import argparse
import warnings
import matplotlib.pyplot as plt

from data_extraction import read_data
from data_preparation import create_labels
from data_preparation import select_features
from model import create_model_cnn
from train import prepare_data, train
from evaluate import evaluate


def show_data(plt_data):
    fig, ax1 = plt.subplots()
    plt.plot(plt_data['Adj Close'].index, plt_data['Adj Close'], 'b-', linewidth=2)

    # Plot the buy signals
    ax1.plot(plt_data.loc[plt_data.labels == 1].index,
             plt_data['Adj Close'][plt_data.labels == 1],
             '^', markersize=5, color='w')

    # Plot the sell signals
    ax1.plot(plt_data.loc[plt_data.labels == 0].index,
             plt_data['Adj Close'][plt_data.labels == 0],
             'v', markersize=5, color='r')
    fig.autofmt_xdate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', type=int, default=0,
                        help='start index of tickers in S&P List')
    parser.add_argument('-e', '--end', type=int, default=50,
                        help='last index of tickers in S&P List')
    parser.add_argument('-m', '--mode', type=str, default='te',
                        help='Execution mode, t as train, e as evaluation, te as train and evaluation, d for data')
    parser.add_argument('-p', '--plot', type=bool, default=False,
                        help='display algorithm output')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    total_years = 20
    total_dates = total_years * 365
    intervals = range(6, 27)  # 21
    ticker = 'IBM'

    data = read_data(ticker, total_dates, intervals)
    labels = create_labels(data)
    data['labels'] = labels
    data.dropna(inplace=True)

    if args.plot:
        show_data(data[-300:])

    if args.mode != 'n':
        feature_idx, start_col, end_col = select_features(data)
        model, params, mcp, rlp, es = create_model_cnn()
        x_test, y_test, x_cv, y_cv, x_train, y_train, sample_weights = prepare_data(data, start_col, end_col, feature_idx)
        if args.mode == 't' or args.mode == 'te':
            train(model, x_train, y_train, params, x_cv, y_cv, mcp, rlp, es, sample_weights)
        if args.mode == 'e' or args.mode == 'te':
            evaluate(x_test, y_test)
