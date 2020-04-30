import logging
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime


DATETIME_NOW = datetime.now().strftime("%Y%m%d%H%M%S")
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def generate_market_logger(model_name):

    market_log_path = '{}-{}-{}'.format(model_name, DATETIME_NOW, 'stock_market.log')

    market_logger = logging.getLogger('stock_market_logger')
    market_logger.setLevel(logging.DEBUG)
    market_log_sh = logging.StreamHandler()
    market_log_sh.setLevel(logging.WARNING)
    market_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, market_log_path))
    market_log_fh.setLevel(logging.DEBUG)
    market_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    market_logger.addHandler(market_log_sh)
    market_logger.addHandler(market_log_fh)

    return market_logger


def generate_algorithm_logger(model_name):

    algorithm_log_path = '{}-{}-{}'.format(model_name, DATETIME_NOW, 'algorithm.log')

    algorithm_logger = logging.getLogger('algorithm_logger')
    algorithm_logger.setLevel(logging.DEBUG)
    algorithm_log_sh = logging.StreamHandler()
    algorithm_log_sh.setLevel(logging.WARNING)
    algorithm_log_fh = logging.FileHandler(os.path.join(LOGS_DIR, algorithm_log_path))
    algorithm_log_fh.setLevel(logging.DEBUG)
    algorithm_log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    algorithm_logger.addHandler(algorithm_log_sh)
    algorithm_logger.addHandler(algorithm_log_fh)
    return algorithm_logger


def plot_stock_series(codes, y, label, save_path, y_desc='Predict', label_desc='Real'):
    row, col = int(math.ceil(len(codes) / 2)), 1 if len(codes) == 1 else 2
    plt.figure(figsize=(20, 15))
    for index, code in enumerate(codes):
        plt.subplot(row * 100 + col * 10 + (index + 1))
        plt.title(code)
        plt.plot(y[:, index], label=y_desc)
        plt.plot(label[:, index], label=label_desc)
        # plt.plot(y[:, index], 'o-', label=y_desc)
        # plt.plot(label[:, index], 'o-', label=label_desc)
        plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=200)


def plot_profits_series(base, profits, save_path):
    plt.figure(figsize=(20, 15))
    plt.subplot(111)
    plt.title("Profits - Baseline")
    plt.plot(base, label='base')
    plt.plot(profits, label='profits')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(save_path, dpi=200)
