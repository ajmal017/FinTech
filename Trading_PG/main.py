import os
import tensorflow as tf
from market import Market
from helper import generate_algorithm_logger
from helper import generate_market_logger
from helper import LOGS_DIR
from algorithm_pg import AlgorithmPG
from algorithm_ddqn import AlgorithmDDQN
from argpaser import stock_codes
from argpaser import model_launcher_parser


CHECKPOINTS_DIR = os.path.dirname(__file__)


def launch_model():
    # Model Name.
    model_name = 'PolicyGradient'
    # Market Type.
    market = 'stock'
    # Codes.
    codes = stock_codes
    # Start date.
    start = "1998-01-01"
    # End date.
    end = "2018-01-01"
    # Episodes.
    episode = 1000
    # Training data ratio.
    training_data_ratio = 0.8

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    mode = "train"

    env = Market(codes, start_date=start, end_date=end, **{
        "market": market,
        "mix_index_state": False,
        "logger": generate_market_logger(model_name),
        "training_data_ratio": training_data_ratio,
    })

    if model_name == "PolicyGradient":
        algorithm = AlgorithmPG(tf.Session(config=config), env, env.trader.action_space, env.data_dim, **{
            "mode": mode,
            "episodes": episode,
            "enable_saver": True,
            "learning_rate": 0.003,
            "enable_summary_writer": True,
            "logger": generate_algorithm_logger(model_name),
            "save_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "summary"),
        })
    else:
        algorithm = AlgorithmDDQN(tf.Session(config=config), env, env.trader.action_space, env.data_dim, **{
            "mode": mode,
            "episodes": episode,
            "enable_saver": True,
            "learning_rate": 0.003,
            "enable_summary_writer": True,
            "logger": generate_algorithm_logger(model_name),
            "save_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "summary"),
        })

    algorithm.run()
    algorithm.eval()
    algorithm.plot()


if __name__ == '__main__':
    launch_model()
