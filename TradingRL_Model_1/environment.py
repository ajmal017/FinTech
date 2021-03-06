import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A 3-stock (MSFT, IBM, QCOM) trading environment.
    State: [# of stock owned, current stock prices, cash in hand]
      - array of length n_stock * 2 + 1
      - price is discretized (to integer) to reduce state space
      - use close price for each stock
      - cash in hand is evaluated at each step based on action performed
    Action: sell (0), hold (1), and buy (2)
      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """
    def __init__(self, train_data, init_invest=20000):
        # data
        # round up to integer to reduce state space
        self.stock_price_history = np.around(train_data)
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.np_random = None
        self.portfolio_hist = []

        # action space
        self.action_space = spaces.Discrete(3**self.n_stock * 3)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [init_invest * 2 // mx for mx in stock_max_price]
        space_range = (np.concatenate((stock_range, stock_max_price, [init_invest * 2]), axis=0))
        self.observation_space = spaces.MultiDiscrete(space_range)

        # seed and start
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.cur_step = 0
        self.portfolio_hist = []
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        return self.get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        prev_val = self.get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.trade(action)
        cur_val = self.get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val,
                'cur_step': self.cur_step,
                'hist': self.portfolio_hist}
        return self.get_obs(), reward, done, info

    def get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs

    def get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_ratio = (action % 3) + 1
        action_vec = 0
        action_type = (int)(action / 3)
        for i, e in enumerate(itertools.product([0, 1, 2], repeat=self.n_stock)):
            if i == action_type:
                action_vec = e

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                sold_stock = (int)(((2**action_ratio) / 4) * self.stock_owned[i])
                self.cash_in_hand += self.stock_price[i] * sold_stock
                self.stock_owned[i] -= sold_stock
        if buy_index:
            can_buy = True
            available_cache = (int)(((2**action_ratio) / 4) * self.cash_in_hand)
            while can_buy:
                for i in buy_index:
                    if available_cache > self.stock_price[i]:
                        # buy one share
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                        available_cache -= self.stock_price[i]
                    else:
                        can_buy = False
        self.portfolio_hist.append(self.get_val())