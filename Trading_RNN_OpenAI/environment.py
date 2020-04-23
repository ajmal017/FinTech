import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from process_data import FeatureExtractor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from collections import deque

# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2


class OhlcvEnv(gym.Env):
    def __init__(self, window_size, data, train=True, show_trade=True, investment=100000):
        self.train = train
        self.show_trade = show_trade
        self.df = data
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = 0.0005
        self.investment = investment
        self.seed()

        self.n_long = 0
        self.n_short = 0
        self.closingPrices = []
        self.np_random = None
        self.current_tick = 0
        self.history = []
        self.action = HOLD
        self.position = FLAT
        self.done = False
        self.closingPrice = 0
        self.krw_balance = 0
        self.portfolio = 0
        self.profit = 0
        self.state_queue = None
        self.state = None

        # load data
        self.load_data()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_data(self):
        extractor = FeatureExtractor(self.df)
        # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
        self.df = extractor.add_bar_features()

        # selected manual features
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        # drops Nan rows
        self.df.dropna(inplace=True)
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_frame(self, frame):
        offline_scalar = StandardScaler()
        observe = frame[..., :-4]
        observe = offline_scalar.fit_transform(observe)
        agent_state = frame[..., -4:]
        temp = np.concatenate((observe, agent_state), axis=1)
        return temp

    def step(self, action):
        s, r, d, i = self._step(action)
        self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue))), r, d, i

    def _step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considered "buy - hold"
        # hold
        self.action = HOLD
        if action == BUY:
            # buy
            if self.position == FLAT:
                # if previous position was flat
                # update position to long
                self.position = LONG
                # record action as buy
                self.action = BUY
                # maintain entry price
                self.entry_price = self.closingPrice
            elif self.position == SHORT:
                # if previous position was short
                # update position to flat
                self.position = FLAT
                # record action as buy
                self.action = BUY
                self.exit_price = self.closingPrice
                # calculate reward
                self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1
                # evaluate cumulative return in krw-won
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                # clear entry price
                self.entry_price = 0
                # record number of short
                self.n_short += 1
        elif action == SELL:
            # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT
                self.action = SELL
                self.entry_price = self.closingPrice
            elif self.position == LONG:
                self.position = FLAT
                self.action = SELL
                self.exit_price = self.closingPrice
                self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1

        # [coin + krw_won] total value evaluated in krw won
        if self.position == LONG:
            temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif self.position == SHORT:
            temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        # if self.show_trade and self.current_tick % 100 == 0:
        #    print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
        #    print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.state = self.update_state()
        info = {
                'portfolio': np.array([self.portfolio]),
                "history": self.history,
                "n_trades": {
                    'long': self.n_long,
                    'short': self.n_short}}
        if self.current_tick > (self.df.shape[0]) - self.window_size-1:
            self.done = True
            # return reward at end of the game
            self.reward = self.get_profit()
            if not self.train:
                np.array([info]).dump(
                    './info/ppo_{0}_LS_{1}_{2}.info'.format(self.portfolio, self.n_long, self.n_short))
        if self.done:
            print("Finish at {0} portfolio:{1:6.3f}".format(self.current_tick, self.portfolio))

        return self.state, self.reward, self.done, info

    def get_profit(self):
        if self.position == LONG:
            profit = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif self.position == SHORT:
            profit = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        # self.current_tick = random.randint(0, self.df.shape[0]-800)
        if self.train:
            self.current_tick = random.randint(0, self.df.shape[0] - 2000)
        else:
            self.current_tick = 0

        #print("start episode ... at {0}" .format(self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        # keep buy, sell, hold action history
        self.history = [] 
        # initial balance, u can change it to whatever u like
        self.krw_balance = self.investment
        # (coin * current_price + current_krw_balance) == portfolio
        self.portfolio = float(self.krw_balance) 
        self.profit = 0
        self.closingPrice = self.closingPrices[self.current_tick]

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()
        return self.state

    def preheat_queue(self):
        while len(self.state_queue) < self.window_size:
            # rand_action = random.randint(0, len(self.actions)-1)
            rand_action = 2
            s, r, d, i = self._step(rand_action)
            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))

    def update_state(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position, 3)
        profit = self.get_profit()
        # append two
        state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
        return state.reshape(1, -1)
