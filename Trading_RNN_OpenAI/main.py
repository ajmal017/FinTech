import numpy as np
import os
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from environment import OhlcvEnv
from open_ai_gym import OpenAIGym
from process_data import get_data, get_data_from_file


def create_btc_env(window_size, data, train):
    raw_env = OhlcvEnv(window_size=window_size, data=data, train=train)
    env = OpenAIGym(raw_env, visualize=False)
    return env


def create_network_spec():
    network_spec = [
        {
            "type": "flatten"
        },
        dict(type='dense', size=64, activation='relu'),
        dict(type='dense', size=64, activation='relu'),
        dict(type='internal_lstm', size=64),
    ]
    return network_spec


def create_baseline_spec():
    baseline_spec = [
        {
            "type": "lstm",
            "size": 64,
        },
        dict(type='dense', size=64, activation='relu'),
        dict(type='dense', size=64, activation='relu'),
    ]
    return baseline_spec


LOAD_DIR = os.path.join(os.getcwd(), "model")
SAVE_DIR = os.path.join(LOAD_DIR, "ppo_agent")


# Callback function printing episode statistics
def episode_finished(r):
    reward = "%.6f" % (r.episode_rewards[-1])
    episod = "%4d" % (r.episode)
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".
          format(ep=episod, ts=r.episode_timestep, reward=reward))

    if np.mean(r.episode_rewards[-1]) > 0:
        r.agent.save_model(SAVE_DIR, append_timestep=False)
    return True


def print_simple_log(r):
    print("Finished episode {ep:2d} after {ts} timesteps (reward: {reward})".
          format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))


def main():

    ticker = 'IBM'
    train_data, test_data = get_data(ticker=ticker)
    # train_data, test_data = get_data_from_file(ticker=ticker)
    # window size
    time_step = 5
    train_environment = create_btc_env(window_size=time_step, data=train_data, train=True)
    test_environment = create_btc_env(window_size=time_step, data=test_data, train=False)

    network_spec = create_network_spec()
    baseline_spec = create_baseline_spec()

    agent = PPOAgent(
        discount=0.9999,
        states=train_environment.states,
        actions=train_environment.actions,
        network=network_spec,
        # Agent
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        # MemoryModel
        update_mode=dict(
            unit='timesteps',
            # 10 episodes per update
            batch_size=32,
            # Every 10 episodes
            frequency=10
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=50000
        ),
        # DistributionModel
        distributions=None,
        entropy_regularization=0.0,  # None
        # PGModel
        baseline_mode='states',
        baseline=dict(type='custom', network=baseline_spec),
        baseline_optimizer=dict(
            type='multi_step',
            optimizer=dict(
                type='adam',
                learning_rate=(1e-4)  # 3e-4
            ),
            num_steps=5
        ),
        gae_lambda=0,  # 0
        # PGLRModel
        likelihood_ratio_clipping=0.2,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=(1e-4)  # 1e-4
        ),
        subsampling_fraction=0.2,  # 0.1
        optimization_steps=10,
        execution=dict(
            type='single',
            session_config=None,
            distributed_spec=None
        )
    )

    train_runner = Runner(
        agent=agent, 
        environment=train_environment
    )
    test_runner = Runner(
        agent=agent,
        environment=test_environment,
    )

    train_runner.run(episodes=100, max_episode_timesteps=500000, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=train_runner.episode,
        ar=np.mean(train_runner.episode_rewards[-100:]))
    )

    test_runner.run(num_episodes=1, deterministic=True, testing=True, episode_finished=print_simple_log)


if __name__ == '__main__':
    main()
