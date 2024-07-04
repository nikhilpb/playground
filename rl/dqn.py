import collections
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from episode_stats import EpisodeStats
import os
import pandas as pd


class SimulationConfig():
    def __init__(self,
                 gamma=0.99,
                 num_episodes=100,
                 num_steps=1000, seed=123,
                 train=True,
                 batch_size=32,
                 save_path='',
                 save_every=10,
                 eps_start=1.0,
                 eps_end=0.01,
                 eps_decay=0.98,
                 load_path=None):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.seed = seed
        self.train = train
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_every = save_every
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.load_path = load_path


class DQNAgent():
    def __init__(self,
                 env,
                 lr=0.001,
                 memory_config={'replay_memory_size': 100000,
                                'min_buffer_size': 2000},
                 hidden_size=64,
                 env_config={
                     'state_size': -1,
                     'action_size': -1,
                     'state_mapper': (lambda x: x),
                     'int_to_action': (lambda x: x)},
                 ):
        assert (env_config['state_size'] > 0)
        assert (env_config['action_size'] > 0)
        self.action_size = env_config['action_size']
        self.qfn = QFunction(input_size=env_config['state_size'], hidden_size=hidden_size,
                             output_size=self.action_size)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.qfn.parameters(), lr=self.lr)
        self.replay_memory_size = memory_config['replay_memory_size']
        self.replay_samples = collections.deque(maxlen=self.replay_memory_size)
        self.min_buffer_size = memory_config['min_buffer_size']
        self.state_mapper = env_config['state_mapper']
        self.int_to_action = env_config['int_to_action']

    def act(self, observation, epsilon=0.0):
        if np.random.uniform() < epsilon:
            action_int = np.random.randint(0, self.action_size)
            return (self.int_to_action(action_int), action_int, None)
        state = self.state_mapper(observation)
        qs = self.qfn(state)
        q_max, action_int = torch.max(qs, 0)
        action_int = action_int.item()
        q_max = q_max.item()
        action = self.int_to_action(action_int)
        return (action, action_int, q_max)

    def simulate(self, config, env):
        np.random.seed(config.seed)
        stats = EpisodeStats()
        evolution = pd.DataFrame(
            columns=['episode', 'bellman_loss', 'epsilon', 'reward'])
        if config.load_path:
            print('Loading model from {config.load_path}'.format(**locals()))
            self.qfn.load_state_dict(torch.load(config.load_path))
        for e in range(config.num_episodes):
            losses = []
            epsilon = config.eps_start * config.eps_decay**e
            epsilon = max(epsilon, config.eps_end)
            if config.train and e % config.save_every == 0 and e > 0:
                model_filename = os.path.join(
                    config.save_path, 'model-episode-{e}.pt'.format(**locals()))
                torch.save(self.qfn.state_dict(), model_filename)
                evolution.to_csv(os.path.join(
                    config.save_path, 'evolution.csv'), index=False)
            print(
                'Episode {e}, epsilon = {epsilon}, current_stats = {stats}'.format(**locals()))
            total_reward = 0
            total_length = 0
            observation, _ = env.reset()
            for _ in range(config.num_steps):
                action, action_int, q_max = self.act(torch.tensor(
                    observation, dtype=torch.float32), epsilon=epsilon)
                next_observation, reward, terminated, truncated, _ = env.step(
                    action)
                total_reward += reward
                total_length += 1
                replay_sample = ReplaySample(
                    self.state_mapper(observation), action, reward, self.state_mapper(next_observation), terminated, action_int)
                self.replay_samples.append(replay_sample)
                if len(self.replay_samples) >= self.min_buffer_size and config.train:
                    loss = self._update_step(config.gamma, config.batch_size)
                    losses.append(loss)
                if terminated or truncated:
                    stats.add_episode(total_reward, total_length)
                    break
                observation = next_observation
            bellman_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
            new_df = pd.DataFrame([{'episode': e, 'bellman_loss': bellman_loss,
                                    'epsilon': epsilon, 'reward': total_reward}])
            evolution = pd.concat([evolution, new_df], ignore_index=True)

    def _update_step(self, gamma, batch_size):
        replay_samples = np.random.choice(self.replay_samples,   batch_size)
        states, _, rewards, next_states, terminateds, action_ints = process_replay_samples(
            replay_samples)
        q_fns = torch.gather(self.qfn(states), 1,
                             action_ints.reshape(-1, 1)).squeeze()
        q_fns_next, _ = torch.max(self.qfn(next_states), dim=1)
        targets = rewards + gamma * q_fns_next * (1.0 - terminateds)
        loss = F.mse_loss(q_fns, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __repr__(self):
        return f"DQNAgent()"


def process_replay_samples(replay_samples):
    states = []
    actions = []
    rewards = []
    next_states = []
    terminateds = []
    action_ints = []
    for replay_sample in replay_samples:
        states.append(replay_sample.state)
        actions.append(replay_sample.action)
        rewards.append(replay_sample.reward)
        next_states.append(replay_sample.next_state)
        terminateds.append(replay_sample.terminated)
        action_ints.append(replay_sample.action_int)
    return torch.tensor(np.array(states), dtype=torch.float32), \
        torch.tensor(np.array(actions), dtype=torch.float32), \
        torch.tensor(np.array(rewards), dtype=torch.float32), \
        torch.tensor(np.array(next_states), dtype=torch.float32), \
        torch.tensor(np.array(terminateds), dtype=torch.float32), \
        torch.tensor(np.array(action_ints), dtype=torch.int64)


class QFunction(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=10):
        super(QFunction, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class ReplaySample():
    def __init__(self, state, action, reward, next_state, terminated, action_int):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminated = terminated
        self.action_int = action_int


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=123)
    agent = DQNAgent(env, lr=0.001,
                     hidden_size=64,
                     env_config={
                         'state_size': 4,
                         'action_size': 2,
                         'state_mapper': lambda x: x,
                         'int_to_action': lambda x: x
                     })

    config = SimulationConfig(gamma=0.99, num_episodes=500, num_steps=1000,
                              train=True, batch_size=256, save_path='models/cartpole_v1',
                              save_every=100, eps_start=1.0, eps_end=0.01, eps_decay=0.98)
    agent.simulate(config, env)
