import collections
import numpy as np


class EpisodeStats:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = collections.deque(maxlen=20)

    def add_episode(self, episode_reward, episode_length):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(episode_reward)

    def mean_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

    def recent_mean_reward(self):
        if len(self.recent_rewards) == 0:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def stddev_reward(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.std(self.episode_rewards)

    def mean_length(self):
        if len(self.episode_lengths) == 0:
            return 0.0
        return sum(self.episode_lengths) / len(self.episode_lengths)

    def __repr__(self):
        return f"Mean Reward: {self.mean_reward():.2f} +/- {self.stddev_reward() * 1.96:.2f}, Recent Rewards mean: {self.recent_mean_reward():.2f}"
