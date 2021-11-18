import gym
import numpy as np
from gym.spaces import Discrete, Box

from commons import Policy


class MultiBandit(gym.Env):
    action_space = Discrete(30)
    observation_space = Discrete(1)
    MaxStep = 1
    gamma = 1

    # observation_space = Box(0, 0, (1,))

    def __init__(self, mean_factor=1, scale_factor=1) -> None:
        super().__init__()
        self.ready = False
        self.state = 0

        np.random.seed(0)
        self.means = np.random.rand(30) * mean_factor
        self.scales = np.random.rand(30) * scale_factor

        # maximum expected returns: 0.978618342232764
        # print(f'maximum expected returns: {self.means.max()}')

    def step(self, action: int):
        assert self.ready, "please reset."
        assert action in range(30)

        r = np.random.normal(self.means[action], self.scales[action])
        self.ready = False
        d = True

        return self.state, r, d, ''

    def reset(self):
        self.ready = True
        return self.state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    def policy_value(self, pi: Policy):
        actions, action_probabilities = pi.action_dist(self.state)
        return np.sum([
            p * self.means[i] for i, p in enumerate(action_probabilities)
        ])

    def optimal_policy(self, state):
        return np.argmax(self.means)
