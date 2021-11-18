import torch
import numpy as np

from commons import Policy, DataCollector
from commons import sample


class OnPolicySampler(DataCollector):

    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0

    def act(self, time_step, state):
        action, pi_e = self.pi_e.sample(state)
        return action, pi_e, pi_e

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        self.step += 1


class EpsilonPolicySampler(DataCollector):

    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0
        self.epsilon = kwargs['ops_epsilon']

        assert 0 <= self.epsilon <= 1.

    def act(self, time_step, state):
        action_dist = self.pi_e.action_dist(state)
        if isinstance(action_dist, torch.distributions.Categorical):
            action_probabilities = action_dist.probs.numpy()
            actions = list(range(len(action_probabilities)))
            uniform_probabilities = np.ones_like(action_probabilities) / len(action_probabilities)
            sample_probabilities = self.epsilon * uniform_probabilities + (1 - self.epsilon) * action_probabilities
            return sample(actions, action_probabilities, sample_probabilities)
        else:
            sample_action_dist = torch.distributions.Normal(action_dist.loc, action_dist.scale * (1 + self.epsilon))
            action = sample_action_dist.sample()
            pi_e = action_dist.log_prob(action).exp()
            pi_b = sample_action_dist.log_prob(action).exp()
            return action.tolist(), pi_e.item(), pi_b.item()

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        self.step += 1
