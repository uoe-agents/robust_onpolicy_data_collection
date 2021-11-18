import copy
import torch
from torch.optim import SGD

from policy import REINFORCE
from commons import DataCollector
from commons import sample


def one_trajectory(trajectory, gamma, pi_b=None):
    g = 0
    pr_e = 1
    pr_b = 1
    for t, s, a, r, s_, d, pie_a, pib_a in reversed(trajectory):
        g = g * gamma + r
        pr_e *= pie_a
        pr_b *= pib_a
    return g, pr_e, pr_b


class BehaviorPolicyGradient(DataCollector):

    def __init__(self, pi_e: REINFORCE, gamma: float, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0
        self.pi_e = pi_e
        self.gamma = gamma
        self.pi_b = copy.deepcopy(pi_e)

        self.lr = kwargs['bpg_lr']
        self.k = kwargs['bpg_k']
        self.optimizer = SGD(self.pi_b.model.parameters(), self.lr)

    def act(self, time_step, state):
        self.step += 1

        action, pi_b = self.pi_b.sample(state)
        pi_e = self.pi_e.action_prob(state, action)

        # if self.step % 1000 == 0:
        #     print('\n', pi_e, pi_b, '\n')

        return action, pi_e, pi_b

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        if done and (len(trajectories) + 1) % self.k == 0:
            self.pi_b.model.eval()  # fix BN
            loss = 0
            for trajectory in trajectories[-self.k:]:
                g = 0
                pr_e = 1
                pr_b = 1
                log_prop = 0
                for t, s, a, r, s_, d, pie_a, pib_a in reversed(trajectory):
                    g = g * self.gamma + r
                    pr_e *= pie_a
                    pr_b *= pib_a
                    log_prop = log_prop + self.pi_b.action_prob(s, a, require_grad=True).log()
                IS_squared = (g * pr_e / pr_b) ** 2
                loss = loss - IS_squared * log_prop

            self.optimizer.zero_grad()
            loss /= self.k
            loss.backward()
            self.optimizer.step()
