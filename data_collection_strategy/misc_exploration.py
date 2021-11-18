import gym
import copy
import time
import torch
import random
import numpy as np
from torch.optim import SGD
from gym.spaces import Discrete
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from commons import Policy, DataCollector, AvgAccumGradOptimizer
from commons import AdaptiveDefaultDict, DefaultHasher
from commons import Global, sample, argmax, finished_trajectories
from policy import REINFORCE, BehaviorCloner
from models import Actor
from models import VarQNetwork, QNetwork
from value_estimation import FitterQ


class CombinedExplorer(DataCollector):
    def __init__(self, pi_e: Policy, **kwargs) -> None:
        super().__init__(pi_e, **kwargs)
        self.step = 0
        self.switch_steps = kwargs['combined_switch_steps']
        self.collectors = [exp(pi_e, **kwargs) for exp in kwargs['combined_collectors']]
        self.ignore_history = kwargs['combined_ignore_history']

        assert len(self.switch_steps) + 1 == len(self.collectors)

    def act(self, time_step, state):
        if self.switch_steps and self.step == self.switch_steps[0]:
            print(f'\nstep {self.step}, explorer switch from {self.collectors[0].__class__.__name__} to {self.collectors[1].__class__.__name__}')
            self.switch_steps = self.switch_steps[1:]
            self.collectors = self.collectors[1:]

        return self.collectors[0].act(time_step, state)

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        self.step += 1

        if self.ignore_history:
            update_collectors = self.collectors[:1]
        else:
            update_collectors = self.collectors

        for collect in update_collectors:
            collect.update(time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories)


# ================================================ robust on-policy ================================================
class RobustOnPolicySampler(DataCollector):
    def __init__(self, pi_e: REINFORCE, hasher: DefaultHasher, env: gym.Env, max_time_step: int, **kwargs):
        DataCollector.__init__(self, pi_e)
        self.pi_e: REINFORCE = pi_e
        self.step = 0
        self.env = env
        self.max_time_step = max_time_step
        self.count = AdaptiveDefaultDict(int, hasher)

        self.lr = kwargs['ros_lr']
        self.pi_b = copy.deepcopy(pi_e)

        self.grad0 = 0
        self.avg_grad = 0
        self.smooth_grad = 0
        self.dynamic_lr = kwargs['ros_dynamic_lr']

        if kwargs['ros_first_layer']:
            # update_modules = self.pi_b.model.first_layer()
            update_modules = self.pi_b.model.last_layer()
        else:
            update_modules = self.pi_b.model

        parameters = (parameter for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)
        parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if not kwargs['ros_only_weight'] or 'bias' not in name)

        # parameters = (parameter for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')
        # parameter_names = ((name, parameter.shape) for name, parameter in update_modules.named_parameters() if name != 'model.1.0.bias')

        self.optimizer = AvgAccumGradOptimizer(parameters, lr=self.lr)
        print('ROS update:')
        [print(item) for item in parameter_names]

    def recover_parameters(self):
        for source_param, dump_param in zip(self.pi_b.model.parameters(), self.pi_e.model.parameters()):
            source_param.data.copy_(dump_param.data)

    def act(self, time_step, state):
        action, pi_b = self.pi_b.sample(state)
        pi_e = self.pi_e.action_prob(state, action)
        # if np.isnan(pi_e):
        #     pass
        return action, pi_e, pi_b

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        self.step += 1
        self.count[(state, 0, action)] += 1

        self.pi_b.model.eval()  # fix BN
        self.recover_parameters()
        loss = self.pi_b.action_prob(state, action, require_grad=True).log()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(old_weight=self.step - 1, new_weight=1)


class RobustOnPolicyActing(RobustOnPolicySampler):
    def __init__(self, pi_e: REINFORCE, hasher: DefaultHasher, env: gym.Env, max_time_step: int, **kwargs):
        kwargs['ros_lr'] = 0
        RobustOnPolicySampler.__init__(self, pi_e, hasher, env, max_time_step, **kwargs)

        self.epsilon = kwargs['ros_epsilon']
        self.n_action = kwargs['ros_n_action']
        self.ros_increasing_epsilon = kwargs['ros_increasing_epsilon']
        self.max_data_collection_step = kwargs['max_data_collection_step']

    def act(self, time_step, state):
        if self.ros_increasing_epsilon:
            epsilon = min(self.epsilon * self.step / self.max_data_collection_step, 1)
        else:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            action_dist = self.pi_e.action_dist(state)
            if isinstance(action_dist, torch.distributions.Categorical):
                action_probabilities = action_dist.probs.numpy()
                actions = list(range(len(action_probabilities)))

                if (action_probabilities == 1).any():  # gradient disappear
                    return sample(actions, action_probabilities, action_probabilities)
            else:
                actions = action_dist.icdf(torch.arange(1, self.n_action + 1) / (self.n_action + 1)).unsqueeze(-1)
                action_probabilities = action_dist.log_prob(actions).squeeze().exp().numpy()
                actions = actions.tolist()

            avg_grads = []
            optimizer = self.optimizer
            log_probs = self.pi_b.action_prob(state, actions, require_grad=True).log()
            for log_prob in log_probs:
                optimizer.zero_grad()
                log_prob.backward(retain_graph=True)
                avg_grads.append(optimizer.step(old_weight=self.step, new_weight=1, update_avg_grad=False))

            return argmax(actions, action_probabilities, -np.array(avg_grads))
        else:
            action, pi_e = self.pi_e.sample(state)
            return action, pi_e, pi_e
