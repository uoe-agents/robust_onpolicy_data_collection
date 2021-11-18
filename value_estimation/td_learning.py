import gym
import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

from models import VarQNetwork
from policy import BehaviorCloner, REINFORCE
from commons import Policy, ValueEstimator
from commons import Global, finished_trajectories


class FitterQ(ValueEstimator):
    def __init__(self, pi_e: REINFORCE, gamma: float, env: gym.Env, dropout: bool = False, **kwargs) -> None:
        seed = kwargs['seed']
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        super().__init__(pi_e, gamma)
        self.pi_e = pi_e
        self.env = env
        self.device = torch.device(kwargs['device'])
        self.model = VarQNetwork(self.env, self.device, dropout)
        self.target_model = copy.deepcopy(self.model).eval()
        self.model_dump = copy.deepcopy(self.model)
        # print(f'FQE model (estimate variance={self.variance})', self.model)

        self.max_time_step = kwargs['max_time_step']
        self.lr = kwargs['fqe_lr']
        self.batch_size = kwargs['fqe_batch']
        self.tau = kwargs['fqe_tau']
        self.num_epochs = kwargs['fqe_epochs']
        self.re_weight = kwargs['fqe_re_weight']
        self.reward_range = kwargs['reward_range']

        self.trainings = TensorDataset()
        self.initials = TensorDataset()
        self.behavior_cloner = BehaviorCloner(pi_e, **kwargs)

    def soft_update(self):
        for source_param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def recover_parameters(self):
        for source_param, dump_param in zip(self.model.parameters(), self.model_dump.parameters()):
            source_param.data.copy_(dump_param.data)
        for source_param, dump_param in zip(self.target_model.parameters(), self.model_dump.parameters()):
            source_param.data.copy_(dump_param.data)

    def build_dataset(self, trajectories, pi_D):
        trainings = []
        initials = []
        for trajectory in trajectories:
            log_Pr_e = 0
            log_Pr_b = 0
            for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
                if self.re_weight:
                    assert pi_D is not None
                    # += vs =
                    log_Pr_e = np.log(pie_a)
                    # log_Pr_b += np.log(pib_a)
                    log_Pr_b = np.log(pi_D.action_prob(s, a))

                n_actions, n_action_ps = self.pi_e.discrete_action_dist(s_)
                trainings.append((log_Pr_e, log_Pr_b, [t], s, a, r, s_, d, n_actions, n_action_ps))
                if t == 0:
                    init_actions, init_action_ps = self.pi_e.discrete_action_dist(s)
                    initials.append(([t], s, init_actions, init_action_ps))

        log_Pr_e, log_Pr_b, time_steps, states, actions, rewards, next_states, dones, next_actions, next_action_ps = (torch.Tensor(tensors) for tensors in zip(*trainings))
        init_time_steps, init_states, init_actions, init_action_ps = (torch.Tensor(tensors) for tensors in zip(*initials))
        weights = (log_Pr_e - log_Pr_b).exp()

        self.trainings = TensorDataset(weights, time_steps, states, actions, rewards, dones, next_states, next_actions, next_action_ps)
        self.initials = TensorDataset(init_time_steps, init_states, init_actions, init_action_ps)

    def get_continuous_action_dist(self, next_states, num_action):
        next_actions = self.pi_e.action_dists(next_states).sample((num_action,)).transpose(0, 1).to(self.device)
        next_action_ps = torch.ones(next_states.size(0), num_action).to(self.device) / num_action

        return next_actions, next_action_ps

    def update(self, trajectories, pi_D: Policy):
        trajectories = finished_trajectories(trajectories)
        if not trajectories:
            return

        self.build_dataset(trajectories, pi_D)

        # self.recover_parameters()
        num_epochs = max(self.num_epochs, 1000 // (len(self.trainings) // self.batch_size + 1))  # min_update_times = 1000
        self.target_model = copy.deepcopy(self.model).eval()
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        # optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        lr_scheduler = CosineAnnealingLR(optimizer, num_epochs)
        for epoch in range(num_epochs):
            losses = []
            self.model.train()
            # self.target_model.train()
            for items in DataLoader(self.trainings, batch_size=self.batch_size, shuffle=True):
                weights, time_steps, states, actions, rewards, dones, next_states, next_actions, next_action_ps = (item.to(self.device) for item in items)

                if isinstance(self.env.action_space, Box):
                    next_actions, next_action_ps = self.get_continuous_action_dist(next_states, num_action=31)

                next_time_steps = torch.stack((time_steps + 1,) * next_actions.size(1), 1)
                next_states = torch.stack((next_states,) * next_actions.size(1), 1)

                q_predict, squared_q_predict = self.model(time_steps, states, actions)
                with torch.no_grad():
                    q_next, squared_q_next = self.target_model(next_time_steps, next_states, next_actions)
                    squared_q_next = squared_q_next.clip(0)

                if self.gamma != 1.:
                    r_min = min(self.reward_range)
                    r_max = max(self.reward_range)
                    possible_q_bound = (r_min, r_max, r_min / (1 - self.gamma), r_max / (1 - self.gamma))
                    q_next = q_next.clip(min(possible_q_bound), max(possible_q_bound))

                mean_q_next = (q_next * next_action_ps).sum(1)
                q_target = rewards + (1 - dones) * self.gamma * mean_q_next
                loss = (q_predict - q_target).pow(2)

                # # for variance of returns
                # mean_squared_q_next = (squared_q_next * next_action_ps).sum(1)
                # # squared_q_target = rewards.pow(2) + (1 - dones) * (2 * self.gamma * rewards * mean_q_next + self.gamma ** 2 * mean_squared_q_next)
                # # squared_q_target = q_target.pow(2)
                # squared_q_target = (q_target - q_predict).pow(2)
                # loss += (squared_q_predict - squared_q_target.clip(0)).pow(2)

                loss = (weights * loss).sum() / weights.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.soft_update()

                losses.append(loss.item())
                Global.setdefault('0002_4', []).extend(weights[weights != 1].cpu().numpy().tolist())

            lr_scheduler.step()
            Global.setdefault('0002_1', []).append(np.mean(losses))
            q, v = self._estimate(debug=True)
            Global.setdefault('0002_2', []).append(q)
            Global.setdefault('0002_3', []).append(v)

        fig = plt.figure(figsize=(8, 20))
        plt.subplot(411)
        plt.plot(Global['0002_1'])
        plt.yscale('log')
        plt.subplot(412)
        plt.plot(Global['0002_2'])
        plt.hlines(1, 0, len(Global['0002_2']), linestyles='--', alpha=0.3)
        plt.subplot(413)
        plt.plot(Global['0002_3'])
        plt.hlines(Global['true_variance'], 0, len(Global['0002_3']), linestyles='--', alpha=0.3)
        weights = np.log(np.array(Global['0002_4']).clip(-1e10, 1e10))
        plt.subplot(414)
        plt.hist(weights, bins=100)
        plt.savefig('results/0002.jpg')
        plt.close(fig)

    def _estimate(self, debug=False):
        self.model.eval()
        init_q_values = []
        init_variances = []
        with torch.no_grad():
            for items in DataLoader(self.initials, batch_size=self.batch_size):
                init_time_steps, init_states, init_actions, init_action_ps = (item.to(self.device) for item in items)

                if isinstance(self.env.action_space, Box):
                    init_actions, init_action_ps = self.get_continuous_action_dist(init_states, num_action=100 if debug else 100)

                init_states = torch.stack((init_states,) * init_actions.size(1), 1)
                init_time_steps = torch.stack((init_time_steps,) * init_actions.size(1), 1)

                init_q, init_squared_q = self.model(init_time_steps, init_states, init_actions)
                mean_init_q = (init_q * init_action_ps).sum(1)
                mean_init_squared_q = (init_squared_q * init_action_ps).sum(1)
                mean_init_variance = mean_init_squared_q - mean_init_q.pow(2)

                init_q_values.append(mean_init_q)
                init_variances.append(mean_init_variance)

        return torch.cat(init_q_values).mean().item(), torch.cat(init_variances).mean().item()

    def estimate(self, trajectories, pi_D):
        if not trajectories[0][-1][5]:
            return 0

        self.update(trajectories, pi_D)
        return self._estimate()[0]

    def estimate_variance(self, time_step, state, actions):
        self.model.eval()
        with torch.no_grad():
            time_steps = torch.FloatTensor([[time_step]] * len(actions)).to(self.device)
            states = torch.FloatTensor([state] * len(actions)).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            q, squared_q = self.model(time_steps, states, actions)
            return q.cpu().numpy(), squared_q.cpu().numpy().clip(0),  # r.cpu().numpy(), squared_r.cpu().numpy().clip(0), m.cpu().numpy().clip(0), n.cpu().numpy().clip(0)

    def estimate_dropout_variance(self, time_step, state, actions):
        n = 1000
        self.model.train()
        with torch.no_grad():
            time_steps = torch.FloatTensor([[[time_step]] * len(actions)] * n).to(self.device)
            states = torch.FloatTensor([[state] * len(actions)] * n).to(self.device)
            actions = torch.FloatTensor([actions] * n).to(self.device)
            q, _ = self.model(time_steps, states, actions)
            return q.cpu().numpy()

# class TabularFQE(ValueEstimator):
#     def __init__(self, pi_e: Policy, gamma: float, **kwargs) -> None:
#         super().__init__(pi_e, gamma)
#         self.q_table = AdaptiveDefaultDict(int)
#
#         self.tau = 1
#         self.target_delta = 0.001
#
#     def estimate_v_value(self, state, time_step):
#         v = 0
#         actions, action_probabilities = self.pi_e.action_dist(state)
#         for a, pa in zip(actions, action_probabilities):
#             v += pa * self.q_table[(state, time_step, a)]
#         return v
#
#     def make_dataset(self, trajectories=None, transitions=None):
#         if transitions is None:
#             trainings = AdaptiveDefaultDict()
#             for trajectory in trajectories:
#                 for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
#                     trainings.setdefault((s, t, a), []).append((r, d, s_))  # todo.
#         else:
#             trainings = AdaptiveDefaultDict()
#             for t, s, a, r, s_, d, pie_a, pib_a in transitions:
#                 trainings.setdefault((s, t, a), []).append((r, d, s_))  # todo.
#         return trainings
#
#     def estimate_q_values(self, time_step, state, actions):
#         return [self.q_table[(state, time_step, a)] for a in actions]
#
#     def _train(self, trainings):
#         # q iteration.
#         while True:
#             max_delta = 0
#             for (s, t, a), rds_ in trainings.items():
#                 r, d, s_ = tuple(np.array(_) for _ in zip(*rds_))
#                 v_ = np.array([self.estimate_v_value(s, t + 1) for s in s_])  # todo t+1
#                 q_target = np.mean(r + (1 - d) * self.gamma * v_)
#                 delta = self.tau * (q_target - self.q_table[(s, t, a)])
#                 self.q_table[(s, t, a)] += delta
#                 max_delta = max(abs(delta), max_delta)
#
#             if max_delta < self.target_delta:
#                 break
#
#     def update(self, transitions):
#         trainings = self.make_dataset(transitions=transitions)
#         self._train(trainings)
#
#     def _estimate(self, trajectories):
#         initial_state_count = AdaptiveDefaultDict(int)
#         for trajectory in trajectories:
#             if len(trajectory):
#                 s0 = trajectory[0][1]
#                 initial_state_count[s0] += 1
#         estimation = 0
#         total = sum(initial_state_count.values())
#         for s0, count in initial_state_count.items():
#             estimation += (count / total) * self.estimate_v_value(s0, 0)
#         return estimation
#
#     def estimate(self, trajectories, pi_D):
#         trainings = self.make_dataset(trajectories=trajectories)
#         self._train(trainings)
#         return self._estimate(trajectories)
#
