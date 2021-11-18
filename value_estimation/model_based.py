import copy
import gym
import torch
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from gym.spaces import Discrete
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from commons import Policy, ValueEstimator
from commons import Global, finished_trajectories
from commons import mc_estimation_true_value
from policy import REINFORCE


class ModelBased(ValueEstimator, torch.nn.Module, gym.Env):

    def __init__(self, pi_e: Policy, gamma: float, env: gym.Env, **kwargs) -> None:
        ValueEstimator.__init__(self, pi_e, gamma)
        torch.nn.Module.__init__(self)
        gym.Env.__init__(self)

        seed = kwargs['seed']
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # ================================= build model =================================
        self.env = env
        self.device = torch.device(kwargs['device'])
        if isinstance(self.env.observation_space, Discrete):
            self.state_n = self.env.observation_space.n
            self.state_dim = 64
            self.state_embedding = torch.nn.Embedding(self.state_n, self.state_dim).to(self.device)
        else:
            self.state_n = 0
            self.state_dim = env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.action_embedding = torch.nn.Embedding(self.env.action_space.n, 64).to(self.device)
            self.action_dim = 64
        else:
            self.action_dim = env.action_space.shape[0]
        self.dynamics_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.8),
            torch.nn.Linear(256, self.state_n if self.state_n else self.state_dim),
        ).to(self.device)
        self.reward_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.8),
            torch.nn.Linear(256, 1),
            torch.nn.Flatten(-2),
        ).to(self.device)
        self.done_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.8),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(-2),
        ).to(self.device)

        self.model_dump = copy.deepcopy(self)
        # self.dynamics_net_dump = copy.deepcopy(self.dynamics_net)
        # self.reward_net_dump = copy.deepcopy(self.reward_net)
        # self.done_net_dump = copy.deepcopy(self.done_net)

        # ================================= training config =================================
        self.lr = kwargs['mb_lr']
        self.num_epochs = kwargs['mb_epochs']
        self.batch_size = kwargs['mb_batch']
        self.trainings = TensorDataset()
        self.done_weight = 1

        # ================================= gym.Env config =================================
        self.initials = []
        self.done = True
        self.time_step = 0
        self.current_state = None
        self.n_trajectories = kwargs['mb_trs']
        self.reward_range = kwargs['reward_range']
        self.max_time_step = kwargs['max_time_step']

    # ================================= value estimation relevant =================================
    def estimate(self, trajectories, pi_D):
        if not trajectories[0][-1][5]:
            return 0

        self.update(trajectories)
        return self._estimate_fast(self.n_trajectories)

    def _estimate_fast(self, n_trajectories=None):
        assert isinstance(self.pi_e, REINFORCE)

        self.eval()
        returns = []
        lengths = []
        states = torch.Tensor().to(self.device)
        time_steps = torch.Tensor().to(self.device)
        accum_rewards = torch.Tensor().to(self.device)
        with torch.no_grad():
            while len(returns) < n_trajectories:
                # fill current time_steps, states, accum_rewards
                new_init_num = min(n_trajectories - len(returns) - len(states), self.batch_size - len(states))
                if new_init_num:
                    new_init_states = [random.choice(self.initials) for _ in range(new_init_num)]
                    states = torch.cat((states, torch.FloatTensor(new_init_states).to(self.device)), 0)
                    time_steps = torch.cat((time_steps, torch.FloatTensor([0] * new_init_num).to(self.device)), 0)
                    accum_rewards = torch.cat((accum_rewards, torch.zeros(new_init_num).to(self.device)), 0)

                # on-policy sampling actions.
                actions = self.pi_e.action_dists(states).sample()

                # transitioning.
                next_states, rewards, dones = self.forward(states, actions)
                rewards = rewards.clip(min(self.reward_range), max(self.reward_range))
                dones = (dones > torch.rand(dones.size())) | (time_steps == self.max_time_step - 1)
                if self.state_n:
                    next_states = torch.distributions.Categorical(F.softmax(next_states, -1)).sample()

                # accumulate rewards.
                accum_rewards += (self.gamma ** time_steps) * rewards

                # extract returns of done trajectories.
                returns.extend(accum_rewards[dones].tolist())
                lengths.extend((time_steps[dones] + 1).tolist())

                # time_steps, states for the next iteration.
                states = next_states[~dones]
                time_steps = time_steps[~dones] + 1
                accum_rewards = accum_rewards[~dones]

                # if returns:
                #     import sys
                #     sys.stdout.write(
                #         '\b' * 1000 +
                #         f'(MB simulating) finished tr: {len(returns)}/{n_trajectories}, '
                #         f'mean returns: {np.mean(returns)}, variance: {np.var(returns)}, '
                #         f'average length: {np.mean(lengths)}')
                #     sys.stdout.flush()

        return np.mean(returns)

    # ================================= nn.Module relevant =================================
    def forward(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        if isinstance(self.env.observation_space, Discrete):
            states = self.state_embedding(states.long())
        if isinstance(self.env.action_space, Discrete):
            actions = self.action_embedding(actions.long())

        state_actions = torch.cat((states, actions), -1)
        return (
            self.dynamics_net(state_actions),
            self.reward_net(state_actions),
            self.done_net(state_actions)
        )

    def recover_parameters(self):
        for source_param, dump_param in zip(self.parameters(), self.model_dump.parameters()):
            source_param.data.copy_(dump_param.data)

    def build_dataset(self, trajectories):
        trainings = []
        initials = []
        for trajectory in finished_trajectories(trajectories):
            weight = 1
            for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
                # weight *= pie_a / pib_a
                trainings.append((weight, [t], s, a, r, s_, d))
                if t == 0:
                    initials.append(s)

        weights, time_steps, states, actions, rewards, next_states, dones = (torch.Tensor(tensors) for tensors in zip(*trainings))
        self.trainings = TensorDataset(weights, time_steps, states, actions, rewards, next_states, dones)
        self.initials = initials
        self.done_weight = 1 / dones.mean().item() - 1

    def update(self, trajectories):
        self.build_dataset(trajectories)

        # self.recover_parameters()
        num_epochs = max(self.num_epochs, 1000 // (len(self.trainings) // self.batch_size + 1))  # min_update_times = 1000
        optimizer = Adam(self.parameters(), lr=self.lr)
        # optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        lr_scheduler = CosineAnnealingLR(optimizer, num_epochs)
        for epoch in range(num_epochs):
            losses = []
            self.train()
            for weights, time_steps, states, actions, rewards, next_states, dones in DataLoader(self.trainings, batch_size=self.batch_size, shuffle=True):
                next_states_predict, rewards_predict, dones_predict = self.forward(states, actions)

                if self.state_n:
                    state_loss = F.cross_entropy(next_states_predict, next_states.long(), reduction='none')
                else:
                    state_loss = (next_states_predict - next_states).pow(2).sum(1)
                reward_loss = (rewards_predict - rewards).pow(2)
                done_loss = F.binary_cross_entropy(dones_predict, dones, reduction='none')
                done_loss *= (time_steps < self.max_time_step - 1).squeeze(-1)  # input without time_step
                # done_loss[dones.bool()] *= self.done_weight
                # done_loss[~dones.bool()] *= 1 / self.done_weight
                assert not (time_steps > self.max_time_step - 1).any(), (time_steps, self.max_time_step)

                loss = state_loss + reward_loss + done_loss
                loss = (weights * loss).sum() / weights.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append((
                    loss.item(),
                    ((weights * state_loss).sum() / weights.sum()).item(),
                    ((weights * reward_loss).sum() / weights.sum()).item(),
                    ((weights * done_loss).sum() / weights.sum()).item(),
                ))
            lr_scheduler.step()
            losses = np.array(losses).mean(0)
            Global.setdefault('0003_1', []).append(losses[0])
            Global.setdefault('0003_2', []).append(losses[1])
            Global.setdefault('0003_3', []).append(losses[2])
            if (epoch + 1) % 100 == 0:
                estimation = self._estimate_fast(self.batch_size)
                Global.setdefault('0003_4', []).append(estimation)
        fig = plt.figure(figsize=(8, 6 * 4))
        plt.subplot(411)
        plt.plot(Global['0003_1'])
        plt.yscale('log')
        plt.subplot(412)
        plt.plot(Global['0003_2'])
        plt.yscale('log')
        plt.subplot(413)
        plt.plot(Global['0003_3'])
        plt.yscale('log')
        plt.subplot(414)
        plt.plot(Global['0003_4'])
        plt.hlines(1, 0, len(Global['0003_4']), linestyles='--', alpha=0.3)
        plt.savefig('results/0003.jpg')
        plt.close(fig)

    # ================================= gym.Env relevant (deprecated) =================================
    def _estimate_slow(self, n_trajectories=None, show=False):
        n_trajectories = n_trajectories if n_trajectories else self.n_trajectories
        return mc_estimation_true_value(self, self.pi_e, self.gamma, n_trajectories, show=show)[0]

    def step(self, action):
        assert not self.done, "please reset."
        self.time_step += 1

        self.eval()
        with torch.no_grad():
            current_state = torch.FloatTensor([self.current_state])
            action = torch.FloatTensor([action])

            next_state, reward, done = [tensor[0] for tensor in self.forward(current_state, action)]

            if self.state_n:
                next_state = torch.distributions.Categorical(F.softmax(next_state, -1)).sample().item()
            else:
                next_state = next_state.cpu().numpy()
            # done = done > 0.5 or self.time_step == self.max_time_step
            done = done > np.random.random() or self.time_step == self.max_time_step
            reward = reward.item()

            self.current_state = next_state
            self.done = done

        return next_state, reward, done, None

    def reset(self):
        self.done = False
        self.time_step = 0
        self.current_state = np.random.choice(self.initials)

        return self.current_state

    def render(self, mode='human'):
        raise NotImplementedError
