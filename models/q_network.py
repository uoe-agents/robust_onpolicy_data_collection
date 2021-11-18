import copy
import torch
import gym
from gym.spaces import Discrete


class QNetwork(torch.nn.Module):
    def __init__(self, env: gym.Env, device: torch.device, dropout: bool = False):
        super().__init__()
        self.env = env
        self.device = device

        if isinstance(self.env.observation_space, Discrete):
            self.state_embedding = torch.nn.Embedding(self.env.observation_space.n, 64).to(self.device)
            self.state_dim = 64
        else:
            self.state_dim = env.observation_space.shape[0]

        if isinstance(self.env.action_space, Discrete):
            self.action_embedding = torch.nn.Embedding(self.env.action_space.n, 64).to(self.device)
            self.action_dim = 64
        else:
            self.action_dim = env.action_space.shape[0]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim + 1, 256),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5) if dropout else torch.nn.Identity(),
            torch.nn.Linear(256, 1),
            torch.nn.Flatten(-2)
        ).to(self.device)

    def forward(self, time_steps, states, actions):
        if isinstance(self.env.observation_space, Discrete):
            states = self.state_embedding(states.long())
        if isinstance(self.env.action_space, Discrete):
            actions = self.action_embedding(actions.long())

        time_state_actions = torch.cat((time_steps, states, actions), -1)
        return self.net(time_state_actions)

    def to(self, device):
        self.device = device
        return super().to(device)


class VarQNetwork(torch.nn.Module):
    def __init__(self, env: gym.Env, device: torch.device, dropout: bool = False, max_time_step=None):
        super().__init__()
        self.env = env
        self.device = device
        self.max_time_step = max_time_step

        self.q = QNetwork(self.env, self.device, dropout)
        # self.squared_q = QNetwork(self.env, self.device, dropout)  # Deprecated

    def forward(self, time_steps, states, actions):
        if self.max_time_step is not None:
            time_steps = self.max_time_step - time_steps

        # # don't consider time_step.
        # time_steps[:] = 0

        q = self.q(time_steps, states, actions)
        # v = self.squared_q(time_steps, states, actions), # Deprecated
        v = torch.zeros_like(q)

        return q, v

    def to(self, device):
        self.device = device
        return super().to(device)
