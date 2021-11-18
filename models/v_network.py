import torch
import gym
from gym.spaces import Discrete


class VNetwork(torch.nn.Module):
    def __init__(self, env: gym.Env, device: torch.device, max_time_step: int = None, dropout: bool = False):
        super().__init__()
        self.env = env
        self.device = device
        self.max_time_step = max_time_step

        if isinstance(self.env.observation_space, Discrete):
            self.state_embedding = torch.nn.Embedding(self.env.observation_space.n, 64).to(self.device)
            self.state_dim = 64
        else:
            self.state_dim = env.observation_space.shape[0]

        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.state_dim + 1),
            torch.nn.Linear(self.state_dim + 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5) if dropout else torch.nn.Identity(),
            torch.nn.Linear(256, 1),
            torch.nn.Flatten(-2)
        ).to(self.device)

    def forward(self, time_steps, states):
        if isinstance(self.env.observation_space, Discrete):
            states = self.state_embedding(states.long())

        # if self.max_time_step is not None:
        #     time_steps = self.max_time_step - time_steps
        #     time_steps /= self.max_time_step

        time_states = torch.cat((time_steps, states), -1)
        return self.net(time_states)

    def to(self, device):
        self.device = device
        return super().to(device)
