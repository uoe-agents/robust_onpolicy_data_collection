import gym
import torch
from gym.spaces import Discrete, Box
from torch.nn.functional import one_hot, softplus


class OneHotLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return one_hot(x)


class Actor(torch.nn.Module):
    def __init__(self, env: gym.Env, device: torch.device):
        super().__init__()
        self.env = env
        self.device = device

        self.discrete_state = isinstance(self.env.observation_space, Discrete)
        self.discrete_action = isinstance(self.env.action_space, Discrete)

        hidden_dim = 64

        if self.discrete_state:
            self.state_dim = self.env.observation_space.n
            backbone = torch.nn.Identity()
        else:
            self.state_dim = env.observation_space.shape[0]
            backbone = torch.nn.Sequential(
                torch.nn.BatchNorm1d(self.state_dim),
                torch.nn.Linear(self.state_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )

        if self.discrete_action:
            self.action_dim = self.env.action_space.n
            output_layer = torch.nn.Sequential(
                torch.nn.Embedding(self.state_dim, self.action_dim) if self.discrete_state else torch.nn.Linear(hidden_dim, self.action_dim),
                torch.nn.Softmax(-1)
            )
        else:
            assert self.env.action_space.shape[0] == 1, 'Only support action_dim = 1.'
            self.action_dim = self.env.action_space.shape[0]
            self.action_scale = torch.from_numpy(self.env.action_space.high)[0]
            output_layer = torch.nn.Embedding(self.state_dim, self.action_dim * 2) if self.discrete_state else torch.nn.Linear(hidden_dim, self.action_dim * 2)

        # !: if change, remember to check ROS and RIS, both of which involve update parts of parameters
        self.model = torch.nn.Sequential(
            backbone,
            output_layer
        ).to(self.device)

    def forward(self, states, logit=False):
        states = states.to(self.device)
        states = states.long() if self.discrete_state else states

        if logit:
            outputs = self.model[0](states)
            return self.model[1][0](outputs)  # skip softmax

        outputs = self.model(states)
        if not self.discrete_action:
            outputs = outputs.view(outputs.size(0), self.action_dim, 2)
            outputs = torch.stack((
                # outputs[..., 0],
                torch.tanh(outputs[..., 0]) * self.action_scale,

                # outputs[..., 1].abs(),
                # outputs[..., 1].pow(2),
                # outputs[..., 1].exp(),
                softplus(outputs[..., 1]),
            ), -1)

        return outputs

    def to(self, device):
        self.device = device
        return super().to(device)

    def first_layer(self):
        return self.model[0][1]

    def last_layer(self):
        return self.model[1]
