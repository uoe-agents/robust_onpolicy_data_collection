import gym
import copy
import torch
import random
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from policy import REINFORCE
from models import Actor
from commons import Policy
from commons import AdaptiveDefaultDict, finished_trajectories, Global


class BehaviorCloner:
    def __init__(self, pi: REINFORCE, **kwargs):
        self.lr = kwargs['bc_lr']
        self.batch_size = kwargs['bc_batch']
        self.num_epochs = kwargs['bc_epochs']
        self.device = torch.device(kwargs['device'])
        self.pi = pi

    def clone(self, trajectories):
        # trajectories = finished_trajectories(trajectories)
        # if not trajectories:
        #     return
        pi = copy.deepcopy(self.pi)

        trainings = []
        for trajectory in trajectories:
            for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
                trainings.append(([0], s, a))

        trainings = TensorDataset(*(torch.Tensor(tensors) for tensors in zip(*trainings)))
        num_epochs = max(self.num_epochs, 1000 // (len(trainings) // self.batch_size + 1))  # min_update_times = 1000
        optimizer = Adam(pi.model.last_layer().parameters(), lr=self.lr)
        # optimizer = Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, num_epochs)
        for epoch in range(num_epochs):
            losses = []
            pi.model.eval()  # fix BN
            for items in DataLoader(trainings, batch_size=self.batch_size, shuffle=True):
                time_steps, states, actions = (item.to(self.device) for item in items)
                loss = -pi.action_probs(states, actions, require_grad=True).log().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            lr_scheduler.step()
            Global.setdefault('0004_1', []).append(np.mean(losses))

        fig = plt.figure(figsize=(8, 6))
        plt.subplot(111)
        plt.plot(Global['0004_1'])
        plt.savefig('results/0004.jpg')
        plt.close(fig)

        return pi
