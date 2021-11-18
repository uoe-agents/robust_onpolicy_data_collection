import sys
import gym
import torch
import numpy as np
from torch.optim import Adam
from collections import defaultdict
from gym.spaces import Discrete, Box

from experiments.gridworld import GridWorld
from experiments.multibandit import MultiBandit
from experiments.continuous_cartpole import ContinuousCartPoleEnv

from commons import Policy
from models import Actor, VNetwork


class REINFORCE(Policy):

    def __init__(self, env: gym.Env, gamma: float, file: str = None, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.device = torch.device(kwargs['device'])
        self.model = Actor(self.env, self.device)
        self.lr = kwargs.get('lr', 1e-2)
        if file:
            self.load(file)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-5)

    def save(self, file):
        print('saving', file)
        self.env.close()
        torch.save(self.model, file)

    def load(self, file):
        print('loading', file)
        self.model = torch.load(file).to(self.device)

    def update(self, trajectories):
        self.model.train()

        trainings = []
        for trajectory in trajectories:
            g = 0
            for t, s, a, r, d, s_ in reversed(trajectory):
                g = r + self.gamma * g
                trainings.append(([t], s, a, r, s_, d, g))
        time_steps, states, actions, rewards, next_states, dones, gs = (torch.Tensor(tensors).to(self.device) for tensors in zip(*trainings))

        advantages = gs
        advantages = advantages / advantages.abs().max()  # reduce variance, but need batch.

        action_dists = self.action_dists(states, require_grad=True, min_sigma=0.00, fix_bn=False)
        log_action_probs = action_dists.log_prob(actions)
        pg_loss = (-advantages * log_action_probs.squeeze(-1)).mean()

        self.optimizer.zero_grad()
        pg_loss.backward()
        self.optimizer.step()

        if np.random.random() < .05:
            pie_info = f'max_p: {action_dists.probs.max(1)[0].mean()}' if isinstance(action_dists, torch.distributions.Categorical) else f'sigma: {action_dists.scale.mean()}'
            print(f'\ta_mean: {advantages.mean().item():.2f}, a_abs_mean: {advantages.abs().mean().item():.2f}, pg_loss: {pg_loss.item():.2f}, {pie_info}')

    def logit_dist(self, state):
        assert isinstance(self.env.action_space, Discrete)
        self.model.eval()
        state = torch.Tensor([state])
        with torch.no_grad():
            return self.model(state, logit=True)[0].numpy()

    def action_dists(self, states, require_grad=False, sigma_grad=True, min_sigma=0., fix_bn=True, _single_state=False):
        if fix_bn:
            self.model.eval()

        states = states if isinstance(states, torch.Tensor) else torch.FloatTensor(states)
        with torch.enable_grad() if require_grad else torch.no_grad():
            outputs = self.model(states)
        outputs = outputs[0] if _single_state else outputs

        if isinstance(self.env.action_space, Discrete):
            action_dists = torch.distributions.Categorical(outputs)
        else:
            action_means = outputs[..., 0]
            action_scales = outputs[..., 1]
            action_dists = torch.distributions.Normal(action_means, (action_scales if sigma_grad else action_scales.detach()) + min_sigma * action_scales.detach().abs())

        return action_dists

    def action_dist(self, state, **kwargs):
        return self.action_dists([state], _single_state=True, **kwargs)

    def action_probs(self, states, actions, **kwargs):
        action_dists = self.action_dists(states, **kwargs)
        actions = actions if isinstance(actions, torch.Tensor) else torch.Tensor(actions)
        action_probs = action_dists.log_prob(actions).exp()
        return action_probs

    def action_prob(self, state, action, require_grad=False, **kwargs):
        action_prob = self.action_probs([state], [action], require_grad=require_grad, **kwargs)[0]
        return action_prob if require_grad else action_prob.item()

    def sample(self, state, min_sigma=0.):
        action_dist = self.action_dist(state, min_sigma=min_sigma)
        action = action_dist.sample()
        # action = action_dist.mean
        action = action.tolist()
        action_prob = action_dist.log_prob(torch.Tensor([action])[0]).exp().item()
        return action, action_prob

    def discrete_action_dist(self, state, n_action=9):  # no grad.
        action_dist = self.action_dist(state)
        if isinstance(action_dist, torch.distributions.Categorical):
            action_probabilities = action_dist.probs.numpy()
            actions = list(range(len(action_probabilities)))
        else:
            actions = action_dist.sample((n_action,)).tolist()
            action_probabilities = np.ones(n_action) / n_action

            # # raise NotImplementedError
            # actions = action_dist.icdf(torch.arange(1, n_action + 1) / (n_action + 1))
            # actions_shift = (actions[1:] + actions[:-1]) / 2
            # actions_cdf = action_dist.cdf(actions_shift)
            # actions_cdf = torch.cat((actions_cdf, torch.Tensor([1])))
            # actions_cdf[1:] -= actions_cdf[:-1].clone()
            # actions = actions.unsqueeze(-1).tolist()
            # action_probabilities = actions_cdf.numpy()

        assert np.isclose(action_probabilities.sum(), 1)
        return actions, action_probabilities


def count_actions(actions):
    action_counts = defaultdict(int)
    for a in actions:
        if 'item' in dir(a):
            a = a.item()
        action_counts[a] += 1

    return {
        k: action_counts[k]
        for k in set(action_counts.keys())
    }


def play_episode(env: gym.Env, max_time_step: int, actor: REINFORCE, render=False):
    t = 0
    d = False
    returns = 0
    s = env.reset()
    trajectory = []
    while not d:
        if render:
            env.render()

        a, _ = actor.sample(s, min_sigma=0.00)
        a = [np.clip(a[0], env.action_space.low[0], env.action_space.high[0])] if isinstance(env.action_space, Box) else a

        s_, r, d, _ = env.step(a)
        d = d or t == max_time_step - 1
        returns += r

        # learning trick for MountainCar
        if exp_name == 'MountainCar' or exp_name == 'MountainCarContinuous':
            d = d or (t == 200)  # max_time_step
            r = (max([t[1][0] for t in trajectory]) + 0.5) * 100 if d else r  # higher reward if closer to goal

        trajectory.append((t, s, a, r, d, s_))
        s = s_
        t += 1

    return trajectory, returns


def train():
    print(f'training: {exp_name}, config: {Configs[exp_name]}')
    env, max_time_step, gamma, lr, save_episodes = Configs[exp_name]

    torch.manual_seed(0)
    np.random.seed(0)
    env.seed(0)
    env.action_space.seed(0)

    training_episodes = max(save_episodes) + 1
    actor = REINFORCE(env, gamma, device='cpu', lr=lr)
    batch = []
    batch_size = 16
    for episode in range(training_episodes):
        if episode in save_episodes and SAVE:
            actor.save(f'policy_models/model_{exp_name}_{episode}.pt')

        tr, returns = play_episode(env, max_time_step, actor, render=False)
        batch.append(tr)

        if len(batch) == batch_size:
            actor.update(batch)
            batch.clear()

        if episode % 100 == 0:
            eval_trs, eval_returns = zip(*[play_episode(env, max_time_step, actor, render=RENDER and i == 0) for i in range(100)])
            print(
                f'episode {episode}\t'
                f'training_tot_returns={returns:.2f}, '
                f'training_dis_returns={sum([t[3] * gamma ** i for i, t in enumerate(tr)]):.2f}\t'
                f'eval_tot_returns={np.mean(eval_returns):.2f}({np.min(eval_returns):.2f}, {np.max(eval_returns):.2f})\t'
                f'eval_dis_returns={np.mean([sum([t[3] * gamma ** i for i, t in enumerate(tr)]) for tr in eval_trs]):.2f}\t'
            )


def get_exp_config(name):
    env_ = None
    if name.startswith('MultiBandit'):
        randomness = name.split('MultiBandit')[-1]
        name = 'MultiBandit'
        if randomness:
            mean_factor, scale_factor = randomness.split('M')[-1].split('S')
            env_ = MultiBandit(int(mean_factor), int(scale_factor))

    if name.startswith('GridWorld'):
        randomness = name.split('GridWorld')[-1]
        name = 'GridWorld'
        if randomness:
            mean_factor, scale_factor = randomness.split('M')[-1].split('S')
            env_ = GridWorld(int(mean_factor), int(scale_factor))

    env, max_time_step, gamma, _, _ = Configs[name]

    if env_ is not None:
        env = env_

    return env, max_time_step, gamma, _, _


Configs = {
    'MultiBandit': (
        MultiBandit(),  # env
        MultiBandit.MaxStep,  # max_time_step
        MultiBandit.gamma,  # gamma
        1e-2,  # lr
        [5000],  # save_episodes
    ),
    'GridWorld': (
        GridWorld(),  # env
        GridWorld.MaxStep,  # max_time_step
        GridWorld.gamma,  # gamma
        1e-2,  # lr
        [5000],  # save_episodes
    ),
    'Taxi': (
        gym.make('Taxi-v3'),  # env
        200,  # max_time_step
        0.99,  # gamma
        1e-2,  # lr
        [10000],  # save_episodes
    ),
    'CartPole': (
        gym.make('CartPole-v0'),  # env
        200,  # max_time_step
        0.99,  # gamma
        1e-3,  # lr
        [1000, ],  # save_episodes
    ),
    'Acrobot': (
        gym.make('Acrobot-v1'),  # env
        500,  # max_time_step
        0.99,  # gamma
        1e-3,  # lr
        [5000, ],  # save_episodes
    ),
    'CartPoleContinuous': (
        ContinuousCartPoleEnv(),  # env
        200,  # max_time_step
        0.99,  # gamma
        1e-4,  # lr
        [3000],  # save_episodes
    ),
}

if __name__ == '__main__':
    SAVE = True
    RENDER = sys.platform != 'linux'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='MultiBandit')
    args = parser.parse_args()
    exp_name = args.exp_name

    # exp_name = "MultiBandit"
    # exp_name = "GridWorld"
    # exp_name = "Taxi"
    # exp_name = "CartPole"
    # exp_name = "CartPoleContinuous"
    # exp_name = "MountainCar"
    # exp_name = "MountainCarContinuous"
    # exp_name = "Pendulum"
    # exp_name = "Acrobot"
    # exp_name = "Acrobot"

    train()

"""
python policy/reinforce.py --exp_name CartPoleContinuous
"""
