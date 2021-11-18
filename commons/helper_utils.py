import os
import sys
import time
import gym
import time
import torch
import pickle
import hashlib
import traceback
import numpy as np
from gym.spaces import Discrete
from torch.optim import Optimizer
from collections import defaultdict

Global = {}


class AvgAccumGradOptimizer(Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        # self.step_i = 0

        super().__init__(params, {'lr': lr, 'avg_grad': 0})

    @torch.no_grad()
    def step(self, old_weight: float, new_weight: float, update_avg_grad=True):
        grad_sum = 0
        grad_num = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]
                avg_grad = state.setdefault('avg_grad', 0)
                # avg_grad = self.step_i / (self.step_i + 1) * avg_grad + d_p / (self.step_i + 1)
                avg_grad = (old_weight * avg_grad + new_weight * d_p) / (old_weight + new_weight)
                if update_avg_grad:
                    state['avg_grad'] = avg_grad
                if group['lr'] > 0:
                    p.add_(avg_grad, alpha=-group['lr'])

                grad_sum += avg_grad.abs().sum().item()
                # grad_sum += avg_grad.pow(2).sum().item()
                grad_num += avg_grad.numel()

        # if update:
        #     self.step_i += 1

        return grad_sum / grad_num
        # return (grad_sum / grad_num) ** 0.5


def compute_avg_grad(trajectories, pi_e, eval_steps):
    from policy import REINFORCE
    pi_e: REINFORCE

    start_time = time.time()
    avg_grads = []
    states = []
    actions = []
    for trajectory in trajectories:
        for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
            states.append(s)
            actions.append(a)
    log_pie = pi_e.action_probs(states, actions, require_grad=True).log()

    optimizer = AvgAccumGradOptimizer(pi_e.model.parameters(), lr=0.)
    from_step = 0
    for to_step in eval_steps:
        loss = -log_pie[from_step: to_step].mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        avg_grad = optimizer.step(old_weight=from_step, new_weight=to_step - from_step)
        avg_grads.append(avg_grad)

    print('computing avg-grad time:', time.time() - start_time)
    return avg_grads


def compute_cross_entropy(trajectories, pi_e, normalized=True):
    start_time = time.time()
    from policy import REINFORCE
    pi_e: REINFORCE

    states = []
    actions = []
    for trajectory in trajectories:
        for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
            states.append(s)
            actions.append(a)
    action_dists = pi_e.action_dists(states)
    true = action_dists.entropy().numpy()
    predict = -action_dists.log_prob(torch.Tensor(actions)).numpy()
    true = np.cumsum(true) / np.arange(1, len(true) + 1)
    predict = np.cumsum(predict) / np.arange(1, len(predict) + 1)

    if normalized:
        predict /= true

    print('computing cross-entropy time:', time.time() - start_time)
    return predict


def compute_kl_divergence(trajectories, pi_e, pi_D):
    start_time = time.time()

    states = []
    actions = []
    for trajectory in trajectories:
        for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
            states.append(s)
            actions.append(a)

    # print('compute kl_divergence time:', time.time() - start_time)
    return (pi_D.action_probs(states, actions).log().mean() - pi_e.action_probs(states, actions).log().mean()).item()


def trace(normalization=True, **kwargs):
    for k, v in kwargs.items():
        if normalization:
            v = np.array(v) / sum(v)
        Global.setdefault('cache', {}).setdefault(k, []).append(v)


def finished_trajectories(trajectories):
    return [
        trajectory for trajectory in trajectories if len(trajectory) and trajectory[-1][5]  # done
    ]


class DefaultHasher:
    def code(self, x):
        if isinstance(x, np.ndarray):
            return tuple(x.reshape(-1))

        if isinstance(x, tuple) or isinstance(x, list):
            return tuple(self.code(i) for i in x)

        return x


class SimHasher(DefaultHasher):
    def __init__(self, state_dim, code_dim, state_mean=0, state_std=1, time_normalization=1) -> None:
        super().__init__()
        self.A = np.random.normal(0, 1, (state_dim + 1, code_dim))
        self.time_normalization = time_normalization
        self.state_mean = state_mean
        self.state_std = state_std
        self.cache = AdaptiveDefaultDict()

    def code(self, x):
        if isinstance(x, tuple) and len(x) > 1:
            s, t = x[:2]
            if isinstance(s, np.ndarray) and s.shape[0] == self.A.shape[0] - 1 and isinstance(t, int):
                st = self.cache.setdefault((s, t), None)
                if st is None:
                    st = state_time((s - self.state_mean) / self.state_std, t / self.time_normalization)
                    st = np.sign(st @ self.A).astype(int)
                    self.cache[(s, t)] = st
                x = (st,) + x[2:]

        return super().code(x)

    def __str__(self):
        return f'SimHasher: A: {self.A.shape}, state_mean: {self.state_mean}, state_std: {self.state_std}, time_norm: {self.time_normalization}'


class ZeroHasher(DefaultHasher):
    def __init__(self) -> None:
        super().__init__()

    def code(self, x):
        return 0


default_hasher = DefaultHasher()


class AdaptiveDefaultDict(defaultdict):
    def __init__(self, default_factory=None, hasher: DefaultHasher = default_hasher):
        super().__init__(default_factory)
        self.hasher = hasher

    def __getitem__(self, k):
        k = self.hasher.code(k)
        return super().__getitem__(k)

    def __setitem__(self, k, v) -> None:
        k = self.hasher.code(k)
        super().__setitem__(k, v)

    def setdefault(self, __key, __default=...):
        __key = self.hasher.code(__key)
        return super().setdefault(__key, __default)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(self.hasher.code(o))

    def copy(self):
        import copy
        result = copy.deepcopy(self)
        result.hasher = self.hasher
        return result


class AdaptiveSet(set):
    def __init__(self, iterable=(), hasher: DefaultHasher = default_hasher) -> None:
        super().__init__()
        self.hasher = hasher
        for i in iterable:
            self.add(i)

    def add(self, element) -> None:
        element = self.hasher.code(element)
        super().add(element)


def state_time(s, t):
    return np.concatenate((s, [t]))


def make_dirs(path):
    if not os.path.exists(path):
        print(f'making dirs {path}')
        os.makedirs(path)


def save(file, obj):
    make_dirs(os.path.dirname(file))
    while True:
        try:
            with open(file, "wb+") as f:
                pickle.dump(obj, f)
                return print(f'saved {file}')
        except:
            print(f'error from saving {file}', file=sys.stderr)
            traceback.print_exc()
            time.sleep(1)


def load(file):
    # print(f'loading {file} (last modified: {time.asctime(time.localtime(os.path.getmtime(file)))})')
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'load {file} error.')
        raise e


def md5(s: str):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()


def cache_run(method: callable, cache_file=None):
    if cache_file is None:
        return method()

    if os.path.exists(cache_file):
        print(f'file {cache_file} exists, load cache.')
        result = load(cache_file)
    else:
        result = method()
        try:
            if not os.path.exists('debug'):
                save(cache_file, result)
        except:
            traceback.print_exc()
    return result


def test():
    a = AdaptiveDefaultDict(int)
    a[1] += 10
    print(a[[1, 2]])
    print(a.setdefault([1, 3], 123))
    print(a)

    b = AdaptiveDefaultDict(lambda: AdaptiveDefaultDict(int))
    b[1][2] += 1
    print(b)
    print(b[1][2])
    print(b[2][2])

    c = AdaptiveDefaultDict(int, SimHasher(4, 3, time_normalization=12))
    d = np.array([1.12, 3.1234, 423, 1])
    c[d, 0, 0] += 1
    c[d, 1235, 0] += 1
    e = c.copy()
    assert e.hasher is c.hasher
    assert e.hasher.time_normalization == c.hasher.time_normalization
    print(c)
    print(e)
    c[d, 0] += 10
    c[d, 1235, 123] += 10
    e[d] += 20
    print(c)
    print(e)
    c.clear()
    print(c)

    c[[(d, 1, 2, 3), (d, 1, 2, 3)]] += 123
    print(c)
    print(c.hasher)
    print(1, c.hasher)
    for k, v in c.items():
        print(k, v, c[k])

    c = AdaptiveDefaultDict(int, SimHasher(4, 0))
    d1 = np.array([1.12, 3.1234, 423, 1])
    d2 = np.array([1.12, 3.1234, -423, 1])
    c[d1, 0, 0] += 1
    c[d2, 0, 0] += 1
    print(c)

    c = AdaptiveDefaultDict(int, SimHasher(4, 1))
    d1 = np.array([1.12, 3.1234, 423, 1])
    d2 = np.array([1.12, 3.1234, -423, 1])
    c[d1, 0, 0] += 1
    c[d2, 0, 0] += 1
    c[((d1, 0, 0), (d2, 0, 0))] += 1
    c[((d1, 0, 0), (d2, 0, 0))] += 1
    print(c)


if __name__ == '__main__':
    test()
