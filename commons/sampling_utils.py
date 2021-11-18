import sys
import gym
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from commons import Policy, DataCollector
from commons import DefaultHasher, default_hasher, AdaptiveSet, AdaptiveDefaultDict


def argmax(actions, pi_e, weights):
    pi_b = np.isclose(weights, np.max(weights))
    pi_b = pi_b / pi_b.sum()

    return sample(actions, pi_e, pi_b)


def sample(actions, pi_e, pi_b, re_weight=True):
    pi_b = pi_b.clip(1e-10)
    pi_b = pi_b / pi_b.sum()
    action_i = np.random.choice(range(len(actions)), p=pi_b)
    if re_weight:
        return actions[action_i], pi_e[action_i], pi_b[action_i]
    else:
        return actions[action_i], pi_e[action_i], pi_e[action_i]


def collecting_data(
        env: gym.Env,
        max_time_step: int,
        collector: DataCollector,
        eval_steps: list,
        render: bool = False,
        normalization: float = None,
        hasher: DefaultHasher = default_hasher,
        max_trajectories: int = np.inf,
        seed: int = None
):
    env.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    step = 0
    t = 0
    s = env.reset()
    trajectory = []
    trajectories = [trajectory]

    sa_pairs = AdaptiveSet(hasher=hasher)
    sta_pairs = AdaptiveSet(hasher=hasher)
    pr_h = AdaptiveDefaultDict(hasher=hasher)
    sa_counts = []
    sta_counts = []
    prh_sum = []
    avg_pie = 0
    avg_pib = 0
    start_time = time.time()

    while step < eval_steps[-1]:
        if render:
            env.render("human")

        a, pie_a, pib_a = collector.act(t, s)
        s_, r, d, _ = env.step(a)
        d = d or t == max_time_step - 1
        if normalization is not None:
            r /= normalization
        trajectory.append((t, s, a, r, s_, d, pie_a, pib_a))
        collector.update(t, s, a, r, s_, d, pie_a, pib_a, trajectories=trajectories)
        avg_pie = (avg_pie * step + pie_a) / (step + 1)
        avg_pib = (avg_pib * step + pib_a) / (step + 1)
        step += 1
        t += 1

        if d:
            pr_h[tuple((t, a) for t, s, a, r, s_, d, pie_a, pib_a in trajectory)] = np.prod([pie_a for t, s, a, r, s_, d, pie_a, pib_a in trajectory])

        sa_pairs.add((s, 0, a))
        sta_pairs.add((s, t, a))
        sa_counts.append(len(sa_pairs))
        sta_counts.append(len(sta_pairs))
        prh_sum.append(sum(pr_h.values()))

        if step in eval_steps:
            print(f"Step: {step}, Trajectories: {len(trajectories)}, prh_sum:{prh_sum[-1]:.2f}, state-action: {len(sa_pairs)}, state-time-action: {len(sta_pairs)}, avg_pie: {avg_pie:.2f}, avg_pib: {avg_pib:.2f}, time: {int(time.time() - start_time)}s")

        if d:
            if len(trajectories) == max_trajectories:
                break

            t = 0
            s = env.reset()
            trajectory = []
            trajectories.append(trajectory)
        else:
            s = s_

    return trajectories, sa_counts, sta_counts, prh_sum


def mc_estimation_true_value(
        env: gym.Env,
        max_time_step: int,
        policy: Policy,
        gamma: float = 1.,
        n_trajectory: int = 10000,
        show: bool = True,
        render: bool = False,
):
    from data_collection_strategy import OnPolicySampler
    collector = OnPolicySampler(policy)

    n = 0
    estimation = 0.
    estimations = []
    returns = []
    start_time = time.time()

    s = env.reset()
    g = 0
    time_step = 0
    step = 0
    avg_pie = 0
    while True:
        if render:
            env.render()

        a, pie_a, _ = collector.act(time_step, s)
        s_, r, d, _ = env.step(a)
        d = d or time_step == max_time_step - 1
        g += r * (gamma ** time_step)
        avg_pie = (avg_pie * step + pie_a) / (step + 1)
        step += 1

        if d:
            returns.append(g)
            estimation = n / (n + 1) * estimation + g / (n + 1)
            estimations.append(estimation)
            n += 1

            if show and n % 100 == 0:
                import sys
                sys.stdout.write('\b' * 1000 + f'tr {n}, time: {int(time.time() - start_time)}s, avg_step: {step / n}, step {time_step}, returns: {g}, avg_pie: {avg_pie}, estimation and variance: {estimation}, {np.var(returns)}')
                sys.stdout.flush()

            if n == n_trajectory:
                if show:
                    plt.hlines(1, 0, len(estimations), linestyles='--', alpha=0.2)
                    plt.plot(np.array(estimations) / estimation)
                    plt.ylim(0.9, 1.1)
                    plt.show()

                return estimation, np.var(returns), step / n, returns

            s = env.reset()
            g = 0
            time_step = 0
        else:
            s = s_
            time_step += 1
