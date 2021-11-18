import os
import sys
import time
import torch
import numpy as np
from gym.spaces import Discrete
from typing import List, Tuple, Type, Dict

from policy import *
from value_estimation import *
from data_collection_strategy import *

from experiments.policy_info import get_policy_info
from commons import DataCollector, ValueEstimator
from commons import collecting_data, mc_estimation_true_value
from commons import AdaptiveDefaultDict, DefaultHasher, SimHasher, ZeroHasher
from commons import Global, save, load, md5, cache_run, finished_trajectories
from commons import compute_cross_entropy, compute_avg_grad, compute_kl_divergence


def yield_data(trs, eval_steps):
    step = 0
    data = []

    if step in eval_steps:
        yield step, data

    for tr in trs:
        d = []
        data.append(d)
        for transition_tuple in tr:
            d.append(transition_tuple)
            step += 1
            if step in eval_steps:
                yield step, data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='0,1')
    parser.add_argument('--suffix', type=str, default='0')
    parser.add_argument('--policy_name', type=str, default='MultiBandit_5000')
    parser.add_argument('--collectors', type=str, default='OnPolicySampler')
    parser.add_argument('--estimators', type=str, default='MonteCarlo')
    parser.add_argument('--eval_steps', type=str, default='1000,2000')
    parser.add_argument('--ros_lr', type=float, default=1e+2)
    parser.add_argument('--ros_beta', type=float, default=0.1)
    parser.add_argument('--ros_epsilon', type=float, default=1.)
    parser.add_argument('--ros_n_action', type=int, default=9)
    parser.add_argument('--ros_dynamic_lr', action='store_true')
    parser.add_argument('--ros_first_layer', action='store_true')
    parser.add_argument('--ros_only_weight', action='store_true')
    parser.add_argument('--ros_increasing_epsilon', action='store_true')
    parser.add_argument('--bpg_lr', type=float, default=1e-4)
    parser.add_argument('--bpg_k', type=int, default=10)
    parser.add_argument('--simhash_dim', type=int, default=10)
    parser.add_argument('--combined_trajectories', type=int, default=0)
    parser.add_argument('--combined_ignore_history', action='store_true')  # not used.
    parser.add_argument('--combined_ops_epsilon', type=float, default=0.1)
    parser.add_argument('--combined_pre_estimate', action='store_true')

    args = parser.parse_args()

    suffix = args.suffix
    policy_name = args.policy_name
    seeds = [int(i) for i in args.seeds.split(',')]
    eval_steps = [int(i) for i in args.eval_steps.split(',')]
    build_collectors = [globals()[i] for i in args.collectors.split(',')]
    build_estimators = [globals()[i] for i in args.estimators.split(',')]
    ros_lr = args.ros_lr
    ros_beta = args.ros_beta
    ros_epsilon = args.ros_epsilon
    ros_n_action = args.ros_n_action
    ros_first_layer = args.ros_first_layer
    ros_only_weight = args.ros_only_weight
    ros_dynamic_lr = args.ros_dynamic_lr
    ros_increasing_epsilon = args.ros_increasing_epsilon
    bpg_lr = args.bpg_lr
    bpg_k = args.bpg_k
    simhash_dim = args.simhash_dim
    combined_ignore_history = args.combined_ignore_history
    combined_ops_epsilon = args.combined_ops_epsilon
    combined_trajectories = args.combined_trajectories
    combined_pre_estimate = args.combined_pre_estimate

    ############################################ code for local debug  ############################################
    if os.path.exists('debug'):
        suffix = '1'
        seeds = [0, 1]
        build_collectors = [OnPolicySampler]
        build_estimators = [MonteCarlo]
        policy_name = 'GridWorld_5000'
        eval_steps = [50 * 2 ** i for i in range(4)]
    ############################################ code for local debug  ############################################

    if combined_trajectories > 0:
        eval_steps.insert(0, 0)

    result_file = f'results/seeds/{policy_name}_{suffix}_{seeds[0]}_{seeds[1]}'
    if os.path.exists(result_file):
        print(f'file {result_file} exists, skip running.')
        exit()

    ############################################ collector config ############################################
    config = {
        'max_data_collection_step': eval_steps[-1],
        'simhash_dim': simhash_dim,

        # off-policy data config
        "ops_epsilon": combined_ops_epsilon,
        "combined_trajectories": combined_trajectories,
        "combined_ignore_history": combined_ignore_history,
    }
    config.update({
        # robust on-policy strategy config
        'ros_lr': ros_lr,
        'ros_beta': ros_beta,
        'ros_first_layer': ros_first_layer,
        'ros_only_weight': ros_only_weight,
        'ros_dynamic_lr': ros_dynamic_lr,
        'ros_epsilon': ros_epsilon,
        'ros_n_action': ros_n_action,
        'ros_increasing_epsilon': ros_increasing_epsilon,
    }) if any([c in build_collectors for c in (RobustOnPolicySampler, RobustOnPolicyActing)]) else None
    config.update({
        # behavior policy gradient config
        "bpg_lr": bpg_lr,
        "bpg_k": bpg_k,
    }) if any([c in build_collectors for c in (BehaviorPolicyGradient,)]) else None
    collector_config_md5 = md5(str(sorted(config.items())))  # for data collection cache.

    ############################################ estimator config ############################################
    config.update({
        "device": "cpu",

        # fitted Q-evaluation config
        "fqe_tau": 1.,
        # "fqe_tau": .5,
        "fqe_epochs": 100,
        "fqe_lr": 1e-3,
        "fqe_batch": 5120,
        "fqe_re_weight": False,

        # model-based config
        'mb_lr': 1e-3,
        'mb_epochs': 100,
        'mb_batch': 5120,
        'mb_trs': 100000,

        # behavior cloning config
        "bc_lr": 1e-3,
        "bc_epochs": 100,
        "bc_batch": 5120,
    })

    exp_name, training_episodes, avg_step, true_value, true_variance, reward_range = get_policy_info(policy_name)
    env, max_time_step, gamma, _, _ = get_exp_config(exp_name)
    pi_e = REINFORCE(env, gamma, file=f'policy_models/model_{policy_name}.pt', **config)
    if isinstance(env.observation_space, Discrete):
        hasher = DefaultHasher()
    elif isinstance(env.action_space, Discrete):
        hasher = SimHasher(env.observation_space.shape[0], simhash_dim, time_normalization=max_time_step)
    else:
        # haven't implemented for continuous action.
        hasher = ZeroHasher()

    config.update({
        "gamma": gamma,
        "max_time_step": max_time_step,
        "reward_range": np.array(reward_range) / true_value,  # normalized
    })
    results: Dict[str] = {
        'suffix': suffix,
        "exp_name": exp_name,
        "training_episodes": training_episodes,
        "eval_steps": eval_steps,
        "true_value": 1,  # normalized
        "true_variance": true_variance / true_value / true_value,  # normalized
        "config": config.copy(),
    }
    config.update({
        "env": env,
        "pi_e": pi_e,
        "hasher": hasher,
    })
    Global['true_variance'] = true_variance / true_value / true_value
    Global['ir_weights'] = exp_name, build_collectors
    [print(k, v) for d in (config, results) for k, v in d.items()]

    # start running
    for seed in range(*seeds):
        print(f'\nrunning seed {seed}')
        config['seed'] = seed
        experiments: List[List[DataCollector, List[ValueEstimator]]] = [[collector(**config), [estimator(**config) for estimator in build_estimators]] for collector in build_collectors]

        #################################### off-policy data ####################################
        op_data = []
        if combined_trajectories > 0:
            op_data = collecting_data(env, max_time_step, EpsilonPolicySampler(**config), [np.inf], normalization=true_value, hasher=hasher, max_trajectories=combined_trajectories, seed=seed + 1000)[0]
        op_data_steps = sum([len(tr) for tr in op_data])
        print(f'\nstart from off-policy data with {len(op_data)} trajectories, {op_data_steps} steps.')

        if not combined_ignore_history and len(op_data):
            print(f'initializing strategies with off-policy data ...')
            for _, data in yield_data(op_data, list(range(1, op_data_steps + 1))):
                t, s, a, r, s_, d, pie_a, pib_a = data[-1][-1]
                for collector, _, in experiments:
                    collector.update(t, s, a, r, s_, d, pie_a, pib_a, data)

        op_value = None
        combined_pre_estimate = combined_pre_estimate and (combined_trajectories > 0)
        if combined_pre_estimate:
            print()
            assert len(op_data) > 0
            op_mc = MonteCarlo(**config).estimate(op_data)
            print(f'MC for off-policy data is {op_mc}.')
            # pi_D = BehaviorCloner(pi_e, **config).clone(op_data)
            # op_wris = WeightedRegressionImportanceSampling(**config).estimate(op_data, pi_D)
            # print(f'WRIS for off-policy data is {op_wris}.')
            op_wis = WeightedImportanceSampling(**config).estimate(op_data)
            print(f'WIS for off-policy data is {op_wis}.')
            op_ois = OrdinaryImportanceSampling(**config).estimate(op_data)
            print(f'OIS for off-policy data is {op_ois}.')

            op_value = op_wis

        ########################################################################################

        behavior_cloner = BehaviorCloner(pi_e, **config) if any(reg in build_estimators for reg in (RegressionImportanceSampling, WeightedRegressionImportanceSampling)) else None
        for collector, estimators in experiments:
            collector_name = collector.__class__.__name__
            print('\n', suffix, exp_name, training_episodes, collector_name)

            data_file = f'results/data/{policy_name}_{collector_name}_{collector_config_md5}_{seed}' if 'test' not in suffix else None
            trajectories, sa_counts, sta_counts, prh_sum = cache_run(lambda: collecting_data(env, max_time_step, collector, eval_steps, normalization=true_value, hasher=hasher, seed=seed), data_file)

            print('\n', len(trajectories), [len(t) for t in trajectories[:10]], '...')
            cross_entropy = compute_cross_entropy(op_data + trajectories, pi_e, normalized=True)

            if combined_trajectories == 0:
                # todo: adapt for combined off-policy data.
                results.setdefault("prh_sum", {}).setdefault(collector_name, []).append([prh_sum[step - 1] for step in eval_steps])
                results.setdefault("sa_counts", {}).setdefault(collector_name, []).append([sa_counts[step - 1] for step in eval_steps])
                results.setdefault("sta_counts", {}).setdefault(collector_name, []).append([sta_counts[step - 1] for step in eval_steps])
            results.setdefault("avg_grad", {}).setdefault(collector_name, []).append(compute_avg_grad(op_data + trajectories, pi_e, [s + op_data_steps for s in eval_steps]))
            results.setdefault("cross_entropy", {}).setdefault(collector_name, []).append([cross_entropy[step - 1 + op_data_steps] for step in eval_steps])
            [results.setdefault("estimations", {}).setdefault((collector_name, estimator.__class__.__name__,), []).append([]) for estimator in estimators]
            results.setdefault("kl_divergence", {}).setdefault(collector_name, []).append([]) if behavior_cloner is not None else None

            for step, new_data in yield_data(trajectories, eval_steps):
                start_time = time.time()
                data = op_data + new_data
                pi_D: REINFORCE = behavior_cloner.clone(data) if behavior_cloner is not None else None
                results.setdefault("kl_divergence", {}).setdefault(collector_name, [])[-1].append(compute_kl_divergence(data, pi_e, pi_D)) if pi_D is not None else None
                log = ''
                for estimator in estimators:
                    estimator_name = estimator.__class__.__name__
                    if combined_pre_estimate:
                        value = estimator.estimate(new_data)
                        new_n_trs = len(finished_trajectories(new_data))
                        value = (len(op_data) * op_value + new_n_trs * value) / (len(op_data) + new_n_trs)
                    else:
                        value = estimator.estimate(data, pi_D)

                    results.setdefault("estimations", {}).setdefault((collector_name, estimator_name), [])[-1].append(value)
                    log += f'{estimator_name}({step}): {value:.6f}   '
                log += f'time: {int(time.time() - start_time)}s'
                print(log)
            print()

        if os.path.exists('debug'):  # debug code.
            save(f'results/{exp_name}_{suffix}', results)

        # clear running trace
        for k in Global:
            if k.startswith('000'):
                Global[k].clear()

    save(result_file, results)  # merge needed.
    print('done.')
    time.sleep(1)  # wait save.


if __name__ == '__main__':
    main()
