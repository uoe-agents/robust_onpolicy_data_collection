import os
import torch
import numpy as np
from policy import get_exp_config, REINFORCE

from commons import mc_estimation_true_value, save

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--policy_name', type=str)
args = parser.parse_args()

n = args.n
seed = args.seed
policy_name = args.policy_name
result_file = f'results/values/{policy_name}_{seed}'
if os.path.exists(result_file):
    print(f'file {result_file} exists, skip running.')
    exit()

if os.path.exists('debug'):
    n = 10
    seed = 999999
    policy_name = 'MultiBanditM1S1_5000'

exp_name = policy_name.split('_')[0]
env, max_time_step, gamma, _, _ = get_exp_config(exp_name)
pi_e = REINFORCE(env, gamma, file=f'policy_models/model_{policy_name}.pt', device='cpu')

env.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
true_value, true_variance, avg_steps, returns = mc_estimation_true_value(env, max_time_step, pi_e, gamma, n_trajectory=n)
print(f'\n{exp_name}: avg_step: {avg_steps}, true policy value and variance: {true_value}, {true_variance}')

if not os.path.exists('debug'):
    save(result_file, (returns, avg_steps))  # merge needed.
