import numpy as np
import os
from commons import save, load

results = ''
root = 'results/values'
for policy_name in (
        'MultiBandit_5000',
        'GridWorld_5000',
        'CartPole_1000',
        'CartPoleContinuous_3000',

        'MultiBanditM1S1_0',
        'MultiBanditM1S1_1',
        'MultiBanditM1S1_2',
        'MultiBanditM1S1_3',
        'MultiBanditM1S1_4',
        'MultiBanditM1S1_5',
        'MultiBanditM1S1_6',
        'MultiBanditM1S1_7',
        'MultiBanditM1S1_8',
        'MultiBanditM1S1_9',
        'MultiBanditM1S1_10',
        'MultiBanditM1S1_5000',
        'MultiBanditM1S2_5000',
        'MultiBanditM1S3_5000',
        'MultiBanditM1S4_5000',
        'MultiBanditM1S5_5000',
        'MultiBanditM1S6_5000',
        'MultiBanditM1S7_5000',
        'MultiBanditM1S8_5000',
        'MultiBanditM1S9_5000',
        'MultiBanditM1S10_5000',
        'MultiBanditM2S1_5000',
        'MultiBanditM3S1_5000',
        'MultiBanditM4S1_5000',
        'MultiBanditM5S1_5000',
        'MultiBanditM6S1_5000',
        'MultiBanditM7S1_5000',
        'MultiBanditM8S1_5000',
        'MultiBanditM9S1_5000',
        'MultiBanditM10S1_5000',

        'GridWorldM1S0_0',
        'GridWorldM1S0_1',
        'GridWorldM1S0_2',
        'GridWorldM1S0_3',
        'GridWorldM1S0_4',
        'GridWorldM1S0_5',
        'GridWorldM1S0_6',
        'GridWorldM1S0_7',
        'GridWorldM1S0_8',
        'GridWorldM1S0_9',
        'GridWorldM1S0_10',
        'GridWorldM1S1_5000',
        'GridWorldM1S2_5000',
        'GridWorldM1S3_5000',
        'GridWorldM1S4_5000',
        'GridWorldM1S5_5000',
        'GridWorldM1S6_5000',
        'GridWorldM1S7_5000',
        'GridWorldM1S8_5000',
        'GridWorldM1S9_5000',
        'GridWorldM1S10_5000',
        'GridWorldM2S1_5000',
        'GridWorldM3S1_5000',
        'GridWorldM4S1_5000',
        'GridWorldM5S1_5000',
        'GridWorldM6S1_5000',
        'GridWorldM7S1_5000',
        'GridWorldM8S1_5000',
        'GridWorldM9S1_5000',
        'GridWorldM10S1_5000',
):
    returns = []
    avg_steps = []
    n = None
    for seed in range(1000):
        file = f'{root}/{policy_name}_{seed}'
        if not os.path.exists(file):
            continue
        r, s = load(file)
        returns.extend(r)
        avg_steps.append(s)
        n = len(r) if n is None else n
        assert n == len(r), f'{len(r)} != {n}: {file}'

    results += f'elif policy_name == "{policy_name}": # {len(returns)} trs\n'
    results += f'\tavg_step, true_value, true_variance ={np.mean(avg_steps)}, {np.mean(returns)}, {np.var(returns)}\n'

print()
print(results)

# python experiments/merge_values.py
