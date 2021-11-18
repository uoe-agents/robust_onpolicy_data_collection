import os

ENV = 'qsub'
# ENV = 'sbatch'

#########################################################################################################
base_job_script = """#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N $job_name$
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l h_vmem=$h_vmem$
#$ -o log/$log.out$
#$ -e log/$log.err$

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
source activate diss

# Run the program
export PYTHONPATH=./
""" if ENV == 'qsub' else """#!/bin/bash
#SBATCH --job-name=$job_name$
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=$h_vmem$
#SBATCH --output=log/$log.out$
#SBATCH --error=log/$log.err$

# Run the program
"""
base_job_script += """
python -u experiments/evaluate.py --suffix "$suffix$" --policy_name $policy_name$ --collectors "$collectors$" --estimators "$estimators$" --eval_steps "$eval_steps$" --seeds "$seeds$" $more_args$
"""

from collections import defaultdict

from value_estimation import *
from data_collection_strategy import *
from experiments.policy_info import get_policy_info

os_estimators = (
    MonteCarlo,
)
bpg_estimators = (
    OrdinaryImportanceSampling,
)
ros_estimators = (
    MonteCarlo,
)

# os_estimators += (RegressionImportanceSampling,)
# bpg_estimators += (RegressionImportanceSampling,)
# ros_estimators += (RegressionImportanceSampling,)
#
# os_estimators += (WeightedRegressionImportanceSampling,)
# bpg_estimators += (WeightedRegressionImportanceSampling,)
# ros_estimators += (WeightedRegressionImportanceSampling,)
#
# os_estimators += (FitterQ,)
# bpg_estimators += (FitterQ,)
# ros_estimators += (FitterQ,)

submitted = defaultdict(int)
for policy_name, h_vmem, env_type in (
        ('MultiBandit_5000', '2G', 0),
        ('GridWorld_5000', '2G', 0),

        ('CartPole_1000', '4G', 1),

        ('CartPoleContinuous_3000', '4G', 2),

        ('MultiBanditM1S1_0', '1G', 3),
        ('MultiBanditM1S1_1', '1G', 3),
        ('MultiBanditM1S1_2', '1G', 3),
        ('MultiBanditM1S1_3', '1G', 3),
        ('MultiBanditM1S1_4', '1G', 3),
        ('MultiBanditM1S1_5', '1G', 3),
        ('MultiBanditM1S1_6', '1G', 3),
        ('MultiBanditM1S1_7', '1G', 3),
        ('MultiBanditM1S1_8', '1G', 3),
        ('MultiBanditM1S1_9', '1G', 3),
        ('MultiBanditM1S1_10', '1G', 3),
        ('MultiBanditM1S1_5000', '1G', 3),
        ('MultiBanditM1S2_5000', '1G', 3),
        ('MultiBanditM1S3_5000', '1G', 3),
        ('MultiBanditM1S4_5000', '1G', 3),
        ('MultiBanditM1S5_5000', '1G', 3),
        ('MultiBanditM1S6_5000', '1G', 3),
        ('MultiBanditM1S7_5000', '1G', 3),
        ('MultiBanditM1S8_5000', '1G', 3),
        ('MultiBanditM1S9_5000', '1G', 3),
        ('MultiBanditM1S10_5000', '1G', 3),
        ('MultiBanditM2S1_5000', '1G', 3),
        ('MultiBanditM3S1_5000', '1G', 3),
        ('MultiBanditM4S1_5000', '1G', 3),
        ('MultiBanditM5S1_5000', '1G', 3),
        ('MultiBanditM6S1_5000', '1G', 3),
        ('MultiBanditM7S1_5000', '1G', 3),
        ('MultiBanditM8S1_5000', '1G', 3),
        ('MultiBanditM9S1_5000', '1G', 3),
        ('MultiBanditM10S1_5000', '1G', 3),

        ('GridWorldM1S0_0', '1G', 3),
        ('GridWorldM1S0_1', '1G', 3),
        ('GridWorldM1S0_2', '1G', 3),
        ('GridWorldM1S0_3', '1G', 3),
        ('GridWorldM1S0_4', '1G', 3),
        ('GridWorldM1S0_5', '1G', 3),
        ('GridWorldM1S0_6', '1G', 3),
        ('GridWorldM1S0_7', '1G', 3),
        ('GridWorldM1S0_8', '1G', 3),
        ('GridWorldM1S0_9', '1G', 3),
        ('GridWorldM1S0_10', '1G', 3),
        ('GridWorldM1S1_5000', '1G', 3),
        ('GridWorldM1S2_5000', '1G', 3),
        ('GridWorldM1S3_5000', '1G', 3),
        ('GridWorldM1S4_5000', '1G', 3),
        ('GridWorldM1S5_5000', '1G', 3),
        ('GridWorldM1S6_5000', '1G', 3),
        ('GridWorldM1S7_5000', '1G', 3),
        ('GridWorldM1S8_5000', '1G', 3),
        ('GridWorldM1S9_5000', '1G', 3),
        ('GridWorldM1S10_5000', '1G', 3),
        ('GridWorldM2S1_5000', '1G', 3),
        ('GridWorldM3S1_5000', '1G', 3),
        ('GridWorldM4S1_5000', '1G', 3),
        ('GridWorldM5S1_5000', '1G', 3),
        ('GridWorldM6S1_5000', '1G', 3),
        ('GridWorldM7S1_5000', '1G', 3),
        ('GridWorldM8S1_5000', '1G', 3),
        ('GridWorldM9S1_5000', '1G', 3),
        ('GridWorldM10S1_5000', '1G', 3),
):
    exp_name, training_episodes, avg_steps = get_policy_info(policy_name)[:3]
    n_seeds = 200
    eval_steps = [int(avg_steps * 2 ** i) for i in range(14)]

    jobs = ((OnPolicySampler, os_estimators, '', '1',),)
    if env_type == 0:
        jobs += (
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-1', '1'),
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-2', '2'),  #
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-3', '3'),

            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+0', '0'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+1', '1'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+2', '2'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+3', '3'),  #
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+4', '4'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+5', '5'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+6', '6'),

            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 1.0', '1'),  #
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.8', '2'),  #
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.6', '3'),

            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10', 'Comb3'),
            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --combined_pre_estimate', f'PreComb3'),
            (BehaviorPolicyGradient, bpg_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --bpg_lr 1e-2', 'Comb3'),
            (RobustOnPolicySampler, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --ros_lr 1e+4', f'Comb3'),
            (RobustOnPolicyActing, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --ros_epsilon 1.0', 'Comb3'),
        )
    elif env_type == 1:
        jobs += (
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-4', '1'),
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 5e-5', '2'),  #
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-5', '3'),
            #
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+0', '1'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+1', '2'),  #
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+2', '3'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+3', '4'),
            #
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.50', '1'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.20', '2'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.10', '3'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.05', '4'),  #
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.02', '5'),

            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.05', 'Comb2'),
            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.05 --combined_pre_estimate', f'PreComb2'),
            (BehaviorPolicyGradient, bpg_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.05 --bpg_lr 5e-5', 'Comb2'),
            (RobustOnPolicySampler, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.05 --ros_lr 1e+1', 'Comb2'),
            (RobustOnPolicyActing, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.05 --ros_epsilon 0.05', 'Comb2'),

            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10', 'Comb3'),
            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --combined_pre_estimate', f'PreComb3'),
            (BehaviorPolicyGradient, bpg_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --bpg_lr 5e-5', 'Comb3'),
            (RobustOnPolicySampler, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --ros_lr 1e+1', f'Comb3'),
            (RobustOnPolicyActing, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.10 --ros_epsilon 0.1', 'Comb3'),
        )

    elif env_type == 2:
        jobs += (
            # (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-4', '1'), #nan
            # (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 5e-5', '2'), #nan
            # (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-5', '3'), #nan
            # (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 5e-6', '4'),  #nan
            (BehaviorPolicyGradient, bpg_estimators, '--bpg_lr 1e-6', '5'),  #

            # # (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e-0', '3'),  # nan.
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e-2', '1'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e-1', '2'),  #
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 2e-1', '4'),
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 5e-1', '5'),

            # # for eps
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.50 --ros_n_action 9', '1'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.20 --ros_n_action 9', '2'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.10 --ros_n_action 9', '3'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.05 --ros_n_action 9', '4'),  #
            # # for m
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.05 --ros_n_action 5', '5'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.05 --ros_n_action 15', '6'),
            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.05 --ros_n_action 19', '7'),

            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.1', 'Comb3'),
            (OnPolicySampler, os_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.1 --combined_pre_estimate', f'PreComb3'),
            (BehaviorPolicyGradient, bpg_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.1 --bpg_lr 1e-6', 'Comb3'),
            (RobustOnPolicySampler, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.1 --ros_lr 1e-1', 'Comb3'),
            (RobustOnPolicyActing, ros_estimators, '--combined_trajectories 100 --combined_ops_epsilon 0.1 --ros_epsilon 0.1 --ros_n_action 9', 'Comb3'),
        )

    elif env_type == 3:
        n_seeds = 500
        eval_steps = [int(avg_steps * 1000)]
        jobs += (
            (RobustOnPolicySampler, ros_estimators, '--ros_lr 1e+3', '1'),

            (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 1.0', '1'),
            # (RobustOnPolicyActing, ros_estimators, '--ros_epsilon 0.8', '2'),
        )

    for seed in range(n_seeds):
        for collector, estimators, more_args, c_suffix in jobs:
            for estimator in estimators:
                if 'combined_pre_estimate' in more_args and estimator is WeightedRegressionImportanceSampling:  # no need.
                    continue

                if estimator is FitterQ:
                    h_vmem_ = str(int(h_vmem[:-1]) * 2) + h_vmem[-1]
                else:
                    h_vmem_ = h_vmem

                seeds = [seed, seed + 1]
                suffix = f'{collector.__name__}{c_suffix}_{estimator.__name__}'

                file = f'results/seeds/{policy_name}_{suffix}_{seeds[0]}_{seeds[1]}'
                if os.path.exists(file):
                    print(f'file {file} already exists, skip running.')
                    continue

                job_script = base_job_script
                job_script = job_script.replace('$h_vmem$', h_vmem_)
                job_script = job_script.replace('$suffix$', suffix)
                job_script = job_script.replace('$policy_name$', policy_name)
                job_script = job_script.replace('$collectors$', str(collector.__name__))
                job_script = job_script.replace('$estimators$', str(estimator.__name__))
                job_script = job_script.replace('$eval_steps$', ','.join(str(i) for i in eval_steps))
                job_script = job_script.replace('$seeds$', f'{seeds[0]},{seeds[1]}')
                job_script = job_script.replace('$more_args$', more_args)

                job_name = f'{exp_name[:2]}_{collector.__name__[:2]}{c_suffix}_{estimator.__name__[:3]}'
                job_script = job_script.replace('$job_name$', job_name)
                job_script = job_script.replace('$log.out$', f'{policy_name}_{suffix}.out' if seed == 0 else 'log.out')
                job_script = job_script.replace('$log.err$', f'{policy_name}_{suffix}.out' if seed == 0 else 'log.err')
                # job_script += '\npython -u experiments/merge_results.py' if seed > 0 and seed % 40 == 0 else ''

                if os.path.exists('stop'):
                    os.remove('stop')
                    exit()

                print(job_script)
                with open('temp_job.sh', 'w+') as f:
                    f.write(job_script)

                if ENV == 'qsub':
                    os.system('qsub temp_job.sh')
                else:
                    os.system('sbatch temp_job.sh')

                submitted[f'{policy_name}_{suffix}'] += 1

[print(i) for i in submitted.items()]
