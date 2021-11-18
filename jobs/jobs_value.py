import os

ENV = 'qsub'
# ENV = 'sbatch'

#########################################################################################################
base_job_script = """#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N $job_name$
#$ -cwd
#$ -l h_rt=01:00:00
#$ -l h_vmem=1G
#$ -o log/value.out
#$ -e log/value.err

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
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=500M
#SBATCH --output=log/value.out
#SBATCH --error=log/value.err

# Run the program
"""
base_job_script += """
python -u experiments/policy_value.py --policy_name $policy_name$ --seed $seed$ --n $n$
"""

total = 1e6
for policy_name in (
        'MultiBandit_5000',
        'GridWorld_5000',
        'CartPole_1000',
        'CartPoleContinuous_3000',

        'GridWorld_0',
        'GridWorld_1',
        'GridWorld_2',
        'GridWorld_3',
        'GridWorld_4',
        'GridWorld_5',
        'GridWorld_6',
        'GridWorld_7',
        'GridWorld_8',
        'GridWorld_9',
        'GridWorld_10',

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
    if 'MultiBanditM' in policy_name or 'GridWorldM' in policy_name:
        n = int(1e5)
    else:
        n = 2000

    for seed in range(int(total / n)):
        exp_name, training_episodes = policy_name.split("_")

        job_script = base_job_script
        job_script = job_script.replace('$job_name$', f'{exp_name[:2]}_{training_episodes}')
        job_script = job_script.replace('$policy_name$', policy_name)
        job_script = job_script.replace('$seed$', str(seed))
        job_script = job_script.replace('$n$', str(n))

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
