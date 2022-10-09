# Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning

Source code of [Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning](https://arxiv.org/pdf/2111.14552.pdf) (NeurIPS 2022).

The code is written in Python 3, using PyTorch for the implementation of the deep networks and OpenAI gym for the experiment domains.

## Requirements

To install the required codebase, it is recommended to create a conda or a virtual environment. Then, run the following command

```
pip install -r requirements.txt
```

## Preparation

To conduct policy evaluation, we need to prepare a set of pretrained policies. _You can skip this part if you already have the pretrained models in `policy_models/` and the corresponding policy values in `experiments/policy_info.py`_

### Pretrained Policy

Train the policy models using REINFORCE in different domains by running:

```shell
python policy/reinfoce.py --exp_name {exp_name}
```

where `{exp_name}` can be MultiBandit, GridWorld, CartPole or CartPoleContinuous. The parameterized epsilon-greedy policies for MultiBandit and GridWorld can be obtained by running:

```shell
python policy/handmade_policy.py
```

### Policy Value

#### Option 1: Run in sequence

For each policy model, the true policy value is estimated with $10^6$ Monte Carlo roll-outs by running:

```shell
python experiments/policy_value.py --policy_name {policy_name} --seed {seed} --n 10e6
```

This will print the average steps, true policy value and variance of returns. Make sure you copy these results into the file ``experiment/policy_info.py``.

#### Option 2: Run in parallel

If you can use `qsub` or `sbatch`, you can also run `jobs/jobs_value.py` with different seeds in parallel and merge them by running `experiments/merge_values.py` to get $10^6$ Monte Carlo roll-outs. The policy values reported in this paper were obtained in this way.

## Evaluation

#### Option 1: Run in sequence

The main running script for policy evaluation is `experiments/evaluate.py`. The following running command is an example of Monte Carlo estimation for Robust On-policy Acting with $\rho=1.0$ for the policy `model_GridWorld_5000.pt` with seeds from 0 to 199.

```shell
python experiments/evaluate.py --policy_name GridWorld_5000 --ros_epsilon 1.0 --collectors RobustOnPolicyActing --estimators MonteCarlo --eval_steps "7,14,29,59,118,237,475,951,1902,3805,7610,15221,30443,60886" --seeds "0,199"
```

To conduct policy evaluation with off-policy data, you need to add the following arguments to the above running command:

```
--combined_trajectories 100 --combined_ops_epsilon 0.10 
```

#### Option 2: Run in parallel

If you can use `qsub` or `sbatch`, you may only need to run the script `jobs/jobs.py` where all experiments in the paper are arranged. The log will be saved in ``log/`` and the seed results will be saved in ``results/seeds``. Note that we save the data collection cache in ``results/data`` and re-use it for different value estimations. To merge results of different seeds, run `experiments/merge_results.py`, and the merged results will be saved in ``results/``.

## Data

All experimental data used to generate plots of the paper can be found in ``data/`` with the following structure:

1. Subdirectories:

     ``data/`` contains six subdirectories, out of which four (``bandit``, ``Gridworld``, ``Cartpole`` and ``con_cartpole``) contain results for each of the four domains. The other two subfolders (``Gridworld_ms`` and ``bandit_ms``) contain results for further experiments used to generate Figures like Figure 4c, d in the paper.

2. File names:

   The name of a file consists of four parts: domain, number of pre-trainings to get the     evaluation policy, sampling method and estimation method. For example, in the ``data/bandit`` folder, there is a file named ``MultiBandit_5000_BehaviorPolicyGradient1_OrdinaryImportanceSampling``. As for ``data/Gridworld_ms`` and ``data/bandit_ms``, file names contain ``ms`` referring to means and scale.


## Plotting

Code is provided to reproduce all the figures included in the paper. See the jupyter notebooks found in ``plotting/``.


## Citing

If you use this repository in your work, please consider citing the paper

```bibtex
@inproceedings{zhong2022robust,
    title = {Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning},
    author = {Rujie Zhong, Duohan Zhang, Lukas Sch\”afer, Stefano V. Albrecht, Josiah P. Hanna},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2022}
}
```
