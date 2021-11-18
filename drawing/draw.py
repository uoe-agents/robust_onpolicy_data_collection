import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from experiments.policy_info import get_policy_info
from experiments.plot_local import switch_name, label_style, linestyles, colors
from commons import load

rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 14,
    'legend.fontsize': 12,
    "mathtext.fontset": 'cm',
})
# rcParams['text.usetex'] = True

cache = {}


def gaussian(x, m, v):
    return 1 / np.sqrt(2 * np.pi * v) * np.exp(- (x - m) ** 2 / 2 / v)


def dist_pieDb():
    x = np.linspace(-1, 1, num=1000)
    pi_e = gaussian(x, 0, 0.1)
    pi_D = 0.8 * pi_e + 0.1 * gaussian(x, -0.1, 0.01) + 0.1 * gaussian(x, 0.3, 0.01)

    alpha = 1
    pi_b_ideal = pi_e + (pi_e - pi_D) * alpha

    pi_D_mean = (x * pi_D).sum() / pi_D.sum()
    pi_b_continuous = gaussian(x, 0.1 - pi_D_mean, 0.08)

    plt.figure(figsize=(8, 4))
    plt.plot(x, pi_e, label='$\\pi_e$', alpha=0.5, linestyle=':')
    plt.plot(x, pi_D, label='$\\pi_D$', alpha=0.5)
    # plt.plot(x, pi_b_ideal, label='$\\pi_b\'$', alpha=0.5)
    plt.xlabel('$a$')
    plt.ylabel('$\\pi(a|s)$', fontsize=14)
    plt.yticks([])
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('writing_aaai/figs/dist_pieDb.pdf', format='pdf')
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(x, pi_e, label='$\\pi_e$', alpha=0.5, linestyle=':')
    plt.plot(x, pi_D, label='$\\pi_D$', alpha=0.5)
    # plt.plot(x, pi_b_ideal, label='$\\pi_b^*$', alpha=0.5)
    plt.plot(x, pi_b_continuous, label='$\\pi_{\\mathbf{\\theta}_b}$', alpha=0.5)  # todo mathbf.
    plt.xlabel('$a$')
    plt.ylabel('$\\pi(a|s)$')
    plt.yticks([])
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('writing_aaai/figs/dist_pieDb_continuous.pdf', format='pdf')
    plt.show()


def mc_confidence():
    for policy_name in (
            'MultiBandit_5000',
            'GridWorld_5000',
            'CartPole_1000',
            'CartPoleContinuous_3000',
    ):
        exp_name, training_episodes, avg_step, true_value, true_variance, reward_range = get_policy_info(policy_name)
        from policy import get_exp_config
        _, max_time_step, gamma, _, _ = get_exp_config(exp_name)

        true_sigma = (true_variance / 1000000) ** 0.5
        print(f'{exp_name} & {avg_step:.2f} & {true_value:.2f} $\\pm$ {true_sigma :.2e} & 1 $\\pm$ {true_sigma / abs(true_value) :.2e} \\\\')


def draw_mab():
    from experiments.multibandit import MultiBandit

    # env = MultiBandit()
    # data = np.array([[np.random.normal(env.means[a], env.scales[a]) for a in range(env.action_space.n)] for _ in range(100)])
    # plt.boxplot(data, showfliers=False)
    # plt.xticks(fontsize=8)
    # plt.xlabel('Action')
    # plt.ylabel('Reward')
    # plt.tight_layout()
    # plt.savefig('writing_aaai/figs/env_multibandit.pdf', format='pdf')
    # plt.show()

    import torch

    env = MultiBandit()
    means = env.means
    scales = env.scales
    for i in range(len(means)):
        dist = torch.distributions.Normal(means[i], scales[i])

        y = torch.arange(101 * 4).float() / 200
        y -= y.mean()
        y += means[i]
        x_range = dist.log_prob(y).exp()
        x_range /= 10
        x_min = i - x_range
        x_max = i + x_range

        plt.fill_betweenx(y, x_min, x_max)
        plt.scatter(i, means[i], color='black', s=1, marker='o', alpha=0.2)

    plt.xticks(fontsize=8)
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    plt.tight_layout()
    plt.savefig('writing_aaai/figs/env_multibandit.pdf', format='pdf')
    plt.show()


groups = (
    (
        ('results/MultiBandit_5000_OnPolicySampler1_MonteCarlo', (), ''),
        ('results/MultiBandit_5000_BehaviorPolicyGradient2_OrdinaryImportanceSampling', (), ''),
        ('results/MultiBandit_5000_RobustOnPolicySampler3_MonteCarlo', (), ''),
        ('results/MultiBandit_5000_RobustOnPolicyActing1_MonteCarlo', (), ''),
    ),
    (
        ('results/GridWorld_5000_OnPolicySampler1_MonteCarlo', (), ''),
        ('results/GridWorld_5000_BehaviorPolicyGradient2_OrdinaryImportanceSampling', (), ''),
        ('results/GridWorld_5000_RobustOnPolicySampler3_MonteCarlo', (), ''),
        ('results/GridWorld_5000_RobustOnPolicyActing2_MonteCarlo', (), ''),  # 0.8
    ),
    (
        ('results/CartPole_1000_OnPolicySampler1_MonteCarlo', (), ''),
        ('results/CartPole_1000_BehaviorPolicyGradient2_OrdinaryImportanceSampling', (), ''),
        ('results/CartPole_1000_RobustOnPolicySampler2_MonteCarlo', (), ''),
        ('results/CartPole_1000_RobustOnPolicyActing4_MonteCarlo', (), ''),
    ),
    (
        ('results/CartPoleContinuous_3000_OnPolicySampler1_MonteCarlo', (), ''),
        ('results/CartPoleContinuous_3000_BehaviorPolicyGradient5_OrdinaryImportanceSampling', (), ''),
        ('results/CartPoleContinuous_3000_RobustOnPolicySampler2_MonteCarlo', (), ''),
        ('results/CartPoleContinuous_3000_RobustOnPolicyActing4_MonteCarlo', (), ''),
    ),
)


def draw_basic():
    final_mse = {}
    for group_i, jobs in enumerate(groups):

        for draw, draw_name in (
                (draw_mean_sigma, 'mse'),
                (draw_median_quantile, 'mq'),
        ):
            for file, _, suffix in jobs:
                results = load(file)
                if cache.setdefault(file, True):
                    cache[file] = False
                    print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
                exp_name = results['exp_name']
                estimations = results['estimations']
                true_value = results['true_value']
                eval_steps = results['eval_steps']

                for (collector_name, estimator_name), estimation in estimations.items():
                    collector_name, estimator_name = switch_name(collector_name, estimator_name)
                    collector_name += suffix
                    label, color, linestyle = label_style(collector_name, estimator_name)
                    estimation = np.array(estimation)
                    mse = (estimation - true_value) ** 2
                    # mse = np.abs(estimation - true_value)
                    draw(eval_steps, mse, label, color, linestyle=linestyle, sigma_range=0.2)

                    final_mse[(exp_name, label)] = mse[..., -1].mean(), mse[..., -1].std() / (len(mse) ** 0.5)

            plt.legend()
            plt.xlabel('Steps')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.xscale('log')
            # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'writing_aaai/figs/basic_{draw_name}_{exp_name}.pdf', format='pdf')
            # plt.show()
            plt.clf()

        drawn = []
        for file, _, suffix in jobs:
            results = load(file)
            exp_name = results['exp_name']
            sa_counts = results['sa_counts']
            eval_steps = results['eval_steps']

            for collector_name, counts in sa_counts.items():
                collector_name, _ = switch_name(collector_name, '')
                collector_name += suffix
                if collector_name not in drawn:
                    drawn.append(collector_name)
                    label, color, linestyle = label_style(collector_name, "MC")
                    draw_mean_sigma(eval_steps, counts, label, color, linestyle=linestyle, sigma_range=1.)

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Unique state-action pairs')
        plt.xscale('log')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/basic_count_{exp_name}.pdf', format='pdf')
        plt.clf()

        drawn = []
        for file, _, suffix in jobs:
            results = load(file)
            exp_name = results['exp_name']
            eval_steps = results['eval_steps']
            cross_entropy = results['cross_entropy']
            for collector_name, loss in cross_entropy.items():
                collector_name, _ = switch_name(collector_name, '')
                collector_name += suffix
                if collector_name not in drawn:
                    drawn.append(collector_name)
                    label, color, linestyle = label_style(collector_name, "MC")

                    loss = np.array(loss)
                    mse = (loss - 1) ** 2
                    draw_mean_sigma(eval_steps, mse, collector_name, color, )

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Steps')
        plt.ylabel('Sampling error')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/basic_ce_{exp_name}.pdf', format='pdf')
        plt.clf()

        drawn = []
        for file, _, suffix in jobs:
            file = file.replace('_MonteCarlo', '_WeightedRegressionImportanceSampling').replace('_OrdinaryImportanceSampling', '_WeightedRegressionImportanceSampling')
            results = load(file)
            print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
            exp_name = results['exp_name']
            eval_steps = results['eval_steps']
            kl_divergence = results.get('kl_divergence', None)
            if kl_divergence is None:
                continue
            for collector_name, values in kl_divergence.items():
                collector_name, _ = switch_name(collector_name, '')
                collector_name += suffix
                if collector_name not in drawn:
                    drawn.append(collector_name)
                    label, color, linestyle = label_style(collector_name, "MC")
                    draw_mean_sigma(eval_steps, values, collector_name, color, sigma_range=1.)

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Steps')
        plt.ylabel('Sampling error')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/basic_kl_{exp_name}.pdf', format='pdf')
        plt.clf()

        drawn = []
        for file, _, suffix in jobs:
            results = load(file)
            exp_name = results['exp_name']
            if 'avg_grad' not in results:
                continue
            avg_grad = results['avg_grad']
            eval_steps = results['eval_steps']
            for collector_name, values in avg_grad.items():
                collector_name, _ = switch_name(collector_name, '')
                collector_name += suffix
                if collector_name not in drawn:
                    drawn.append(collector_name)
                    label, color, linestyle = label_style(collector_name, "MC")
                    draw_mean_sigma(eval_steps, values, collector_name, color, sigma_range=1.)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('$|| \\nabla ||$', rotation=0, labelpad=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/basic_grad_{exp_name}.pdf', format='pdf')
        plt.clf()

    draw_hyper_parameters(groups)
    print()
    build_mse_table(final_mse, estimators=('MC', 'OIS'))


def draw_grad():
    drawn = []
    jobs = []
    exp_names = []
    [jobs.extend(j) for j in groups]
    for file, _, suffix in jobs:
        results = load(file)
        exp_name = results['exp_name']
        exp_names.append(exp_name) if exp_name not in exp_names else None
        if 'avg_grad' not in results:
            continue
        avg_grad = results['avg_grad']
        eval_steps = results['eval_steps']
        for collector_name, values in avg_grad.items():
            collector_name, _ = switch_name(collector_name, '')
            if collector_name != 'ROS':
                continue
            collector_name += suffix
            key = (exp_name, collector_name)
            if key not in drawn:
                drawn.append(key)
                # linestyle = linestyles[exp_names.index(exp_name)]
                # _, color, _ = label_style(collector_name, "MC")
                color = colors[drawn.index(key)]
                draw_mean_sigma([2 ** i for i in range(len(eval_steps))], values, exp_name, color, linestyle=None)
    plt.legend()
    plt.xlabel('Trajectories')
    plt.ylabel('$| \\nabla |$', rotation=0, labelpad=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'writing_aaai/figs/grad.pdf', format='pdf')
    plt.clf()


def draw_alpha():
    alpha_groups = (
        (
            ('results/MultiBandit_5000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/MultiBandit_5000_RobustOnPolicySampler0_MonteCarlo', (), '', 0.3, ':'),
            ('results/MultiBandit_5000_RobustOnPolicySampler1_MonteCarlo', (), '', 0.3, '-.'),
            ('results/MultiBandit_5000_RobustOnPolicySampler2_MonteCarlo', (), '', 0.3, '--'),
            ('results/MultiBandit_5000_RobustOnPolicySampler3_MonteCarlo', (), '', 0.6, '-'),
        ),
        (
            ('results/GridWorld_5000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/GridWorld_5000_RobustOnPolicySampler0_MonteCarlo', (), '', 0.3, ':'),
            ('results/GridWorld_5000_RobustOnPolicySampler1_MonteCarlo', (), '', 0.3, '-.'),
            ('results/GridWorld_5000_RobustOnPolicySampler2_MonteCarlo', (), '', 0.6, '--'),
            ('results/GridWorld_5000_RobustOnPolicySampler3_MonteCarlo', (), '', 0.9, '-'),
        ),
        (
            ('results/CartPole_1000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/CartPole_1000_RobustOnPolicySampler1_MonteCarlo', (), '', 0.3, ':'),
            ('results/CartPole_1000_RobustOnPolicySampler2_MonteCarlo', (), '', 0.6, '-.'),
            ('results/CartPole_1000_RobustOnPolicySampler3_MonteCarlo', (), '', 0.9, '--'),
            ('results/CartPole_1000_RobustOnPolicySampler4_MonteCarlo', (), '', 0.9, '-'),
        ),
        (
            ('results/CartPoleContinuous_3000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/CartPoleContinuous_3000_RobustOnPolicySampler1_MonteCarlo', (), '', 0.3, ':'),
            ('results/CartPoleContinuous_3000_RobustOnPolicySampler2_MonteCarlo', (), '', 0.6, '--'),
            # ('results/CartPoleContinuous_3000_RobustOnPolicySampler4_MonteCarlo', (), '', 0.9, '-'),  # todo. nan.
        ),
    )
    for group_i, jobs in enumerate(alpha_groups):
        for file, _, suffix, alpha, linestyle in jobs:
            results = load(file)
            print(f"{file}: {len(list(results['sa_counts'].values())[0])} seeds")
            exp_name = results['exp_name']
            estimations = results['estimations']
            true_value = results['true_value']
            eval_steps = results['eval_steps']
            suffix = f'($\\alpha={0 if "_OnPolicySampler1_" in file else results["config"]["ros_lr"]}$)'

            for (collector_name, estimator_name), estimation in estimations.items():
                collector_name, estimator_name = switch_name(collector_name, estimator_name)
                _, color, _ = label_style(collector_name, estimator_name)
                collector_name += suffix
                label, _, _ = label_style(collector_name, estimator_name)
                estimation = np.array(estimation)
                mse = (estimation - true_value) ** 2
                draw_mean_sigma(eval_steps, mse, label=label, color=color, linestyle=linestyle)

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.xscale('log')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/alpha_{exp_name}.pdf', format='pdf')
        plt.clf()


def draw_rho():
    eps_groups = (
        (
            ('results/MultiBandit_5000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/MultiBandit_5000_RobustOnPolicyActing3_MonteCarlo', (), '', 0.3, ':'),
            ('results/MultiBandit_5000_RobustOnPolicyActing2_MonteCarlo', (), '', 0.6, '--'),
            ('results/MultiBandit_5000_RobustOnPolicyActing1_MonteCarlo', (), '', 0.9, '-'),
        ),
        (
            ('results/GridWorld_5000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/GridWorld_5000_RobustOnPolicyActing3_MonteCarlo', (), '', 0.3, ':'),
            ('results/GridWorld_5000_RobustOnPolicyActing2_MonteCarlo', (), '', 0.6, '--'),
            ('results/GridWorld_5000_RobustOnPolicyActing1_MonteCarlo', (), '', 0.9, '-'),
        ),
        (
            ('results/CartPole_1000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/CartPole_1000_RobustOnPolicyActing4_MonteCarlo', (), '', 0.3, ':'),
            ('results/CartPole_1000_RobustOnPolicyActing3_MonteCarlo', (), '', 0.6, '--'),
            ('results/CartPole_1000_RobustOnPolicyActing2_MonteCarlo', (), '', 0.9, '-'),
        ),
        (
            ('results/CartPoleContinuous_3000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing4_MonteCarlo', (), '', 0.3, ':'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing3_MonteCarlo', (), '', 0.6, '--'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing2_MonteCarlo', (), '', 0.9, '-'),
        ),
    )
    for group_i, jobs in enumerate(eps_groups):
        for file, _, suffix, alpha, linestyle in jobs:
            results = load(file)
            print(f"{file}: {len(list(results['sa_counts'].values())[0])} seeds")
            exp_name = results['exp_name']
            estimations = results['estimations']
            true_value = results['true_value']
            eval_steps = results['eval_steps']
            if "_OnPolicySampler1_" in file:
                suffix = f'($\\rho=0$)'
            else:
                suffix = f'($\\rho={results["config"]["ros_epsilon"]}'
                if 'Continuous' in exp_name:
                    suffix += f', m={results["config"]["ros_n_action"]}'
                suffix += '$)'

            for (collector_name, estimator_name), estimation in estimations.items():
                collector_name, estimator_name = switch_name(collector_name, estimator_name)
                _, color, _ = label_style(collector_name, estimator_name)
                collector_name += suffix
                label, _, _ = label_style(collector_name, estimator_name)
                estimation = np.array(estimation)
                mse = (estimation - true_value) ** 2
                draw_mean_sigma(eval_steps, mse, label=label, color=color, linestyle=linestyle)

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.xscale('log')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/eps_{exp_name}.pdf', format='pdf')
        plt.clf()


def draw_m():
    m_groups = (
        (
            ('results/CartPoleContinuous_3000_OnPolicySampler1_MonteCarlo', (), '', 0.1, '-'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing5_MonteCarlo', (), '', 0.3, ':'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing4_MonteCarlo', (), '', 0.6, '-.'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing6_MonteCarlo', (), '', 0.9, '--'),
            ('results/CartPoleContinuous_3000_RobustOnPolicyActing7_MonteCarlo', (), '', 0.9, '-'),
        ),
    )
    for group_i, jobs in enumerate(m_groups):
        for file, _, suffix, alpha, linestyle in jobs:
            results = load(file)
            print(f"{file}: {len(list(results['sa_counts'].values())[0])} seeds")
            exp_name = results['exp_name']
            estimations = results['estimations']
            true_value = results['true_value']
            eval_steps = results['eval_steps']

            if "_OnPolicySampler1_" in file:
                suffix = f''
            else:
                suffix = f'($\\rho={results["config"]["ros_epsilon"]}, m={results["config"]["ros_n_action"]}$)'

            for (collector_name, estimator_name), estimation in estimations.items():
                collector_name, estimator_name = switch_name(collector_name, estimator_name)
                _, color, _ = label_style(collector_name, estimator_name)
                collector_name += suffix
                label, _, _ = label_style(collector_name, estimator_name)
                estimation = np.array(estimation)
                mse = (estimation - true_value) ** 2
                draw_mean_sigma(eval_steps, mse, label=label, color=color, linestyle=linestyle)

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.xscale('log')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/m_{exp_name}.pdf', format='pdf')
        plt.clf()


comb_groups = (
    (
        (f'results/MultiBandit_5000_OnPolicySampler1_MonteCarlo', (), 'OS - MC', 1.0, '-'),
        (f'results/MultiBandit_5000_OnPolicySamplerComb3_MonteCarlo', (), '(OPD + OS) - MC', 1.0, '--'),
        (f'results/MultiBandit_5000_OnPolicySamplerPreComb3_MonteCarlo', (), '(OPD + OS) - (WIS + MC)', 1.0, ':'),
        (f'results/MultiBandit_5000_BehaviorPolicyGradientComb3_OrdinaryImportanceSampling', (), '(OPD + BPG) - OIS', 1.0, '--'),
        (f'results/MultiBandit_5000_RobustOnPolicySamplerComb3_MonteCarlo', (), '(OPD + ROS) - MC', 1.0, '--'),
        (f'results/MultiBandit_5000_RobustOnPolicyActingComb3_MonteCarlo', (), '(OPD + ROA) - MC', 1.0, '--'),
    ),
    (
        (f'results/GridWorld_5000_OnPolicySampler1_MonteCarlo', (), 'OS - MC', 1.0, '-'),
        (f'results/GridWorld_5000_OnPolicySamplerComb3_MonteCarlo', (), '(OPD + OS) - MC', 1.0, '--'),
        (f'results/GridWorld_5000_OnPolicySamplerPreComb3_MonteCarlo', (), '(OPD + OS) - (WIS + MC)', 1.0, ':'),
        (f'results/GridWorld_5000_BehaviorPolicyGradientComb3_OrdinaryImportanceSampling', (), '(OPD + BPG) - OIS', 1.0, '--'),
        (f'results/GridWorld_5000_RobustOnPolicySamplerComb3_MonteCarlo', (), '(OPD + ROS) - MC', 1.0, '--'),
        (f'results/GridWorld_5000_RobustOnPolicyActingComb3_MonteCarlo', (), '(OPD + ROA) - MC', 1.0, '--'),
    ),
    (
        (f'results/CartPole_1000_OnPolicySampler1_MonteCarlo', (), 'OS - MC', 1.0, '-'),
        (f'results/CartPole_1000_OnPolicySamplerComb3_MonteCarlo', (), '(OPD + OS) - MC', 1.0, '--'),
        (f'results/CartPole_1000_OnPolicySamplerPreComb3_MonteCarlo', (), '(OPD + OS) - (WIS + MC)', 1.0, ':'),
        (f'results/CartPole_1000_BehaviorPolicyGradientComb3_OrdinaryImportanceSampling', (), '(OPD + BPG) - OIS', 1.0, '--'),
        (f'results/CartPole_1000_RobustOnPolicySamplerComb3_MonteCarlo', (), '(OPD + ROS) - MC', 1.0, '--'),
        (f'results/CartPole_1000_RobustOnPolicyActingComb3_MonteCarlo', (), '(OPD + ROA) - MC', 1.0, '--'),
    ),
    (
        (f'results/CartPoleContinuous_3000_OnPolicySampler1_MonteCarlo', (), 'OS - MC', 1.0, '-'),
        (f'results/CartPoleContinuous_3000_OnPolicySamplerComb3_MonteCarlo', (), '(OPD + OS) - MC', 1.0, '--'),
        (f'results/CartPoleContinuous_3000_OnPolicySamplerPreComb3_MonteCarlo', (), '(OPD + OS) - (WIS + MC)', 1.0, ':'),
        (f'results/CartPoleContinuous_3000_BehaviorPolicyGradientComb3_OrdinaryImportanceSampling', (), '(OPD + BPG) - OIS', 1.0, '--'),
        (f'results/CartPoleContinuous_3000_RobustOnPolicySamplerComb3_MonteCarlo', (), '(OPD + ROS) - MC', 1.0, '--'),
        (f'results/CartPoleContinuous_3000_RobustOnPolicyActingComb3_MonteCarlo', (), '(OPD + ROA) - MC', 1.0, '--'),
        # (f'results/CartPoleContinuous_3000_RobustOnPolicyActing4_MonteCarlo', (), '(OPD + ROA) - MC', 1.0, ':'),
    ),
)


def draw_comb():
    final_mse = {}
    for group_i, jobs in enumerate(comb_groups):
        for draw, draw_name in (
                (draw_mean_sigma, 'mse'),
                (draw_median_quantile, 'mq'),
        ):
            vline_y_max = 0
            vline_x = None
            for file, _, suffix, alpha, linestyle in jobs:
                results = load(file)
                exp_name = results['exp_name']
                estimations = results['estimations']
                true_value = results['true_value']
                eval_steps = results['eval_steps']

                for (collector_name, estimator_name), estimation in estimations.items():
                    collector_name, estimator_name = switch_name(collector_name, estimator_name)
                    print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")

                    _, color, _ = label_style(collector_name, estimator_name)
                    collector_name += suffix
                    label, _, _ = label_style(collector_name, estimator_name)
                    estimation = np.array(estimation)
                    mse = (estimation - true_value) ** 2

                    vline_y_max = max(mse.mean(0).max(), vline_y_max)
                    if eval_steps[0] == 0:
                        vline_x = eval_steps[1]

                    draw(eval_steps, mse, suffix, color, linestyle=linestyle, sigma_range=0.2, alpha_alpha=alpha)
                    final_mse[(exp_name, suffix)] = mse[..., -1].mean(), mse[..., -1].std() / (len(mse) ** 0.5)

            # plt.vlines(vline_x, 0, vline_y_max * 2, linestyles='--', alpha=0.2)
            plt.legend()
            plt.xlabel('Steps')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.xscale('log')
            # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'writing_aaai/figs/comb_{draw_name}_{exp_name}.pdf', format='pdf')
            plt.clf()

        drawn = []
        vline_y_max = 0
        vline_x = None
        for file, _, suffix, alpha, linestyle in jobs:
            file = file.replace('MonteCarlo', 'WeightedRegressionImportanceSampling')
            file = file.replace('OrdinaryImportanceSampling', 'WeightedRegressionImportanceSampling')
            try:
                results = load(file)
            except:
                continue
            exp_name = results['exp_name']
            eval_steps = results['eval_steps']
            kl_divergence = results.get('kl_divergence', None)
            if kl_divergence is None:
                continue

            for collector_name, values in kl_divergence.items():
                values = np.array(values)
                collector_name, _ = switch_name(collector_name, '')
                label, color, _ = label_style(collector_name, 'MC')
                collector_name += suffix
                if collector_name not in drawn:
                    drawn.append(collector_name)

                    vline_y_max = max(values.mean(0).max(), vline_y_max)
                    if eval_steps[0] == 0:
                        vline_x = eval_steps[1]

                    # values = np.clip(values, 1e-10, np.inf)
                    draw_mean_sigma(eval_steps, values, suffix.split(' - ')[0], color, linestyle=linestyle, sigma_range=1., alpha_alpha=alpha)

        # plt.vlines(vline_x, 0, vline_y_max * 2, linestyles='--', alpha=0.2)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Steps')
        plt.ylabel('Sampling error')
        # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/comb_kl_{exp_name}.pdf', format='pdf')
        plt.clf()

        drawn = []
        vline_y_max = 0
        vline_x = None
        for file, _, suffix, alpha, linestyle in jobs:
            results = load(file)
            exp_name = results['exp_name']
            if 'avg_grad' not in results:
                continue
            avg_grad = results['avg_grad']
            eval_steps = results['eval_steps']
            for collector_name, values in avg_grad.items():
                values = np.array(values)
                collector_name, _ = switch_name(collector_name, '')
                label, color, _ = label_style(collector_name, 'MC')
                collector_name = suffix.split(' - ')[0]
                if collector_name not in drawn:
                    drawn.append(collector_name)

                    vline_y_max = max(values.mean(0).max(), vline_y_max)
                    if eval_steps[0] == 0:
                        vline_x = eval_steps[1]

                    draw_mean_sigma(eval_steps, values, collector_name, color, linestyle=linestyle, sigma_range=1.)

        # plt.vlines(vline_x, 0, vline_y_max * 2, linestyles='--', alpha=0.2)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('$|| \\nabla ||$', rotation=0, labelpad=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/comb_grad_{exp_name}.pdf', format='pdf')
        plt.clf()

    draw_hyper_parameters(comb_groups)
    print()
    build_mse_table(final_mse, estimators=('MC', 'OIS'))


def build_mse_table(final_mse, estimators: tuple = None):
    labels = []
    exp_names = []
    mathbf = '\\textbf{value}'
    normal = 'value'

    for exp_name, label in final_mse:
        exp_names.append(exp_name) if exp_name not in exp_names else None
        if label not in labels:
            if estimators and any(e in label for e in estimators):
                labels.append(label)

    print('\\hline')
    print('Policy Evaluation & ' + ' & '.join(
        [exp_name for exp_name in exp_names]
    ) + '\\\\')
    print('\\hline')
    for label in labels:
        print(
            f'{label} & ' +
            ' & '.join([
                (mathbf if 'ROA' in label else normal).replace('value', f'{final_mse[(exp_name, label)][0]:.2e} $\\pm$ {final_mse[(exp_name, label)][1]:.2e}')
                for exp_name in exp_names]) +
            '\\\\'
        )
    print('\\hline')

    # print('\\hline')
    # print('Strategy & Estimation & ' + ' & '.join(
    #     [exp_name for exp_name in exp_names]
    # ) + '\\\\')
    # print('\\hline')
    # for label in labels:
    #     s, e = label.split(' - ')
    #     print(
    #         f'{s} & {e} & ' +
    #         ' & '.join([f'{final_mse[(exp_name, label)]:.2e}' for exp_name in exp_names]) +
    #         '\\\\'
    #     )
    # print('\\hline')


def draw_est():
    est_groups = (
        (
            ('results/CartPole_OnPolicySampler1_MonteCarlo', (), ''),
            ('results/CartPole_RobustOnPolicySampler2_MonteCarlo', (), '(update $\\theta$)'),
        ),
    )
    fig_i = 0
    for group_i, jobs in enumerate(est_groups):
        for file, _, suffix in jobs:
            results = load(file)
            print(f"{file}: {len(list(results['sa_counts'].values())[0])} seeds")
            print([(k, v) for k, v in results['config'].items() if k.startswith('ros') or k.startswith('bpg')])
            exp_name = results['exp_name']
            estimations = results['estimations']
            eval_steps = results['eval_steps']

            for (collector_name, estimator_name), estimation in estimations.items():
                collector_name, estimator_name = switch_name(collector_name, estimator_name)
                collector_name += suffix

                label, color, linestyle = label_style(collector_name, estimator_name)
                for i, values in enumerate(estimation):
                    plt.plot(eval_steps[:len(values)], values, alpha=0.3, label=label if i == 0 else None, color=color, linestyle=linestyle)

                plt.hlines(1, eval_steps[0], eval_steps[-1] * 1.2, linestyles='--', alpha=0.2)
                plt.xlabel('Steps')
                plt.ylabel('Estimation')
                plt.title('estimations')
                plt.legend()
                plt.xscale('log')
                plt.tight_layout()
                plt.savefig(f'writing_aaai/figs/est_{exp_name}_{fig_i}.pdf', format='pdf')
                plt.clf()
                fig_i += 1


def draw_basic_others():
    final_mse = {}
    for target, target_name in (
            # ('RegressionImportanceSampling', 'ris'),
            ('WeightedRegressionImportanceSampling', 'wris'),
            ('FitterQ', 'fqe'),

    ):
        for group_i, jobs in enumerate(groups):
            for draw, draw_name in (
                    (draw_mean_sigma, 'mse'),
                    # (draw_median_quantile, 'mq'),
            ):
                for file, _, suffix in jobs:
                    wris_file = file.replace('MonteCarlo', target).replace('OrdinaryImportanceSampling', target)
                    for file in (
                            file,
                            wris_file,
                    ):
                        results = load(file)
                        if cache.setdefault(file, True):
                            cache[file] = False
                            print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
                        exp_name = results['exp_name']
                        estimations = results['estimations']
                        true_value = results['true_value']
                        eval_steps = results['eval_steps']

                        for (collector_name, estimator_name), estimation in estimations.items():
                            collector_name, estimator_name = switch_name(collector_name, estimator_name)
                            collector_name += suffix
                            label, color, linestyle = label_style(collector_name, estimator_name)
                            estimation = np.array(estimation)
                            mse = (estimation - true_value) ** 2
                            draw(eval_steps, mse, label, color, linestyle=linestyle, sigma_range=0.2)

                            final_mse[(exp_name, label)] = mse[..., -1].mean(), mse[..., -1].std() / (len(mse) ** 0.5)

                plt.legend()
                plt.xlabel('Steps')
                plt.ylabel('MSE')
                plt.yscale('log')
                plt.xscale('log')
                # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'writing_aaai/figs/basic_{target_name}_{draw_name}_{exp_name}.pdf', format='pdf')
                # plt.show()
                plt.clf()


def draw_comb_wris():
    final_mse = {}
    for group_i, jobs in enumerate(comb_groups):
        for draw, draw_name in (
                (draw_mean_sigma, 'mse'),
                # (draw_median_quantile, 'mq'),
        ):
            vline_y_max = 0
            vline_x = None
            for file, _, suffix, alpha, linestyle in jobs:
                wris_file = file.replace('_MonteCarlo', '_WeightedRegressionImportanceSampling').replace('_OrdinaryImportanceSampling', '_WeightedRegressionImportanceSampling')
                for file in (
                        file,
                        wris_file,
                ):
                    try:
                        results = load(file)
                    except:
                        continue
                    exp_name = results['exp_name']
                    estimations = results['estimations']
                    true_value = results['true_value']
                    eval_steps = results['eval_steps']

                    for (collector_name, estimator_name), estimation in estimations.items():
                        collector_name, estimator_name = switch_name(collector_name, estimator_name)
                        print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")

                        _, color, _ = label_style(collector_name, estimator_name)
                        collector_name += suffix
                        label, _, _ = label_style(collector_name, estimator_name)
                        estimation = np.array(estimation)
                        mse = (estimation - true_value) ** 2

                        vline_y_max = max(mse.mean(0).max(), vline_y_max)
                        if eval_steps[0] == 0:
                            vline_x = eval_steps[1]

                        draw(eval_steps, mse, suffix, color, linestyle=linestyle, sigma_range=0.2, alpha_alpha=alpha)
                        final_mse[(exp_name, suffix)] = mse[..., -1].mean(), mse[..., -1].std() / (len(mse) ** 0.5)

            # plt.vlines(vline_x, 0, vline_y_max * 2, linestyles='--', alpha=0.2)
            plt.legend()
            plt.xlabel('Steps')
            plt.ylabel('MSE')
            plt.yscale('log')
            plt.xscale('log')
            # plt.title(f'({"abcdefg"[group_i]}) {exp_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'writing_aaai/figs/comb_wris_{draw_name}_{exp_name}.pdf', format='pdf')
            plt.clf()


def draw_improvement_diff_pie():
    os_mse = None

    rand_group = (
        ('OnPolicySampler1', ''),
        ('RobustOnPolicySampler1', ''),
        ('RobustOnPolicyActing1', ''),
        # ('RobustOnPolicyActing2', ''),
    )
    for exp_name in (
            'MultiBanditM1S1',
            'GridWorldM1S0',
    ):
        for collector, suffix in rand_group:
            epsilons = np.arange(0, 11)
            estimations = []
            for eps in epsilons:
                file = f'results/{exp_name}_{eps}_{collector}_MonteCarlo'
                results = load(file)
                print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
                for (c_name, e_name), es in results['estimations'].items():
                    estimations.append(es)

            estimations = np.array(estimations)[..., 0].T
            c_name, e_name = switch_name(c_name, e_name)
            _, color, linestyle = label_style(c_name, e_name)
            c_name += suffix
            label, _, _ = label_style(c_name, e_name)
            mses = (estimations - 1) ** 2

            # draw_mean_sigma(epsilons / 10, mses, c_name, color, )
            # plt.xscale('log')
            # continue

            if c_name == 'OS':
                os_mse = mses
                continue
            # improve = 1 - mses / os_mse.mean(0)
            improve = mses / os_mse.mean(0)
            improve = improve.mean(0)[None, ...]
            draw_mean_sigma(epsilons / 10, improve, c_name, color, )

        plt.yscale('log')
        plt.legend()
        plt.xlabel('Epsilon')
        # plt.ylabel('Improvement')
        plt.ylabel('$\\frac{\\mathrm{MSE}}{\\mathrm{MSE}(\\mathrm{OS})}$', rotation=0, labelpad=20)

        import matplotlib.ticker as mtick

        # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/imp_pie_{exp_name}.pdf', format='pdf')
        # plt.show()
        plt.clf()

    # rand_group = (tuple((f'results/MultiBanditM1S1_5000_{collector}_MonteCarlo', None, None, None) for collector, _ in rand_group),)
    # draw_hyper_parameters(rand_group)


def draw_improvement_diff_env():
    rand_group = (
        ('OnPolicySampler1', ''),
        ('RobustOnPolicySampler1', ''),
        ('RobustOnPolicyActing1', ''),
        # ('RobustOnPolicyActing2', ''),
    )

    for exp_name in (
            'MultiBandit',
            'GridWorld',
    ):
        for changing, env_suffix, linestyle in (
                ('mean', 'M{}S1', '-'),
                ('scale', 'M1S{}', '--'),
        ):
            os_mse = None
            for collector, suffix in rand_group:
                factors = np.arange(1, 11)
                estimations = []

                for factor in factors:
                    file = f'results/{exp_name}{env_suffix.format(factor)}_5000_{collector}_MonteCarlo'
                    results = load(file)
                    print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
                    for (c_name, e_name), es in results['estimations'].items():
                        estimations.append(es)

                estimations = np.array(estimations)[..., 0].T
                c_name, e_name = switch_name(c_name, e_name)
                _, color, _ = label_style(c_name, e_name)
                c_name += suffix
                label, _, _ = label_style(c_name, e_name)
                mses = (estimations - 1) ** 2

                if c_name == 'OS':
                    os_mse = mses
                    continue
                # improve = 1 - mses / os_mse.mean(0)
                improve = mses / os_mse.mean(0)
                improve = improve.mean(0)[None, ...]
                draw_mean_sigma(factors, improve, f'{c_name} ({changing})', color, linestyle)

        plt.yscale('log')
        plt.legend()
        plt.xlabel('Factors')
        # plt.ylabel('Improvement')
        plt.ylabel('$\\frac{\\mathrm{MSE}}{\\mathrm{MSE}(\\mathrm{OS})}$', rotation=0, labelpad=20)

        import matplotlib.ticker as mtick

        # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.tight_layout()
        plt.savefig(f'writing_aaai/figs/imp_env_{exp_name}.pdf', format='pdf')
        # plt.show()
        plt.clf()


def draw_mean_sigma(x, values, label, color, linestyle=None, sigma_range=1., alpha_alpha=1.):
    values = np.array(values)

    y_mean = values.mean(0)
    y_std = values.std(0)
    y_mean_std = y_std / (values.shape[0] ** 0.5)
    # y_min = y_mean - y_mean_std * 1.96
    # y_max = y_mean + y_mean_std * 1.96
    y_min = y_mean - y_mean_std * 1.
    y_max = y_mean + y_mean_std * 1.

    plt.plot(x, y_mean, alpha=0.5 * alpha_alpha, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, y_min, y_max, alpha=0.1 * alpha_alpha, color=color)


def draw_median_quantile(x, values, label, color, linestyle=None, sigma_range=1., alpha_alpha=1.):
    values = np.array(values)

    y_median = np.median(values, 0)
    y_min = np.quantile(values, 0.25, 0)
    y_max = np.quantile(values, 0.75, 0)

    plt.plot(x, y_median, alpha=0.5 * alpha_alpha, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, y_min, y_max, alpha=0.1 * alpha_alpha, color=color)


def draw_hyper_parameters(exp_groups):
    configs = {}
    for group_i, jobs in enumerate(exp_groups):
        for item in jobs:
            file = item[0]
            results = load(file)
            exp_name = results['exp_name']
            config = results['config']
            collector_name = list(results['cross_entropy'].keys())[0]
            collector_name, _ = switch_name(collector_name, 'MC')

            exp_configs = configs.setdefault(exp_name, {})
            if collector_name == 'BPG':
                exp_configs['BPG - $k$'] = config['bpg_k']
                exp_configs['BPG - $\\alpha$'] = config['bpg_lr']
            elif collector_name == 'ROS':
                exp_configs['ROS - $\\alpha$'] = config['ros_lr']
            elif collector_name == 'ROA':
                exp_configs['ROA - $\\rho$'] = config['ros_epsilon']
                exp_configs['ROA - $m$'] = config['ros_n_action'] if 'Continuous' in exp_name else '/'

    titles = list(exp_configs.keys())
    print('\\hline')
    print('Donmain & ' + ' & '.join(titles) + ' \\\\')
    print('\\hline')
    for exp_name, exp_configs in configs.items():
        print(f'{exp_name} & {" & ".join(f"{exp_configs[t]}" for t in titles)} \\\\')
    print('\\hline')


def new_plot(x, y, **kwargs):
    plot(x[1:], y[1:], **kwargs)


def new_fill_between(x, y_min, y_max, **kwargs):
    fill_between(x[1:], y_min[1:], y_max[1:], **kwargs)


if __name__ == '__main__':
    dist_pieDb()
    mc_confidence()
    draw_mab()
    draw_comb()
    draw_comb_wris()
    draw_improvement_diff_pie()
    draw_improvement_diff_env()

    plot = plt.plot
    fill_between = plt.fill_between
    plt.plot = new_plot
    plt.fill_between = new_fill_between

    draw_alpha()
    draw_rho()
    draw_m()
    draw_basic()
    draw_basic_others()
    draw_grad()
    draw_est()

    time.sleep(1)
