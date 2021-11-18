import torch
import traceback
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

from commons import load

linestyles = ['-', '--', '-.', ':', ]
colors = list(matplotlib.colors.TABLEAU_COLORS.values())
# collector_names = ['OnPolicy', 'ROS', 'BPG', 'UCB', 'UTF', 'Combined']
collector_names = ['OS', 'ROS', 'BPG', 'ROA', 'Combined', 'SROS', 'UTF', 'EPS']
estimator_names = ['MC', 'FQE', 'MB', '', 'OIS', 'RIS', '', '', '', 'WRIS']

estimation_path_i = 0
loss_path_i = 0
loss_true_on_policy = []
loss_drawn = []
avg_drawn = []
lr_drawn = []
cache = {}


def label_style(collector_name, estimator_name):
    if collector_name not in collector_names:
        collector_names.append(collector_name)
    if estimator_name not in estimator_names:
        estimator_names.append(estimator_name)

    color = colors[collector_names.index(collector_name) % len(colors)]
    linestyle = linestyles[estimator_names.index(estimator_name) % len(linestyles)]

    return f"{collector_name} - {estimator_name}", color, linestyle


def switch_name(collector_name, estimator_name):
    if collector_name == 'OnPolicySampler':
        collector_name = 'OS'
    if collector_name == 'PolicySampler':
        collector_name = 'OS'
    if collector_name == 'CombinedExplorer':
        collector_name = 'Combined'
    elif collector_name == 'MinMSEActing':
        collector_name = 'MMA'
    elif collector_name == 'RobustOnPolicySampler':
        collector_name = 'ROS'
    elif collector_name == 'DirectedRobustOnPolicySampler':
        collector_name = 'SROS'
    elif collector_name == 'RobustOnPolicySampler2':
        collector_name = 'ROS'
    elif collector_name == 'RobustOnPolicySampler3':
        collector_name = 'ROS'
    elif collector_name == 'RobustOnPolicyActing':
        collector_name = 'ROA'
    elif collector_name == 'BehaviorPolicyGradient':
        collector_name = 'BPG'
    elif collector_name == 'UCBExplorer':
        collector_name = 'UCB'
    elif collector_name == 'UnTestFirstExplorer':
        collector_name = 'UTF'
    elif collector_name == 'EpsilonPolicySampler':
        collector_name = 'EPS'

    if estimator_name == 'MonteCarlo':
        estimator_name = "MC"
    elif estimator_name == 'FitterQ':
        estimator_name = 'FQE'
    elif estimator_name == 'ModelBased':
        estimator_name = 'MB'
    elif estimator_name == 'OrdinaryImportanceSampling':
        estimator_name = 'OIS'
    elif estimator_name == 'RegressionImportanceSampling':
        estimator_name = 'RIS'
    elif estimator_name == 'WeightedRegressionImportanceSampling':
        estimator_name = 'WRIS'

    return collector_name, estimator_name


def plot(estimations, true_value, eval_steps, exp_name, training_episodes, draw_collectors, suffix, sa_counts=None, sta_counts=None, cross_entropy=None, avg_grad=None, ros_lr=None, correction_term=None, prh_sum=None, kl_divergence=None, alpha_alpha=1., line_sty=None, **kwargs):
    if len(draw_collectors) == 0:
        draw_collectors = collector_names

    global estimation_path_i, loss_path_i, loss_true_on_policy
    eval_steps = np.array(eval_steps)

    plt.subplot(331)
    for (collector_name, estimator_name), estimation in estimations.items():
        collector_name, estimator_name = switch_name(collector_name, estimator_name)
        if collector_name not in draw_collectors or estimator_name not in draw_estimators:
            continue
        label, color, linestyle = label_style(collector_name, estimator_name)
        collector_name += suffix
        label = label_style(collector_name, estimator_name)[0]

        estimation = estimation[:-1] if len(estimation) > 1 and len(estimation[-1]) != len(estimation[-2]) else estimation
        estimation = np.array(estimation)[:n]
        mse = (estimation - true_value) ** 2
        draw_mean_sigma(eval_steps, mse, label, color, linestyle=linestyle if line_sty is None else line_sty, sigma_range=0., alpha_alpha=alpha_alpha)

    plt.legend()
    if eval_steps[0] == 0 and not cache.setdefault('est_vline', False):
        plt.vlines(eval_steps[1], 0, 1, linestyles='--', alpha=0.2)
        cache['est_vline'] = True
    plt.title(f"{exp_name} (training_episodes of $\\pi_e$ is {training_episodes})")
    plt.xlabel('steps')
    plt.ylabel('MSE')
    # plt.ylim([1e-3, 1e-1])
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()

    # todo draw estimation
    plt.subplot(332)
    for (collector_name, estimator_name), estimation in estimations.items():
        collector_name, estimator_name = switch_name(collector_name, estimator_name)
        if collector_name not in draw_collectors or estimator_name not in draw_estimators:
            continue
        label, color, linestyle = label_style(collector_name, estimator_name)
        collector_name += suffix
        label = label_style(collector_name, estimator_name)[0]

        for i, values in enumerate(estimation[:n][:50]):
            # plt.plot(eval_steps[:len(values)], values, alpha=0.3, label=label if i == 0 else None, color=color, linestyle=linestyle)
            plt.plot(estimation_path_i * eval_steps[-1] + eval_steps[:len(values)], values, alpha=0.3 * alpha_alpha, label=label if i == 0 else None, color=color, linestyle=linestyle if line_sty is None else line_sty)
        estimation_path_i += 1

    # plt.hlines(1, eval_steps[0], eval_steps[-1], linestyles='--', alpha=0.2)
    plt.hlines(1, eval_steps[0], eval_steps[-1] * estimation_path_i * 1.01, linestyles='--', alpha=0.2)
    plt.xlabel('steps')
    plt.ylabel('estimation')
    plt.title('estimations')
    plt.legend()
    # plt.xscale('log')
    plt.tight_layout()

    # # todo draw count
    # plt.subplot(334)
    # plt.title('len(set(state-action))')
    # plt.xlabel('steps')
    # plt.ylabel('count')
    # for collector_name, counts in sa_counts.items():
    #     collector_name, _ = switch_name(collector_name, '')
    #     if collector_name not in draw_collectors:
    #         continue
    #     collector_name += suffix
    #     label, color, linestyle = label_style(collector_name, "MC")
    #     plt.plot(eval_steps, np.mean(counts[:n], 0), label=collector_name, alpha=0.5, color=color)
    # plt.legend()
    # plt.xscale('log')
    # plt.tight_layout()
    #
    # plt.subplot(335)
    # plt.title('len(set(state-timestep-action))')
    # plt.xlabel('steps')
    # plt.ylabel('count')
    # for collector_name, counts in sta_counts.items():
    #     collector_name, _ = switch_name(collector_name, '')
    #     if collector_name not in draw_collectors:
    #         continue
    #     collector_name += suffix
    #     label, color, linestyle = label_style(collector_name, "MC")
    #     plt.plot(eval_steps, np.mean(counts[:n], 0), label=collector_name, alpha=0.5, color=color)
    # plt.legend()
    # # plt.xscale('log')
    # plt.tight_layout()

    loss_avg_drawn = []
    plt.subplot(337)
    for collector_name, loss in cross_entropy.items():
        collector_name, _ = switch_name(collector_name, '')
        if collector_name not in draw_collectors:
            continue
        label, color, linestyle = label_style(collector_name, "MC")
        collector_name += suffix

        if collector_name in loss_drawn:
            continue
        loss_avg_drawn.append(collector_name)

        loss = np.array(loss)[:n]
        mse = (loss - true_value) ** 2

        draw_mean_sigma(eval_steps, mse, collector_name, color, linestyle=linestyle if line_sty is None else line_sty, sigma_range=0., alpha_alpha=alpha_alpha)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('Sampling error')
    plt.tight_layout()

    # plt.subplot(338)
    # for collector_name, loss in cross_entropy.items():
    #     collector_name, _ = switch_name(collector_name, '')
    #     if collector_name not in draw_collectors:
    #         continue
    #     collector_name += suffix
    #
    #     if collector_name in loss_drawn:
    #         continue
    #     loss_avg_drawn.append(collector_name)
    #
    #     label, color, linestyle = label_style(collector_name, "MC")
    #     for i, values in enumerate(loss[:n][:50]):
    #         plt.plot(loss_path_i * eval_steps[-1] + eval_steps[:len(values)], values, alpha=0.3, label=collector_name if i == 0 else None, color=color)
    #     loss_path_i += 1
    #     plt.hlines(1, eval_steps[0], eval_steps[-1] * loss_path_i * 1.01, linestyles='--', alpha=0.2)
    #
    # plt.ylabel('Normalized cross-entropy')
    # plt.xlabel('steps')
    # # plt.xscale('log')
    # plt.legend()
    # plt.tight_layout()
    #
    # loss_drawn.extend(loss_avg_drawn)

    if avg_grad is not None:
        plt.subplot(339)
        plt.xlabel('steps')
        plt.ylabel('$|\\nabla|$')
        for collector_name, values in avg_grad.items():
            collector_name, _ = switch_name(collector_name, '')
            if collector_name not in draw_collectors:
                continue
            label, color, linestyle = label_style(collector_name, "MC")
            collector_name += suffix

            if collector_name in avg_drawn:
                continue
            avg_drawn.append(collector_name)

            values = np.mean(values[:n], 0)
            plt.plot(eval_steps, values, label=collector_name, alpha=0.5 * alpha_alpha, color=color, linestyle=linestyle if line_sty is None else line_sty)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

    # if ros_lr is not None:
    #     plt.subplot(336)
    #     plt.xlabel('steps')
    #     plt.ylabel('lr')
    #     for collector_name, values in ros_lr.items():
    #         collector_name, _ = switch_name(collector_name, '')
    #         if collector_name not in draw_collectors:
    #             continue
    #         collector_name += suffix
    #
    #         if collector_name in lr_drawn:
    #             continue
    #         lr_drawn.append(collector_name)
    #
    #         label, color, linestyle = label_style(collector_name, "MC")
    #         values = np.mean(values[:n], 0)
    #         plt.plot(eval_steps, values, label=collector_name, alpha=0.5, color=color)
    #     plt.legend()
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.tight_layout()

    # if correction_term is not None:
    #     plt.subplot(333)
    #     plt.xlabel('steps')
    #     plt.ylabel('correction_term')
    #     for collector_name, values in correction_term.items():
    #         collector_name, _ = switch_name(collector_name, '')
    #         if collector_name not in draw_collectors:
    #             continue
    #         collector_name += suffix
    #
    #         ct_drawn = cache.setdefault('ct_drawn', [])
    #         if collector_name in ct_drawn:
    #             continue
    #         ct_drawn.append(collector_name)
    #
    #         label, color, linestyle = label_style(collector_name, "MC")
    #         values = np.mean(values[:n], 0)
    #         plt.plot(eval_steps, values, label=collector_name, alpha=0.5, color=color)
    #     plt.legend()
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.tight_layout()

    # if prh_sum is not None:
    #     plt.subplot(336)
    #     plt.xlabel('steps')
    #     plt.ylabel('Pr(H)')
    #     for collector_name, values in prh_sum.items():
    #         collector_name, _ = switch_name(collector_name, '')
    #         if collector_name not in draw_collectors:
    #             continue
    #         collector_name += suffix
    #
    #         if collector_name in lr_drawn:
    #             continue
    #         lr_drawn.append(collector_name)
    #
    #         label, color, linestyle = label_style(collector_name, "MC")
    #         values = np.mean(values[:n], 0)
    #         plt.plot(eval_steps, values, label=collector_name, alpha=0.5, color=color)
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.tight_layout()

    if kl_divergence is not None:
        plt.subplot(338)
        plt.xlabel('steps')
        plt.ylabel('KL')
        for collector_name, values in kl_divergence.items():
            collector_name, _ = switch_name(collector_name, '')
            if collector_name not in draw_collectors:
                continue
            label, color, linestyle = label_style(collector_name, "MC")
            collector_name += suffix

            kl_drawn = cache.setdefault('kl_drawn', [])
            if collector_name in kl_drawn:
                continue
            kl_drawn.append(collector_name)

            draw_mean_sigma(eval_steps, values[:n], collector_name, color, linestyle=linestyle if line_sty is None else line_sty, sigma_range=0.5)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()


def draw_mean_sigma(x, values, label, color, linestyle=None, sigma_range=1., alpha_alpha=1.):
    values = np.array(values)

    y_mean = values.mean(0)
    y_std = values.std(0)
    y_mean_std = y_std / values.shape[0]
    y_min = y_mean - y_mean_std * 1.96
    y_max = y_mean + y_mean_std * 1.96

    plt.plot(x, y_mean, alpha=0.5 * alpha_alpha, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, y_min, y_max, alpha=0.1 * alpha_alpha, color=color)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    draw_estimators = (
        'MC',
        'OIS',
        'FQE',
        'MB',
        'RIS',
        'WRIS',
    )
    n = 100000
    print(n)
    plt.figure(figsize=(24, 18))
    for file_i, (file, draw_collectors, suffix, alpha, line_sty) in enumerate((
            (f'results/CartPole_1000_OnPolicySampler1_WeightedRegressionImportanceSampling', (), '', 2.0, '-'),
            (f'results/CartPole_1000_OnPolicySampler1_MonteCarlo', (), '', 2.0, '--'),
            (f'results/CartPole_1000_RobustOnPolicyActing4_WeightedRegressionImportanceSampling', (), '', 2.0, '-'),
            (f'results/CartPole_1000_RobustOnPolicyActing4_MonteCarlo', (), '', 2.0, '--'),
    )):
        # file += '_test'

        results = load(file)
        print(f"{file}: {len(list(results['cross_entropy'].values())[0])} seeds")
        print(results['config'])
        results['suffix'] = suffix
        results['draw_collectors'] = draw_collectors
        results['alpha_alpha'] = alpha
        results['line_sty'] = line_sty
        plot(**results)
    # plt.savefig(f'results/0000.pdf', format='pdf')
    plt.show()
