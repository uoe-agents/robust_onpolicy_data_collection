import copy
import numpy as np

from policy import BehaviorCloner, REINFORCE
from commons import ValueEstimator, Policy
from commons import finished_trajectories


def get_weights_returns(trajectories, gamma, pi_D=None, normalization=False):
    gs = []
    pie_weights = []
    pib_weights = []
    for trajectory in trajectories:
        g, pr_e, pr_b = one_trajectory(trajectory, gamma, pi_D)
        gs.append(g)
        pie_weights.append(pr_e)
        pib_weights.append(pr_b)

    gs = np.array(gs)
    pie_weights = np.array(pie_weights)
    pib_weights = np.array(pib_weights)
    return gs, pie_weights, pib_weights


def one_trajectory(trajectory, gamma, pi_D=None):
    g = 0
    pr_e = 1
    pr_b = 1
    for t, s, a, r, s_, d, pie_a, pib_a in reversed(trajectory):
        g = g * gamma + r
        pr_e *= pie_a
        pr_b *= pib_a
    return g, pr_e, pr_b


class MonteCarlo(ValueEstimator):
    def estimate(self, trajectories, pi_D=None):
        trajectories = finished_trajectories(trajectories)
        if not len(trajectories):
            return 0

        gs, pie_weights, pib_weights = get_weights_returns(trajectories, self.gamma, )
        return gs.mean()


class OrdinaryImportanceSampling(ValueEstimator):

    def estimate(self, trajectories, pi_D=None):
        trajectories = finished_trajectories(trajectories)
        if not len(trajectories):
            return 0

        gs, pie_weights, pib_weights = get_weights_returns(trajectories, self.gamma)
        weights = pie_weights / pib_weights
        return (gs * weights).mean()


class WeightedImportanceSampling(ValueEstimator):

    def estimate(self, trajectories, pi_D=None):
        trajectories = finished_trajectories(trajectories)
        if not len(trajectories):
            return 0

        gs, pie_weights, pib_weights = get_weights_returns(trajectories, self.gamma)
        weights = pie_weights / pib_weights

        # from commons import save, Global
        # exp_name, build_collectors = Global["ir_weights"]
        # save(f'results/{exp_name}_{build_collectors[0].__name__}.pkl', (pie_weights, pib_weights))
        # exit()

        return (gs * weights).sum() / weights.sum()


class RegressionImportanceSampling(ValueEstimator):

    def __init__(self, pi_e: REINFORCE, gamma=1., **kwargs) -> None:
        super().__init__(pi_e, gamma, **kwargs)
        self.pi_e = pi_e
        self.gamma = gamma
        self.IS = OrdinaryImportanceSampling(pi_e, gamma, **kwargs)

    def clean(self, trajectories, pi_D):
        trajectories = finished_trajectories(trajectories)

        trs = []
        for trajectory in trajectories:
            tr = []
            trs.append(tr)
            for t, s, a, r, s_, d, pie_a, pib_a in trajectory:
                piD_a = pi_D.action_prob(s, a)
                tr.append((
                    t, s, a, r, s_, d, pie_a, piD_a
                ))

        return trs

    def estimate(self, trajectories, pi_D):
        assert pi_D is not None
        trs = self.clean(trajectories, pi_D)
        return self.IS.estimate(trs)


class WeightedRegressionImportanceSampling(RegressionImportanceSampling):
    def __init__(self, pi_e: REINFORCE, gamma=1., **kwargs) -> None:
        super().__init__(pi_e, gamma, **kwargs)
        self.IS = WeightedImportanceSampling(pi_e, gamma, **kwargs)
