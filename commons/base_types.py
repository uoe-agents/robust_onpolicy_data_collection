class Policy:
    def action_dist(self, state):
        raise NotImplementedError

    def discrete_action_dist(self, state, n_action=None):
        raise NotImplementedError

    def action_prob(self, state, action):
        raise NotImplementedError

    def sample(self, state):
        raise NotImplementedError


class DataCollector:
    def __init__(self, pi_e: Policy, **kwargs) -> None:
        self.pi_e = pi_e

    def act(self, time_step, state):
        raise NotImplementedError

    def update(self, time_step, state, action, reward, next_state, done, pie_a, pib_a, trajectories=None):
        raise NotImplementedError


class ValueEstimator:
    def __init__(self, pi_e: Policy, gamma=1., **kwargs) -> None:
        super().__init__()
        self.pi_e = pi_e
        self.gamma = gamma

    def estimate(self, trajectories, pi_D=None):
        raise NotImplementedError

    # def update(self, transitions):
    #     raise NotImplementedError
    #
    # def estimate_q_values(self, time_step, state, actions):
    #     raise NotImplementedError
