import gym
import numpy as np
from gym.spaces import Discrete, Box


class GridWorld(gym.Env):
    Left = 0
    Right = 1
    Down = 2
    Up = 3
    MaxStep = 100
    gamma = 1

    action_space = Discrete(4)
    observation_space = Discrete(4 * 4)

    def __init__(self, mean_factor=1., scale_noise=0.) -> None:
        super().__init__()
        self.ready = False
        self.time_step = 0
        self.state = np.zeros(2, np.int)

        self.terminal_state = np.array([3, 3])
        self.trap_state = np.array([1, 1])
        self.reward_state = np.array([1, 3])

        self.mean_factor = mean_factor
        self.scale_noise = scale_noise

    def step(self, action: int):
        assert self.ready, "please reset."
        assert action in (GridWorld.Up, GridWorld.Down, GridWorld.Left, GridWorld.Right)
        self.time_step += 1
        old_state: np.ndarray = self.state.copy()

        i = action // 2
        j = action % 2
        self.state[i] += 1 if j else -1
        self.state = self.state.clip(0, 3)

        d = self.time_step == GridWorld.MaxStep

        if (self.state == old_state).all():
            r = -1
        else:
            if (self.state == self.terminal_state).all():
                r, d = 10, True
            elif (self.state == self.trap_state).all():
                r = -10
            elif (self.state == self.reward_state).all():
                r = 1
            else:
                r = -1

        if d:
            self.ready = False

        r *= self.mean_factor
        if self.scale_noise:
            r += self.scale_noise * (2 * np.random.random() - 1)  # [-scale_noise, scale_noise]

        return self.to_observation(), r, d, ''

    def to_observation(self):
        # return self.state.copy()
        return self.state[0] * 4 + self.state[1]

    def to_state(self, state):
        # return state
        return np.array([state // 4, state % 4])

    def reset(self):
        self.state = np.zeros(2, np.int)
        self.ready = True
        self.time_step = 0
        return self.to_observation()

    def render(self, mode='human'):
        return
        image = np.zeros((4, 4), np.str)

        image[3 - self.terminal_state[1], self.terminal_state[0]] = "T"
        image[3 - self.trap_state[1], self.trap_state[0]] = "t"
        image[3 - self.reward_state[1], self.reward_state[0]] = "r"
        image[3 - self.state[1], self.state[0]] = "X"
        print(image)

    def seed(self, seed=None):
        np.random.seed(seed)


def play():
    env = GridWorld()
    s = env.reset()
    d = False
    while not d:
        env.render()
        a = env.action_space.sample()
        s_, r, d, _ = env.step(a)
        print(a, r)
        s = s_


if __name__ == '__main__':
    play()
