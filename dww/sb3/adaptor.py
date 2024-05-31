import gymnasium as gym
from pettingzoo.sisl import waterworld_v4


class WaterworldAdaptor(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = waterworld_v4.env(*args, **kwargs, n_pursuers=1)
        self.action_space = self.env.action_space("pursuer_0")
        self.observation_space = self.env.observation_space("pursuer_0")

    def step(self, action):
        observation, reward, termination, truncation, info = self.env.last()

        if termination or truncation:
            action = None
        self.env.step(action)

        return observation, reward, termination, truncation, info

    def reset(self):
        self.env.reset()
        return self.env.last()[0], {}

    def render(self):
        self.env.render()

    def close(self):
        ...
