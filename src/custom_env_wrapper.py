import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np

class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        """
        The reward parameter is the default reward
        """
        position, velocity = self.unwrapped.state

        shaping = self.cfg.get("reward_shaping", {})
        position_coeff = shaping.get("position_coeff", 20.0)
        velocity_coeff = shaping.get("velocity_coeff", 1.0)
        energy_penalty = shaping.get("energy_penalty", 0.1)

        new_reward = reward

        # mountain car positions are roughly [-1.2, 0.6]
        # adding 0.55 centers it so it makes the reward mostly positive
        new_reward += position_coeff * (position + 0.55)

        # only adds reward if the car is moving to the goal (right)
        if velocity > 0:
            new_reward += velocity_coeff * velocity

        # retrieves the last action made by agent
        # np.square() penalizes big pushes more heavily
        if hasattr(self.env, "action_space"):
            last_action = self.env.unwrapped.last_action if hasattr(self.env.unwrapped, "last_action") else 0
            new_reward -= energy_penalty * np.square(last_action).sum()

        return float(new_reward)