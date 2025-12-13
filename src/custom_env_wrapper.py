import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np

class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg
        self.last_action = 0

    def step(self, action):
        self.last_action = action
        return super().step(action)

    def reward(self, reward: float) -> float:
        position, velocity = self.unwrapped.state

        shaping = self.cfg.get("reward_shaping", {})
        goal_position = shaping.get("goal_position", 0.45)
        position_scale = shaping.get("position_scale", 100.0)
        velocity_scale = shaping.get("velocity_scale", 10.0)
        energy_penalty = shaping.get("energy_penalty", 0.01)
        success_bonus = shaping.get("success_bonus", 1000.0)

        # 1. Distance to goal reward (encourages moving toward 0.45)
        # As position approaches goal, this becomes more positive
        distance_to_goal = goal_position - position
        position_reward = -position_scale * distance_to_goal

        # 2. Velocity bonus (only reward rightward velocity when left of goal)
        # This encourages building momentum in the right direction
        velocity_reward = 0.0
        if position < goal_position and velocity > 0:
            velocity_reward = velocity_scale * velocity

        # 3. Energy penalty (discourage wasteful large actions)
        if isinstance(self.last_action, np.ndarray):
            energy_cost = energy_penalty * np.sum(np.square(self.last_action))
        else:
            energy_cost = energy_penalty * (self.last_action ** 2)

        # 4. Success bonus (large reward for reaching goal)
        success_reward = success_bonus if position >= goal_position else 0.0

        new_reward = position_reward + velocity_reward - energy_cost + success_reward

        return float(new_reward)