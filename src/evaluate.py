# import gymnasium as gym
# from stable_baselines3 import SAC, PPO, A2C, DQN
# import numpy as np
# import os
#
# ALGOS = {
#     "SAC": SAC,
#     "PPO": PPO,
#     "A2C": A2C,
#     "DQN": DQN
# }
#
# def evaluate(model_path, env_id, episodes=10):
#     algo_name = os.path.basename(model_path).split("_")[0]
#     algo_class = ALGOS[algo_name]
#     model = algo_class.load(model_path)
#     env = gym.make(env_id)
#     rewards = []
#
#     for ep in range(episodes):
#         obs = env.reset()[0]
#         finished = False
#         ep_reward = 0.0
#
#         while not finished:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             finished = terminated or truncated
#             ep_reward += reward
#
#         rewards.append(ep_reward)
#         print(f"Episode {ep + 1}: Reward = {ep_reward:.2f}")
#
#     mean_reward = np.mean(rewards)
#     std_reward = np.std(rewards)
#
#     print(f"\nEvaluation over {episodes} episodes: mean_reward = {mean_reward:.2f}, std_reward = {std_reward:.2f}")
#     return mean_reward, std_reward
#
#
# if __name__ == "__main__":
#     implementations = ["baseline", "custom", "extension"]
#     env_id = "MountainCarContinuous-v0"
#
#     for impl in implementations:
#         print(f"\nEvaluating {impl} implementation:")
#         results_dir = f"results/{impl}"
#         for trial_file in sorted(os.listdir(results_dir)):
#             model_path = os.path.join(results_dir, trial_file)
#             evaluate(model_path, env_id)
