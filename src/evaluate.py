import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN
from custom_env_wrapper import CustomRewardWrapper
from utils import load_config

# Supported algorithms
ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

def evaluate_model(model, env, episodes=10):
    """Evaluate a trained model deterministically and compute client-friendly metrics."""

    rewards = []
    ep_lengths = []
    energies = []
    successes = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        ep_len = 0
        total_energy = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            ep_len += 1

            # Energy: sum of squared actions (works for scalar or array actions)
            if isinstance(action, np.ndarray):
                total_energy += np.sum(np.square(action))
            else:
                total_energy += action ** 2

            done = terminated or truncated

        rewards.append(total_reward)
        ep_lengths.append(ep_len)
        energies.append(total_energy)

        # Success detection for MountainCarContinuous-v0 (goal position >= 0.45)
        if obs[0] >= 0.45:
            successes += 1

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_ep_len = np.mean(ep_lengths)
    mean_energy = np.mean(energies)
    success_rate = successes / episodes * 100

    return mean_reward, std_reward, mean_ep_len, mean_energy, success_rate


def eval_all(root_dir, env_name, experiments, num_trials=3, eval_episodes=10):
    """Evaluate all models inside the results folders with key metrics."""
    results = {}

    for exp in experiments:
        exp_path = os.path.join(root_dir, exp)
        if not os.path.isdir(exp_path):
            print(f"[WARNING] Folder not found: {exp_path}. Skipping.")
            continue

        print(f"\n===== Evaluating {exp.upper()} =====")

        cfg_path = f"../config/config_{exp}.yaml"
        if os.path.exists(cfg_path):
            cfg = load_config(cfg_path)
            max_steps = cfg.get('max_episode_steps', 999)
            use_custom_reward = (exp in ['custom', 'extension'])
            print(f"  Config: max_steps={max_steps}, custom_reward={use_custom_reward}")
        else:
            print(f"  [WARNING] Config not found: {cfg_path}, using defaults")
            max_steps = 999
            use_custom_reward = False
            cfg = {}

        exp_results = []

        for trial in range(1, num_trials + 1):
            model_filename = None
            for file in os.listdir(exp_path):
                if file.endswith(f"trial{trial}.zip"):
                    model_filename = file
                    break

            if model_filename is None:
                print(f"   No model found for trial {trial} in {exp_path}")
                continue

            model_path = os.path.join(exp_path, model_filename)
            algo_name = model_filename.split("_")[0]
            algo_class = ALGOS.get(algo_name)

            if algo_class is None:
                print(f"   Unknown algorithm in file: {model_filename}")
                continue

            print(f"  Evaluating {model_filename} ...")
            model = algo_class.load(model_path)

            env = gym.make(env_name, max_episode_steps=max_steps)

            if use_custom_reward:
                env = CustomRewardWrapper(env, cfg)

            mean_reward, std_reward, mean_ep_len, mean_energy, success_rate = evaluate_model(
                model, env, eval_episodes
            )
            print(f"    ->Mean Reward: {mean_reward:.2f}, Std: {std_reward:.2f}, "
                  f"Avg Length: {mean_ep_len:.1f}, Avg Energy: {mean_energy:.2f}, "
                  f"Success Rate: {success_rate:.1f}%")

            exp_results.append((mean_reward, std_reward, mean_ep_len, mean_energy, success_rate))

            env.close()

        results[exp] = exp_results

    return results


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(ROOT, "results")

    final_results = eval_all(
        root_dir=RESULTS_DIR,
        env_name="MountainCarContinuous-v0",
        experiments=("baseline", "custom", "extension"),
        num_trials=3,
        eval_episodes=10
    )

    # Summary printout
    print("\n\n========== FINAL SUMMARY ==========")
    for exp, scores in final_results.items():
        if not scores:
            continue

        mean_rewards = [m for m, s, l, e, sr in scores]
        std_rewards = [s for m, s, l, e, sr in scores]
        mean_lengths = [l for m, s, l, e, sr in scores]
        mean_energies = [e for m, s, l, e, sr in scores]
        success_rates = [sr for m, s, l, e, sr in scores]

        print(f"\n{exp.upper()}:")

        print(f"  Overall Mean Reward: {np.mean(mean_rewards):.2f}")
        print(f"  Overall Reward Std: {np.mean(std_rewards):.2f}")
        print(f"  Overall Avg Episode Length: {np.mean(mean_lengths):.1f}")
        print(f"  Overall Avg Energy Used: {np.mean(mean_energies):.2f}")
        print(f"  Overall Success Rate: {np.mean(success_rates):.1f}%")

