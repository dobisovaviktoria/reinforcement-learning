import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, DQN
from custom_env_wrapper import CustomRewardWrapper
from utils import load_config

ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

def evaluate(model_path, env_id, episodes=10):

    # Containers for metrics
    rewards = []
    ep_lengths = []
    energies = []
    successes = 0
    reward_per_energy_list = []
    steps_to_success = []
    energy_per_success = []
    action_magnitudes = []
    action_smoothness = []
    max_positions = []

    for ep in range(episodes):
        obs, _ = env_id.reset(seed=42 + ep)
        finished = False
        total_reward = 0
        total_energy = 0
        ep_len = 0
        prev_action = None
        max_pos = obs[0]  # track max position for learning progress

        while not finished:
            action, _ = model_path.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_id.step(action)

            # Update basic metrics
            total_reward += reward
            ep_len += 1

            # Energy calculation
            if isinstance(action, np.ndarray):
                total_energy += np.sum(np.square(action))
                action_magnitudes.append(np.linalg.norm(action))
            else:
                total_energy += action ** 2
                action_magnitudes.append(abs(action))

            # Action smoothness (difference from previous action)
            if prev_action is not None:
                if isinstance(action, np.ndarray):
                    action_smoothness.append(np.linalg.norm(action - prev_action))
                else:
                    action_smoothness.append(abs(action - prev_action))
            prev_action = action

            # Update max position reached
            max_pos = max(max_pos, obs[0])

            # Episode finished
            finished = terminated or truncated

        # Append per-episode metrics
        rewards.append(total_reward)
        ep_lengths.append(ep_len)
        energies.append(total_energy)
        reward_per_energy_list.append(total_reward / (total_energy + 1e-8))  # avoid dividing by 0
        max_positions.append(max_pos)

        # Success detection for MountainCarContinuous-v0 (goal >= 0.45)
        if obs[0] >= 0.45:
            successes += 1
            steps_to_success.append(ep_len)
            energy_per_success.append(total_energy)

    # Aggregate metrics across all episodes
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_ep_len = np.mean(ep_lengths)
    mean_energy = np.mean(energies)
    success_rate = successes / episodes * 100
    mean_reward_per_energy = np.mean(reward_per_energy_list)
    mean_steps_to_success = np.mean(steps_to_success) if steps_to_success else np.nan
    mean_energy_per_success = np.mean(energy_per_success) if energy_per_success else np.nan
    mean_action_magnitude = np.mean(action_magnitudes) if action_magnitudes else np.nan
    mean_action_smoothness = np.mean(action_smoothness) if action_smoothness else np.nan
    mean_max_position = np.mean(max_positions)

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_ep_len": mean_ep_len,
        "mean_energy": mean_energy,
        "success_rate": success_rate,
        "reward_per_energy": mean_reward_per_energy,
        "mean_steps_to_success": mean_steps_to_success,
        "mean_energy_per_success": mean_energy_per_success,
        "mean_action_magnitude": mean_action_magnitude,
        "mean_action_smoothness": mean_action_smoothness,
        "mean_max_position": mean_max_position
    }


def eval_all(root_dir, env_name, experiments, num_trials=3, eval_episodes=10):

    results = {}

    for exp in experiments:
        exp_path = os.path.join(root_dir, exp)
        if not os.path.isdir(exp_path):
            print(f"WARNING: Folder not found: {exp_path}. Skipping.")
            continue

        print(f"\n EVALUATING... \n {exp.upper()} \n")

        # Load config if exists
        cfg_path = f"config/config_{exp}.yaml"
        if os.path.exists(cfg_path):
            cfg = load_config(cfg_path)
            max_steps = cfg.get('max_episode_steps', 999)
            use_custom_reward = (exp in ['custom', 'extension'])
            print(f"Config: max_steps={max_steps}, custom_reward={use_custom_reward}")
        else:
            print(f"WARNING: Config not found: {cfg_path}, using defaults")
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

            algo_name = model_filename.split("_")[0]
            algo_class = ALGOS.get(algo_name)

            if algo_class is None:
                print(f"Unknown algorithm in file: {model_filename}")
                continue

            print(f"Evaluating {model_filename} ...")
            model = algo_class.load(os.path.join(exp_path, model_filename))

            env = gym.make(env_name, max_episode_steps=max_steps)
            if use_custom_reward:
                env = CustomRewardWrapper(env, cfg)

            metrics = evaluate(model, env, eval_episodes)

            # Print metrics per trial
            print(f"-> Mean Reward: {metrics['mean_reward']:.2f}, Std: {metrics['std_reward']:.2f}")
            print(f"-> Avg Episode Length: {metrics['mean_ep_len']:.1f}")
            print(f"-> Avg Energy Used: {metrics['mean_energy']:.2f}")
            print(f"-> Success Rate: {metrics['success_rate']:.1f}%")
            print(f"-> Reward per Unit Energy: {metrics['reward_per_energy']:.3f}")
            print(f"-> Steps to Success: {metrics['mean_steps_to_success']:.1f}")
            print(f"-> Energy per Success: {metrics['mean_energy_per_success']:.2f}")
            print(f"-> Mean Action Magnitude: {metrics['mean_action_magnitude']:.3f}")
            print(f"-> Mean Action Smoothness: {metrics['mean_action_smoothness']:.3f}")
            print(f"-> Max Position Reached: {metrics['mean_max_position']:.3f}")

            exp_results.append(metrics)
            env.close()

        results[exp] = exp_results

    # Final summary across all trials
    print("\n\nFINAL SUMMARY: ")
    for exp, scores in results.items():
        if not scores:
            continue

        print(f"\n{exp.upper()}:")
        for key in scores[0].keys():
            values = [m[key] for m in scores]
            print(f"Overall {key.replace('_', ' ').title()}: {np.nanmean(values):.3f}")

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
