from stable_baselines3 import SAC, PPO, A2C, DQN
from custom_env_wrapper import CustomRewardWrapper
from utils import load_config
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os

ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs("results/custom", exist_ok=True)
    os.makedirs("logs/custom", exist_ok=True)

    for trial in range(1, cfg['num_trials'] + 1):
        max_steps = cfg.get('max_episode_steps', 999)
        env = gym.make(cfg['environment'], max_episode_steps=max_steps)
        algo = ALGOS[cfg['algorithm']]
        env = CustomRewardWrapper(env, cfg)
        model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/custom/{cfg['algorithm']}_trial{trial}/")
        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=f"logs/custom/{cfg['algorithm']}_trial{trial}/checkpoints/")
        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(f"results/custom/{cfg['algorithm']}_trial{trial}")

if __name__ == "__main__": train_custom()

