import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from environments.resection_env import TORSResectionEnv


# === 🧾 Optional: Step-wise reward logging wrapper
from gym import Wrapper

class RewardLoggingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_rewards = []
        self.total_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1
        self.step_rewards.append((self.total_steps, reward))
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def save_step_rewards(self, path="training/ppo_step_rewards.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "reward"])
            for t, r in self.step_rewards:
                writer.writerow([t, r])
        print(f"✅ Step-wise rewards saved to {path}")


# === 🧠 MSE and Reward logging callback
class MSEAndRewardLoggingCallback(BaseCallback):
    def __init__(self, save_freq=10000, csv_path="training/ppo_logs.csv", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.csv_path = csv_path
        self.mse_logs = []
        self.reward_logs = []

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mse_loss", "episode_reward"])

    def _on_step(self) -> bool:
        # === MSE Logging ===
        if hasattr(self.model, "rollout_buffer"):
            values = self.model.rollout_buffer.values
            returns = self.model.rollout_buffer.returns
            if values.shape == returns.shape:
                mse = ((values - returns) ** 2).mean().item()
                self.mse_logs.append((self.num_timesteps, mse))

        # === Reward Logging ===
        if "infos" in self.locals:
            infos = self.locals["infos"]
            for info in infos:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.reward_logs.append((self.num_timesteps, ep_reward))

        # === Save logs every save_freq
        if self.num_timesteps % self.save_freq < self.model.n_steps:
            self._flush_logs()

        return True

    def _flush_logs(self):
        if not self.mse_logs and not self.reward_logs:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            max_len = max(len(self.mse_logs), len(self.reward_logs))
            for i in range(max_len):
                timestep = self.mse_logs[i][0] if i < len(self.mse_logs) else ""
                mse = self.mse_logs[i][1] if i < len(self.mse_logs) else ""
                reward = self.reward_logs[i][1] if i < len(self.reward_logs) else ""
                writer.writerow([timestep, mse, reward])
        self.mse_logs.clear()
        self.reward_logs.clear()

    def save_remaining(self):
        self._flush_logs()
        print(f"✅ Final logs saved to {self.csv_path}")

    def plot(self, path="training/ppo_mse_and_rewards.png"):
        import pandas as pd
        try:
            df = pd.read_csv(self.csv_path)
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("MSE Loss", color="tab:red")
            ax1.plot(df["timesteps"], df["mse_loss"], label="MSE Loss", color="tab:red")
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Episode Reward", color="tab:blue")
            ax2.plot(df["timesteps"], df["episode_reward"], label="Episode Reward", color="tab:blue", alpha=0.6)
            ax2.tick_params(axis="y", labelcolor="tab:blue")

            plt.title("PPO Training: MSE vs. Reward")
            plt.tight_layout()
            plt.savefig(path)
            plt.show()
            print(f"📊 Plot saved to {path}")
        except Exception as e:
            print(f"⚠️ Could not plot MSE vs. reward: {e}")


# === 🧪 Custom MLP architecture
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    activation_fn=torch.nn.ReLU
)


# === 🌱 Initialize environment with wrappers
def make_env():
    raw_env = RewardLoggingWrapper(TORSResectionEnv())  # optional for per-step rewards
    return raw_env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecMonitor(env)


# === 🤖 Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cpu",
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=0.5,
)

# === 🧠 Train PPO with callback
total_timesteps = 200_000
mse_callback = MSEAndRewardLoggingCallback(save_freq=10000)
model.learn(total_timesteps=total_timesteps, callback=mse_callback)

# === 💾 Save model
model.save("training/ppo_resection")
print("✅ PPO model saved!")

# === 💾 Save logs + plot
mse_callback.save_remaining()
mse_callback.plot("training/ppo_mse_and_rewards.png")

# === 💾 Save step-wise rewards if using RewardLoggingWrapper
raw_env = env.venv.envs[0].env
if isinstance(raw_env, RewardLoggingWrapper):
    raw_env.save_step_rewards()

# === 👀 Test trained PPO model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
    if np.any(dones):
        obs = env.reset()
