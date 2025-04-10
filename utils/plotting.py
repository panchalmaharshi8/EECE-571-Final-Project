import numpy as np
import matplotlib.pyplot as plt

# **Load training logs**
ppo_rewards = np.load("training/ppo_rewards.npy")
bc_rewards = np.load("training/bc_rewards.npy")

# **Plot PPO vs. BC Convergence**
plt.figure(figsize=(8, 5))
plt.plot(ppo_rewards, label="PPO (Reinforcement Learning)", linestyle="-", marker="o")
plt.plot(bc_rewards, label="BC (Imitation Learning)", linestyle="--", marker="s")
plt.xlabel("Training Steps / Epochs")
plt.ylabel("Average Episode Reward")
plt.title("Comparison of PPO vs. BC Training")
plt.legend()
plt.grid(True)
plt.show()
