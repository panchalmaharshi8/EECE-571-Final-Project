import torch
import numpy as np
from environments.custom_resection_env import BCCompatibleTORSResectionEnv
from training.bc_policy import BCPolicy

# ✅ Load normalization stats
norm_stats = np.load("training/bc_norm_stats.npz")
obs_mean = norm_stats["mean"]
obs_std = norm_stats["std"]

# ✅ Load environment
env = BCCompatibleTORSResectionEnv()
obs = env.reset()
print(f"🧠 Raw observation shape: {obs.shape}")

# ✅ Normalize initial obs
norm_obs = (obs - obs_mean) / obs_std

# ✅ Get dimensions and load policy
obs_dim = norm_obs.shape[0]
act_dim = env.action_space.shape[0]
policy = BCPolicy(obs_dim, act_dim)
policy.load_state_dict(torch.load("training/bc_policy.pth", map_location="cpu"))
policy.eval()

# ✅ Inference loop
obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
done = False
step = 0

while not done:
    with torch.no_grad():
        action = policy(obs_tensor).squeeze(0).numpy()

    # 🛠 Optional control tweaks
    sim = env.env.sim
    gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
    can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
    distance_to_can = np.linalg.norm(gripper_pos - can_pos)

    obs, reward, done, _ = env.step(action)
    norm_obs = (obs - obs_mean) / obs_std  # ✅ normalize next obs
    obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)

    env.render()
    print(f"Step {step}: Reward = {reward:.4f}, Done = {done}, Gripper = {action[-1]:.2f}")
    step += 1

env.close()
print("✅ Test Completed!")

# import torch
# import numpy as np
# from environments.custom_resection_env import BCCompatibleTORSResectionEnv
# from training.bc_policy import BCPolicy  # Make sure this matches your file path

# # ✅ Load environment
# env = BCCompatibleTORSResectionEnv()
# obs = env.reset()
# print("🧠 DEBUG — Observation shape:", obs.shape)

# # ✅ Set dimensions
# obs_dim = obs.shape[0]
# act_dim = env.action_space.shape[0]

# # ✅ Load trained policy
# policy = BCPolicy(obs_dim, act_dim)
# policy.load_state_dict(torch.load("training/bc_policy.pth", map_location="cpu"))
# policy.eval()

# # ✅ Inference loop
# obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
# done = False
# step = 0

# while not done:
#     with torch.no_grad():
#         action = policy(obs_tensor).squeeze(0).numpy()

#     # 🛠 Optional control tweaks
#     sim = env.env.sim
#     gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
#     can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
#     distance_to_can = np.linalg.norm(gripper_pos - can_pos)

#     # if distance_to_can < 0.045 and action[-1] >= 0.0:
#     #     print(f"👋 Forcing grip at step {step} — distance to can: {distance_to_can:.3f}")
#     #     action[-1] = -1.0
#     # elif 0.045 <= distance_to_can < 0.07 and action[-1] == 0.0 and step > 10:
#     #     print("⬇️ Nudging down to reach can...")
#     #     action[2] -= 0.02
#     action[:3] *= 0.05  # scale x/y/z translation delta
#     action[3:6] *= 0.1  # scale rotations if too aggressive
#     # action[6] is gripper — maybe leave as-is or clip?
#     obs, reward, done, _ = env.step(action)
#     obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

#     env.render()
#     print(f"Action @ Step {step}: {action}")
#     # print(f"Step {step}: Reward = {reward:.4f}, Done = {done}, Gripper = {action[-1]:.2f}")
#     step += 1

# env.close()
# print("✅ Test Completed!")

