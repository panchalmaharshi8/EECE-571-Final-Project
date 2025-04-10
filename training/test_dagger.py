import torch
import numpy as np
from training.bc_policy import BCPolicy
from environments.resection_env import TORSResectionEnv

# === 🔧 SAME OBS EXTRACTION AS TRAINING
def extract_custom_obs(sim):
    gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
    gripper_quat = sim.data.body_xquat[sim.model.body_name2id("gripper0_right_eef")]
    can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
    target_pos = sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")]
    
    joint_qpos = sim.data.qpos[:7]
    gripper_width = sim.data.qpos[7] if sim.data.qpos.shape[0] > 7 else 0.0
    raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0
    gripper_closed = 1.0 if abs(raw_force) > 0.04 else 0.0

    relative_can_pos = can_pos - gripper_pos
    relative_target_pos = target_pos - can_pos

    return np.concatenate([
        relative_can_pos,
        relative_target_pos,
        [gripper_closed],
        gripper_pos,
        gripper_quat,
        [gripper_width],
        joint_qpos
    ]).astype(np.float32)

# === 🧠 Load normalization stats
stats = np.load("training/bc_norm_stats.npz")
obs_mean = stats["mean"]
obs_std = stats["std"]

# === 🧠 Load trained policy
obs_dim = obs_mean.shape[0]
act_dim = 7  # You can also load this dynamically
policy = BCPolicy(obs_dim, act_dim)
policy.load_state_dict(torch.load("training/dagger_policy.pth"))
policy.eval()

# === 🎮 Rollout in environment
env = TORSResectionEnv()
num_episodes = 5
render = True

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        sim = getattr(env.env, "sim", None)
        if sim is None:
            raise RuntimeError("❌ Can't access sim in env!")

        custom_obs = extract_custom_obs(sim)
        norm_obs = (custom_obs - obs_mean) / obs_std
        obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy(obs_tensor).squeeze().numpy()

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        if render:
            env.render()

    print(f"✅ Episode {episode + 1} finished after {step_count} steps | Total reward: {total_reward:.2f}")

env.close()
