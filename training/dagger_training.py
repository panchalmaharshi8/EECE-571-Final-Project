import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.data_loader import load_human_demos
from environments.resection_env import TORSResectionEnv
from training.bc_policy import BCPolicy
from training.oracle import NearestNeighborOracle
import csv
import matplotlib.pyplot as plt

# === 🧠 Extract custom 22D observation from MuJoCo sim
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

# === ✅ Load human demonstrations
print("✅ Loading human demos...")
obs_data, action_data = load_human_demos("training/human_demos.json")
obs_data = np.array(obs_data, dtype=np.float32)
action_data = np.array(action_data, dtype=np.float32)

# === 🧼 Normalize observations
obs_mean = np.mean(obs_data, axis=0)
obs_std = np.std(obs_data, axis=0) + 1e-8
obs_data_norm = (obs_data - obs_mean) / obs_std

# === 🧠 Initialize models + oracles
obs_dim = obs_data.shape[1]
act_dim = action_data.shape[1]
policy = BCPolicy(obs_dim, act_dim)
oracle = BCPolicy(obs_dim, act_dim)
nn_oracle = NearestNeighborOracle(obs_data, action_data, obs_mean, obs_std)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

# === 📊 Store all MSE loss values
all_mse_losses = []  # Each entry: (dagger_iteration, epoch, mse_loss)

# === 🧠 Train oracle model
print("🧠 Training neural oracle on demos...")
oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=1e-4)
oracle_dataset = TensorDataset(torch.tensor(obs_data_norm), torch.tensor(action_data))
oracle_loader = DataLoader(oracle_dataset, batch_size=64, shuffle=True)

for epoch in range(20):
    total_loss = 0.0
    for batch_obs, batch_act in oracle_loader:
        oracle_optimizer.zero_grad()
        pred = oracle(batch_obs)
        loss = loss_fn(pred, batch_act)
        loss.backward()
        oracle_optimizer.step()
        total_loss += loss.item()
    print(f"📚 Oracle Epoch {epoch+1}: Loss = {total_loss/len(oracle_loader):.4f}")

# === 🧠 Main BC training loop (with logging)
def train_bc_model(obs, actions, epochs=100, dagger_iter=-1):
    dataset = TensorDataset(torch.tensor(obs), torch.tensor(actions))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_act in dataloader:
            optimizer.zero_grad()
            pred = policy(batch_obs)
            loss = loss_fn(pred, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        all_mse_losses.append((dagger_iter, epoch, avg_loss))
        print(f"🎯 Epoch {epoch+1} (Iter {dagger_iter}): Loss = {avg_loss:.4f}")

# === Phase 1: Train initial policy
print("\n🎯 Training initial policy on human demos...")
train_bc_model(obs_data_norm, action_data, dagger_iter=-1)

# === Phase 2: DAgger loop with hybrid oracle
env = TORSResectionEnv()
dagger_iterations = 5
dagger_rollouts_per_iter = 10

for iteration in range(dagger_iterations):
    print(f"\n🔁 DAgger Iteration {iteration + 1}")
    new_obs = []
    new_actions = []

    for rollout in range(dagger_rollouts_per_iter):
        env.reset()
        done = False

        while not done:
            sim = getattr(env.env, "sim", None)
            if sim is None:
                raise RuntimeError("❌ ERROR: `sim` not found in `env.env`!")

            custom_obs = extract_custom_obs(sim)
            obs_norm = (custom_obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(obs_norm).unsqueeze(0)

            with torch.no_grad():
                pred_action = policy(obs_tensor).squeeze().numpy()

            _, _, done, _ = env.step(pred_action)

            # 🔀 Hybrid oracle logic
            if iteration < 2:
                expert_action = nn_oracle(custom_obs)
            else:
                with torch.no_grad():
                    expert_action = oracle(obs_tensor).squeeze().numpy()

            new_obs.append(custom_obs)
            new_actions.append(expert_action)

    # === Aggregate new data and retrain
    new_obs = np.array(new_obs, dtype=np.float32)
    new_actions = np.array(new_actions, dtype=np.float32)
    new_obs_norm = (new_obs - obs_mean) / obs_std

    obs_data_norm = np.vstack([obs_data_norm, new_obs_norm])
    action_data = np.vstack([action_data, new_actions])

    print(f"📦 Aggregated {len(new_obs)} new samples. Total: {len(obs_data_norm)}")
    train_bc_model(obs_data_norm, action_data, dagger_iter=iteration)

# === 💾 Save final model + stats
torch.save(policy.state_dict(), "training/dagger_policy.pth")
np.savez("training/bc_norm_stats.npz", mean=obs_mean, std=obs_std)
print("\n✅ DAgger-trained policy saved!")

# === 💾 Save MSE loss log to CSV
csv_path = "training/mse_loss_log.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dagger_iteration", "epoch", "mse_loss"])
    for row in all_mse_losses:
        writer.writerow(row)
print(f"📈 MSE loss log saved to '{csv_path}'")

# === 📊 Plot loss curves
dagger_iters = [row[0] for row in all_mse_losses]
epochs = [row[1] for row in all_mse_losses]
losses = [row[2] for row in all_mse_losses]

plt.figure(figsize=(10, 6))
for iter_id in sorted(set(dagger_iters)):
    xs = [epoch for i, epoch in enumerate(epochs) if dagger_iters[i] == iter_id]
    ys = [loss for i, loss in enumerate(losses) if dagger_iters[i] == iter_id]
    label = f"DAgger Iter {int(iter_id)}" if iter_id >= 0 else "Initial"
    plt.plot(xs, ys, label=label)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Behavioral Cloning MSE Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training/mse_loss_plot.png")
plt.show()

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from utils.data_loader import load_human_demos
# from environments.resection_env import TORSResectionEnv
# from training.bc_policy import BCPolicy
# from training.oracle import NearestNeighborOracle

# # === 🧠 Extract custom 22D observation from MuJoCo sim
# def extract_custom_obs(sim):
#     gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
#     gripper_quat = sim.data.body_xquat[sim.model.body_name2id("gripper0_right_eef")]
#     can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
#     target_pos = sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")]
    
#     joint_qpos = sim.data.qpos[:7]
#     gripper_width = sim.data.qpos[7] if sim.data.qpos.shape[0] > 7 else 0.0
#     raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0
#     gripper_closed = 1.0 if abs(raw_force) > 0.04 else 0.0

#     relative_can_pos = can_pos - gripper_pos
#     relative_target_pos = target_pos - can_pos

#     return np.concatenate([
#         relative_can_pos,
#         relative_target_pos,
#         [gripper_closed],
#         gripper_pos,
#         gripper_quat,
#         [gripper_width],
#         joint_qpos
#     ]).astype(np.float32)

# # === ✅ Load human demonstrations
# print("✅ Loading human demos...")
# obs_data, action_data = load_human_demos("training/human_demos.json")
# obs_data = np.array(obs_data, dtype=np.float32)
# action_data = np.array(action_data, dtype=np.float32)

# # === 🧼 Normalize observations
# obs_mean = np.mean(obs_data, axis=0)
# obs_std = np.std(obs_data, axis=0) + 1e-8
# obs_data_norm = (obs_data - obs_mean) / obs_std

# # === 🧠 Initialize models + oracles
# obs_dim = obs_data.shape[1]
# act_dim = action_data.shape[1]
# policy = BCPolicy(obs_dim, act_dim)
# oracle = BCPolicy(obs_dim, act_dim)
# nn_oracle = NearestNeighborOracle(obs_data, action_data, obs_mean, obs_std)

# optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
# loss_fn = torch.nn.MSELoss()

# # === 🧠 Train oracle model
# print("🧠 Training neural oracle on demos...")
# oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=1e-4)
# oracle_dataset = TensorDataset(torch.tensor(obs_data_norm), torch.tensor(action_data))
# oracle_loader = DataLoader(oracle_dataset, batch_size=64, shuffle=True)

# for epoch in range(20):
#     total_loss = 0.0
#     for batch_obs, batch_act in oracle_loader:
#         oracle_optimizer.zero_grad()
#         pred = oracle(batch_obs)
#         loss = loss_fn(pred, batch_act)
#         loss.backward()
#         oracle_optimizer.step()
#         total_loss += loss.item()
#     print(f"📚 Oracle Epoch {epoch+1}: Loss = {total_loss/len(oracle_loader):.4f}")

# # === Main BC training loop
# def train_bc_model(obs, actions, epochs=100):
#     dataset = TensorDataset(torch.tensor(obs), torch.tensor(actions))
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for batch_obs, batch_act in dataloader:
#             optimizer.zero_grad()
#             pred = policy(batch_obs)
#             loss = loss_fn(pred, batch_act)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"🎯 Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

# # === Phase 1: Train initial policy
# print("\n🎯 Training initial policy on human demos...")
# train_bc_model(obs_data_norm, action_data)

# # === Phase 2: DAgger loop with hybrid oracle
# env = TORSResectionEnv()
# dagger_iterations = 5
# dagger_rollouts_per_iter = 10

# for iteration in range(dagger_iterations):
#     print(f"\n🔁 DAgger Iteration {iteration + 1}")
#     new_obs = []
#     new_actions = []

#     for rollout in range(dagger_rollouts_per_iter):
#         env.reset()
#         done = False

#         while not done:
#             sim = getattr(env.env, "sim", None)
#             if sim is None:
#                 raise RuntimeError("❌ ERROR: `sim` not found in `env.env`!")

#             custom_obs = extract_custom_obs(sim)
#             obs_norm = (custom_obs - obs_mean) / obs_std
#             obs_tensor = torch.tensor(obs_norm).unsqueeze(0)

#             with torch.no_grad():
#                 pred_action = policy(obs_tensor).squeeze().numpy()

#             _, _, done, _ = env.step(pred_action)

#             # 🔀 Hybrid oracle logic
#             if iteration < 2:
#                 expert_action = nn_oracle(custom_obs)
#             else:
#                 with torch.no_grad():
#                     expert_action = oracle(obs_tensor).squeeze().numpy()

#             new_obs.append(custom_obs)
#             new_actions.append(expert_action)

#     # === Aggregate new data and retrain
#     new_obs = np.array(new_obs, dtype=np.float32)
#     new_actions = np.array(new_actions, dtype=np.float32)
#     new_obs_norm = (new_obs - obs_mean) / obs_std

#     obs_data_norm = np.vstack([obs_data_norm, new_obs_norm])
#     action_data = np.vstack([action_data, new_actions])

#     print(f"📦 Aggregated {len(new_obs)} new samples. Total: {len(obs_data_norm)}")
#     train_bc_model(obs_data_norm, action_data)

# # === 💾 Save final model + stats
# torch.save(policy.state_dict(), "training/dagger_policy.pth")
# np.savez("training/bc_norm_stats.npz", mean=obs_mean, std=obs_std)
# print("\n✅ DAgger-trained policy saved!")
