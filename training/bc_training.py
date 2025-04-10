import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from environments.resection_env import TORSResectionEnv
from gym import spaces
from training.bc_policy import BCPolicy
from utils.data_loader import load_human_demos

# ✅ Load human demonstrations
obs_data, action_data = load_human_demos("training/human_demos.json")

# ✅ Convert to NumPy arrays
obs_data = np.array(obs_data, dtype=np.float32)
action_data = np.array(action_data, dtype=np.float32)

# ✅ Compute normalization stats
obs_mean = np.mean(obs_data, axis=0)
obs_std = np.std(obs_data, axis=0) + 1e-8  # Add epsilon to avoid division by zero

# ✅ Normalize observations
normalized_obs = (obs_data - obs_mean) / obs_std

# ✅ Save stats for inference
np.savez("training/bc_norm_stats.npz", mean=obs_mean, std=obs_std)

# ✅ Format tensors
obs_tensor = torch.tensor(normalized_obs, dtype=torch.float32)
action_tensor = torch.tensor(action_data, dtype=torch.float32)

print("🧠 Normalized obs shape:", obs_tensor.shape)
print("🎯 Action tensor shape:", action_tensor.shape)

# ✅ Create DataLoader
dataset = TensorDataset(obs_tensor, action_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# ✅ Dimensions
obs_dim = obs_tensor.shape[1]
act_dim = action_tensor.shape[1]

# ✅ Define observation/action spaces (for compatibility)
obs_space = spaces.Box(low=-1000, high=1000, shape=(obs_dim,), dtype=np.float32)
act_space = spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32)

# ✅ Initialize your policy
policy = BCPolicy(obs_dim, act_dim)

# ✅ Training setup
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
num_epochs = 1000

# ✅ Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_obs, batch_action in dataloader:
        optimizer.zero_grad()
        predicted_action = policy(batch_obs)
        loss = loss_fn(predicted_action, batch_action)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# ✅ Save trained model
torch.save(policy.state_dict(), "training/bc_policy.pth")
print("✅ BC Policy and normalization stats saved!")

# import torch
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from environments.resection_env import TORSResectionEnv
# from gym import spaces
# from training.bc_policy import BCPolicy  # ✅ Make sure this path is correct!
# from utils.data_loader import load_human_demos

# # ✅ Load human demonstrations
# obs_data, action_data = load_human_demos("training/human_demos.json")

# # ✅ Auto-detect dimensions
# obs_dim = np.array(obs_data).shape[1]
# action_dim = np.array(action_data).shape[1]

# # ✅ Format properly
# obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
# action_tensor = torch.tensor(action_data, dtype=torch.float32)
# print("🧠 Observation tensor shape:", obs_tensor.shape)
# print("🧠 Action tensor shape:", action_tensor.shape)

# # ✅ Dataset and DataLoader
# dataset = TensorDataset(obs_tensor, action_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# # ✅ Define observation/action spaces
# obs_space = spaces.Box(low=-1000, high=1000, shape=(obs_dim,), dtype=np.float32)
# act_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

# # ✅ Initialize your custom policy
# policy = BCPolicy(obs_dim, action_dim)

# # ✅ Training setup
# optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
# loss_fn = torch.nn.MSELoss()
# num_epochs = 1000

# # ✅ Sanity check
# for batch_obs, batch_action in dataloader:
#     print(f"✅ Batch shape: obs = {batch_obs.shape}, action = {batch_action.shape}")
#     break

# # ✅ Training loop
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     for batch_obs, batch_action in dataloader:
#         optimizer.zero_grad()
#         predicted_action = policy(batch_obs)
#         loss = loss_fn(predicted_action, batch_action)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# # ✅ Save model
# torch.save(policy.state_dict(), "training/bc_policy.pth")
# print("✅ BC Policy Saved!")
