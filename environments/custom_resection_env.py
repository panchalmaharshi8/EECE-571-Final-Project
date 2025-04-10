from gym import spaces
import numpy as np
from environments.resection_env import TORSResectionEnv


class BCCompatibleTORSResectionEnv(TORSResectionEnv):
    def __init__(self):
        super().__init__()

        # ✅ Gymnasium-safe Box limits: Use reasonable large finite bounds
        obs_low = np.full((22,), -1e3, dtype=np.float32)
        obs_high = np.full((22,), 1e3, dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = self.env.action_space

        self.min_bounds = np.array([0.3, -0.3, 0.7])
        self.max_bounds = np.array([0.9, 0.3, 1.2])
        self.edge_margin = 0.05

    def reset(self):
        raw_obs, _ = self.env.reset()

        # If it's already flattened (57D), we slice out robot0_proprio-state
        obs_vec = raw_obs[:22]  # First 22 dims = proprio features

        if obs_vec.shape != (22,):
            raise ValueError(f"Expected 22D obs, got {obs_vec.shape}")

        return obs_vec.astype(np.float32)

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        obs_vec = raw_obs[:22].astype(np.float32)

        # ✅ Edge penalty logic
        sim = self.env.sim
        gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
        dist_to_min = gripper_pos - self.min_bounds
        dist_to_max = self.max_bounds - gripper_pos
        edge_penalty = 0.0

        for axis in range(3):
            if dist_to_min[axis] < self.edge_margin:
                edge_penalty -= (self.edge_margin - dist_to_min[axis]) * 5.0
            if dist_to_max[axis] < self.edge_margin:
                edge_penalty -= (self.edge_margin - dist_to_max[axis]) * 5.0

        reward += edge_penalty
        info["edge_penalty"] = edge_penalty

        done = terminated or truncated
        return obs_vec, reward, done, info
    


# import numpy as np
# from gym import spaces
# from environments.resection_env import TORSResectionEnv


# class BCCompatibleTORSResectionEnv(TORSResectionEnv):
#     def __init__(self):
#         super().__init__()

#         # ✅ Observation = [rel_can (3), rel_target (3), grip_active (1)] → Total = 7D
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=self.env.action_space.low,
#             high=self.env.action_space.high,
#             shape=self.env.action_space.shape,
#             dtype=np.float32
#         )

#         # ✅ Define workspace constraints (you can tweak these)
#         self.min_bounds = np.array([0.3, -0.3, 0.7])  # Min XYZ for gripper
#         self.max_bounds = np.array([0.9, 0.3, 1.2])   # Max XYZ for gripper
#         self.edge_margin = 0.05  # Distance from edge to start penalizing

#     def reset(self):
#         _ = super().reset()  # Reset the underlying env

#         sim = self.env.sim
#         gripper_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")])
#         can_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("Can_main")])
#         target_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")])
#         raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0

#         # ✅ Convert raw force to binary signal (1 = gripping, 0 = not gripping)
#         grip_active = 1.0 if abs(raw_force) > 0.04 else 0.0

#         relative_can_pos = can_pos - gripper_pos
#         relative_target_pos = target_pos - can_pos

#         obs = np.concatenate([
#             relative_can_pos,
#             relative_target_pos,
#             [grip_active]
#         ]).astype(np.float32)
#         return obs

#     def step(self, action):
#         _, reward, done, info = super().step(action)

#         sim = self.env.sim
#         gripper_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")])
#         can_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("Can_main")])
#         target_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")])
#         raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0
#         grip_active = 1.0 if abs(raw_force) > 0.04 else 0.0

#         relative_can_pos = can_pos - gripper_pos
#         relative_target_pos = target_pos - can_pos

#         obs = np.concatenate([
#             relative_can_pos,
#             relative_target_pos,
#             [grip_active]
#         ]).astype(np.float32)

#         # ✅ Soft edge penalty: penalize proximity to workspace boundaries
#         dist_to_min = gripper_pos - self.min_bounds
#         dist_to_max = self.max_bounds - gripper_pos

#         edge_penalty = 0.0
#         for axis in range(3):
#             if dist_to_min[axis] < self.edge_margin:
#                 edge_penalty -= (self.edge_margin - dist_to_min[axis]) * 5.0  # Tune as needed
#             if dist_to_max[axis] < self.edge_margin:
#                 edge_penalty -= (self.edge_margin - dist_to_max[axis]) * 5.0

#         reward += edge_penalty
#         info["edge_penalty"] = edge_penalty

#         return obs, reward, done, info
