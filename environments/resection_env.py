import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
import keyboard
import gym
from gym import spaces
from gym.wrappers import TimeLimit  


class TORSResectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = suite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True, 
            control_freq=20,
            render_camera="frontview",  # Enables camera control
            camera_names="agentview",
            camera_heights=720,
            camera_widths=1280,
        )
        self.env = GymWrapper(self.env)

        # ✅ **Define Observation and Action Spaces** (Required for SB3)
        obs_shape = self.env.observation_space.shape
        action_dim = self.env.action_space.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        self.prev_grip_forces = []
        self.release_delay = 5  # 🔥 Require force drop for 5 steps
        self.force_drop_threshold = 0.1  # 🔥 Detect sudden force decrease

        # Debugging Information
        print("✅ Environment successfully created!")
        print(f"🔹 Expected action dimension: {self.env.action_space.shape[0]}")
        print(f"🔹 Available Cameras: {self.env.env.sim.model.camera_id2name}")

        # 🔥 **NEW: Print available MuJoCo site names**
        print(f"✅ Available Sites in MuJoCo: {self.env.sim.model.site_names}")

    def reset(self):
        obs_tuple = self.env.reset()
        obs = obs_tuple[0]  # ✅ Extract only the observation

        print(f"🔍 DEBUG: Extracted observation type: {type(obs)}")
        print(f"🔍 DEBUG: Extracted observation shape: {obs.shape}")

        if isinstance(obs, dict):  
            obs = np.concatenate([np.asarray(v).flatten() for v in obs.values()], axis=0)
        else:
            obs = np.asarray(obs).flatten()  # ✅ Convert to a NumPy array

        print(f"✅ DEBUG: Processed observation shape (before reshape): {obs.shape}")

        obs = obs.astype(np.float32).reshape(1, -1)  # ✅ Force shape to (1, num_features)

        print(f"✅ DEBUG: Final reshaped observation shape: {obs.shape}")
        return obs  # ✅ SB3 expects shape (1, num_features)

    def step(self, action):
        # ✅ Step through the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # ✅ Overwrite the default reward with your custom reward
        custom_reward = self.compute_custom_reward(obs)
        reward = custom_reward  # ✅ THIS is what gets passed back to PPO

        # ✅ Extract gripper and can positions
        try:
            gripper_pos = np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("robot0_right_hand")])  
            can_pos = np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("Can_main")])  
            target_pos = np.array(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("VisualCan_main")])
            print(f"✅ Target Position Found (Object State): {target_pos}")
        except Exception as e:
            print(f"❌ ERROR: Could not extract body positions: {e}")
            gripper_pos, can_pos, target_pos = np.zeros(3), np.zeros(3), np.zeros(3)  # Prevent crashes

        # ✅ Compute Distances
        distance_to_target = np.linalg.norm(can_pos - target_pos)  # Distance between can & target
        distance_to_can = np.linalg.norm(gripper_pos - can_pos)  # Distance between gripper & can

        # ✅ **Placement Detection**
        placement_threshold = 0.07  # 🔥 Tuned for more reliable success detection
        is_placed = distance_to_target < placement_threshold  

        # ✅ **End Episode When Placement is Achieved**
        if is_placed:
            print("🎯 **TASK SUCCESS: Object Placed Correctly!**")
            info["task_success"] = True
            done = True  # ✅ Force episode termination
        else:
            info["task_success"] = False

        # ✅ Debugging Output
        if info.get("step_count", 0) % 10 == 0:
            print(f"🔍 DEBUG: Step {info.get('step_count', 'UNKNOWN')}")
            print(f"   Gripper Position: {gripper_pos}")
            print(f"   Can Position: {can_pos}")
            print(f"   Target Position: {target_pos}")
            print(f"   🔥 Corrected Distance to Can: {distance_to_can:.4f}")
            print(f"   Distance to Target: {distance_to_target:.4f}")
            print(f"   is_placed: {is_placed}")
            print(f"   Task Success: {info['task_success']}")

        return obs, reward, done, info  # ✅ Ensure the correct number of return values

    def compute_custom_reward(self, obs):
        # ✅ Extract gripper and can positions
        gripper_pos = obs[:3]  # XYZ position of the gripper/tool
        can_pos = obs[54:57]  # XYZ position of the can
        gripper_force = obs[50]  # Measure gripping force

        # ✅ Extract target position manually from MuJoCo
        try:
            target_pos = np.array(self.env.sim.data.get_site_xpos("Can_default_site"))  # ✅ Correct site name
        except Exception as e:
            print(f"❌ ERROR: Could not extract target position: {e}")
            target_pos = np.array([0, 0, 0])  # Prevent crashes

        # ✅ 1️⃣ Distance Rewards
        distance_to_can = np.linalg.norm(gripper_pos - can_pos)  # Distance to can
        distance_to_target = np.linalg.norm(can_pos - target_pos)  # Distance to target
        approach_reward = max(1.5 - distance_to_can, 0)  # Reward for approaching can

        # ✅ 2️⃣ Grasping Reward
        grip_threshold = 0.04
        grasp_reward = 1.0 if abs(gripper_force) > grip_threshold else -0.5

        # ✅ 3️⃣ Lifting Reward
        height_diff = gripper_pos[2] - can_pos[2]
        lift_reward = 2.0 if height_diff > 0.02 else -0.5

        # ✅ 4️⃣ Placement Reward
        placement_threshold = 0.5
        placement_reward = 3.0 if distance_to_target < placement_threshold else -1.0

        # ✅ 5️⃣ Dropping Penalty (Avoid Misplacement)
        drop_penalty = -2.0 if height_diff < -0.02 else 0

        # ✅ 6️⃣ Success Reward (NEW: Boost reward when task is fully completed)
        success_reward = 5.0 if (distance_to_target < placement_threshold and abs(gripper_force) < grip_threshold) else 0.0

        # ✅ Total Reward Calculation
        total_reward = (
            approach_reward +
            grasp_reward +
            lift_reward +
            placement_reward +
            drop_penalty +
            success_reward
        )

        # ✅ Debugging Info
        print(f"🔍 DEBUG: Distance to Can: {distance_to_can:.3f}, Distance to Target: {distance_to_target:.3f}")
        print(f"   Approach: {approach_reward:.2f}, Grip: {grasp_reward:.2f}, Lift: {lift_reward:.2f}")
        print(f"   Placement: {placement_reward:.2f}, Drop Penalty: {drop_penalty:.2f}, Success: {success_reward:.2f}")
        print(f"   🔥 Total Reward: {total_reward:.3f}")

        return total_reward

    def compute_custom_reward(self, obs):
        sim = getattr(self.env, "sim", None)  
        if not sim:
            raise RuntimeError("❌ ERROR: `sim` not found in `self.env`! Fix the environment setup.")

        try:
            # ✅ Correctly extract object positions from MuJoCo
            gripper_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")])
            can_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("Can_main")])
            target_pos = np.array(sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")])  # ✅ Fixed!

        except KeyError as e:
            print(f"\n❌ ERROR: {e}. Available bodies: {sim.model.body_names}")
            raise RuntimeError("❌ ERROR: One or more required body names are missing in MuJoCo!")

        # ✅ Compute correct distances
        distance_to_can = np.linalg.norm(can_pos - gripper_pos)  
        distance_to_target = np.linalg.norm(can_pos - target_pos)  

        # ✅ Extract gripper force safely
        gripper_force = obs[50] if len(obs) > 50 else 0.0  

        # ✅ Reward Components
        approach_reward = max(1.5 - distance_to_can, 0)  # ✅ Encourages approaching the can
        grasp_reward = 1.0 if abs(gripper_force) > 0.04 else -0.5  # ✅ Encourages gripping
        lift_reward = 2.0 if (gripper_pos[2] - can_pos[2]) > 0.02 else -0.5  # ✅ Encourages lifting

        placement_threshold = 0.15  # ✅ Tightened threshold for precision
        placement_reward = 3.0 if distance_to_target < placement_threshold else -1.0  

        drop_penalty = -2.0 if (gripper_pos[2] - can_pos[2]) < -0.02 else 0  

        success_reward = 5.0 if (distance_to_target < placement_threshold and abs(gripper_force) < 0.04) else 0.0  

        # ✅ Total Reward Calculation
        total_reward = (
            approach_reward +
            grasp_reward +
            lift_reward +
            placement_reward +
            drop_penalty +
            success_reward
        )

        return total_reward


    def render(self):
        self.env.render()

    def close(self):
        self.env.close()  # ✅ Added a close function (SB3 needs this)



