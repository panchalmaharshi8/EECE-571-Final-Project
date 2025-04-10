import json
import os
import numpy as np
import keyboard
import time
from environments.resection_env import TORSResectionEnv
from control.keyboard_control import KeyboardController

# ✅ Initialize environment & controller
env = TORSResectionEnv()
controller = KeyboardController()

# ✅ Load or initialize demo storage
demo_file = "training/human_demos.json"

if os.path.exists(demo_file):
    with open(demo_file, "r") as f:
        data = json.load(f)
    print(f"📂 Loaded existing demo file: {demo_file}")
    print(f"🔹 Previous observations: {len(data['observations'])}")
else:
    data = {"observations": [], "actions": []}
    print("📁 No existing demos found. Starting a new dataset.")

print("\n🔴 Recording Started - Use WASD keys to control. Press ESC to stop.")

# ✅ Keys to monitor for continued input
control_keys = ["w", "a", "s", "d", "r", "f", "q", "e", "z", "x"]

while True:
    obs, info = env.reset(), {}
    step_count = 0  
    episode_data = {"observations": [], "actions": []}
    last_action = np.zeros(7)
    has_action = False

    print("\n🕹️  New episode started. Execute the task.")

    while True:
        # ✅ Get user input for action
        action = controller.get_action()
        step_count += 1

        if not np.all(action == 0.0):
            last_action = action.copy()
            has_action = True
        else:
            # Only repeat last action if you're still holding a key
            if any(keyboard.is_pressed(k) for k in control_keys):
                action = last_action.copy()
            else:
                action = np.zeros(7)

        sim = getattr(env.env, "sim", None)  
        if not sim:
            raise RuntimeError("❌ ERROR: `sim` not found in `env.env`! Check environment setup.")

        try:
            # ✅ Extract MuJoCo state
            gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
            gripper_quat = sim.data.body_xquat[sim.model.body_name2id("gripper0_right_eef")]
            can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
            target_pos = sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")]

            joint_qpos = sim.data.qpos[:7]  # First 7 joints of the robot arm
            gripper_width = sim.data.qpos[7] if sim.data.qpos.shape[0] > 7 else 0.0
            raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0
            gripper_closed = 1.0 if abs(raw_force) > 0.04 else 0.0

        except KeyError as e:
            print(f"\n❌ ERROR: {e}. Available bodies: {sim.model.body_names}")
            raise RuntimeError("❌ ERROR: One or more required body names are missing in MuJoCo!")

        # ✅ Compute relative positions
        relative_can_pos = can_pos - gripper_pos  
        relative_target_pos = target_pos - can_pos  

        # ✅ Debug printouts
        print(f"\n🔍 Step {step_count}:")
        print(f"🔹 Relative to Can: {relative_can_pos}")
        print(f"🔹 Relative to Target: {relative_target_pos}")
        print(f"🔸 Gripper Closed: {gripper_closed}, Width: {gripper_width:.4f}")
        print(f"🧠 Joint Angles: {joint_qpos}")
        print(f"🎮 Action used: {action}")

        # ✅ Final 22D observation vector
        modified_obs = np.concatenate([
            relative_can_pos,         # (3,)
            relative_target_pos,      # (3,)
            [gripper_closed],         # (1,)
            gripper_pos,              # (3,)
            gripper_quat,             # (4,)
            [gripper_width],          # (1,)
            joint_qpos                # (7,)
        ])

        # ✅ Record only after first meaningful action
        if has_action:
            episode_data["observations"].append(modified_obs.tolist())
            episode_data["actions"].append(action.tolist())

        # ✅ Step environment
        obs, reward, done, info = env.step(action)
        env.render()

        if keyboard.is_pressed("esc"):  
            print("❌ ESC pressed! Stopping recording.")
            break  
        if done:
            print(f"✅ Episode completed after {step_count} steps. Resetting...")
            break  

        time.sleep(0.03)  # Smooth polling

    data["observations"].extend(episode_data["observations"])
    data["actions"].extend(episode_data["actions"])

    if keyboard.is_pressed("esc"):
        break

# ✅ Save updated dataset
with open(demo_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"\n✅ Human demonstrations saved to {demo_file}")
print(f"📈 Total samples now: {len(data['observations'])}")



# import json
# import os
# import numpy as np
# import keyboard
# from environments.resection_env import TORSResectionEnv
# from control.keyboard_control import KeyboardController

# # ✅ Initialize environment & controller
# env = TORSResectionEnv()
# controller = KeyboardController()

# # ✅ Load or initialize demo storage
# demo_file = "training/human_demos.json"

# if os.path.exists(demo_file):
#     with open(demo_file, "r") as f:
#         data = json.load(f)
#     print(f"📂 Loaded existing demo file: {demo_file}")
#     print(f"🔹 Previous observations: {len(data['observations'])}")
# else:
#     data = {"observations": [], "actions": []}
#     print("📁 No existing demos found. Starting a new dataset.")

# print("\n🔴 Recording Started - Use WASD keys to control. Press ESC to stop.")

# while True:
#     obs, info = env.reset(), {}
#     step_count = 0  
#     episode_data = {"observations": [], "actions": []}

#     print("\n🕹️  New episode started. Execute the task.")

#     while True:
#         # ✅ Get user input for action
#         action = controller.get_action()
#         step_count += 1

#         sim = getattr(env.env, "sim", None)  
#         if not sim:
#             raise RuntimeError("❌ ERROR: `sim` not found in `env.env`! Check environment setup.")

#         try:
#             # ✅ Extract MuJoCo state
#             gripper_pos = sim.data.body_xpos[sim.model.body_name2id("gripper0_right_eef")]
#             gripper_quat = sim.data.body_xquat[sim.model.body_name2id("gripper0_right_eef")]
#             can_pos = sim.data.body_xpos[sim.model.body_name2id("Can_main")]
#             target_pos = sim.data.body_xpos[sim.model.body_name2id("VisualCan_main")]

#             joint_qpos = sim.data.qpos[:7]  # First 7 joints of the robot arm
#             gripper_width = sim.data.qpos[7] if sim.data.qpos.shape[0] > 7 else 0.0
#             raw_force = float(sim.data.sensordata[0]) if sim.data.sensordata.shape[0] > 0 else 0.0
#             gripper_closed = 1.0 if abs(raw_force) > 0.04 else 0.0

#         except KeyError as e:
#             print(f"\n❌ ERROR: {e}. Available bodies: {sim.model.body_names}")
#             raise RuntimeError("❌ ERROR: One or more required body names are missing in MuJoCo!")

#         # ✅ Compute relative positions
#         relative_can_pos = can_pos - gripper_pos  
#         relative_target_pos = target_pos - can_pos  

#         # ✅ Debugging printouts
#         print(f"\n🔍 Step {step_count}:")
#         print(f"🔹 Relative to Can: {relative_can_pos}")
#         print(f"🔹 Relative to Target: {relative_target_pos}")
#         print(f"🔸 Gripper Closed: {gripper_closed}, Width: {gripper_width:.4f}")
#         print(f"🧠 Joint Angles: {joint_qpos}")

#         # ✅ Final 22D observation vector
#         modified_obs = np.concatenate([
#             relative_can_pos,         # (3,)
#             relative_target_pos,      # (3,)
#             [gripper_closed],         # (1,)
#             gripper_pos,              # (3,)
#             gripper_quat,             # (4,)
#             [gripper_width],          # (1,)
#             joint_qpos                # (7,)
#         ])

#         episode_data["observations"].append(modified_obs.tolist())
#         episode_data["actions"].append(action.tolist())

#         # ✅ Step environment
#         obs, reward, done, info = env.step(action)
#         env.render()

#         if keyboard.is_pressed("esc"):  
#             print("❌ ESC pressed! Stopping recording.")
#             break  
#         if done:
#             print(f"✅ Episode completed after {step_count} steps. Resetting...")
#             break  

#     data["observations"].extend(episode_data["observations"])
#     data["actions"].extend(episode_data["actions"])

#     if keyboard.is_pressed("esc"):
#         break

# # ✅ Save updated dataset
# with open(demo_file, "w") as f:
#     json.dump(data, f, indent=4)

# print(f"\n✅ Human demonstrations saved to {demo_file}")
# print(f"📈 Total samples now: {len(data['observations'])}")

