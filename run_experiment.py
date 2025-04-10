from environments.resection_env import TORSResectionEnv
from control.keyboard_control import KeyboardController
import numpy as np
import keyboard
import time  # ✅ Prevent flickering

def main():
    env = TORSResectionEnv()
    controller = KeyboardController()
    
    obs = env.reset()

    # ✅ Resize MuJoCo Window
    if hasattr(env.env, "viewer") and env.env.viewer is not None:
        env.env.viewer._window_size = (1920, 1080)  # Make MuJoCo window larger

    frame_skip = 3  # ✅ Reduce rendering frequency
    step_count = 0

    done = False
    while not done:
        action = controller.get_action()
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        if step_count % frame_skip == 0:  # ✅ Only render every few steps
            if hasattr(env.env, "viewer") and env.env.viewer is not None:
                env.env.viewer._n_frames = 0  # ✅ Force frame buffer reset

            env.env.sim.forward()  # ✅ Ensure physics is updated
            env.render()  # ✅ Standard render call

            time.sleep(0.01)  # ✅ Add a tiny delay to prevent flickering

        if terminated or truncated:
            print("Episode finished")
            done = True

    # ✅ Manually close MuJoCo viewer (Fixes `env.close()` error)
    if hasattr(env.env, "viewer") and env.env.viewer is not None:
        env.env.viewer.close()

if __name__ == "__main__":
    main()
