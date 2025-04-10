import keyboard
import numpy as np

class KeyboardController:
    def __init__(self):
        self.action_dim = 7  # Assuming 7-DOF Panda robot

        # **Block MuJoCo Debugging Keys**
        keyboard.block_key("d")  # Prevents toggling debug visualization
        keyboard.block_key("t")  # Prevents toggling transparency
        keyboard.block_key("q")
        keyboard.block_key("w")
        keyboard.block_key("a") 
        keyboard.block_key("r")  # Prevents simulation reset
        keyboard.block_key("y")  # Prevents toggling wireframe mode
        keyboard.block_key("m")  # Prevents contact forces visualization
        keyboard.block_key("n")  # Prevents surface normals from showing
        keyboard.block_key("v")  # Prevents velocity arrows from showing
        keyboard.block_key("x")  # Prevents velocity arrows from showing


    def get_action(self):
        action = np.zeros(self.action_dim)

        # **Movement Mapping**
        if keyboard.is_pressed("d"):  # Right
            action[1] = 0.5
        if keyboard.is_pressed("a"):  # Left
            action[1] = -0.5
        if keyboard.is_pressed("s"):  # Backward
            action[0] = -0.5
        if keyboard.is_pressed("w"):  # Forward
            action[0] = 0.5
        if keyboard.is_pressed("r"):  # Up
            action[2] = 0.5
        if keyboard.is_pressed("f"):  # Down
            action[2] = -0.5
        if keyboard.is_pressed("q"):  # Rotate left
            action[5] = -0.5
        if keyboard.is_pressed("e"):  # Rotate right
            action[5] = 0.5
        
        # **New Gripper Controls**
        if keyboard.is_pressed("z"):  # Close gripper (grab object)
            action[6] = -1.0  # Close grip
        if keyboard.is_pressed("x"):  # Open gripper (release object)
            action[6] = 1.0  # Open grip

        return action
