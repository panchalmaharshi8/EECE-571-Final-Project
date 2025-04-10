import json
import numpy as np

def load_human_demos(filepath):
    """Loads human demonstrations from a JSON file with separate 'observations' and 'actions' keys."""
    
    # ✅ Load JSON properly
    with open(filepath, "r") as file:
        try:
            data = json.load(file)  # Use json.load() to parse properly
        except json.JSONDecodeError as e:
            print(f"❌ JSON ERROR: {e}")
            return None, None  # Prevent crash

    # ✅ Check for correct keys
    if "observations" not in data or "actions" not in data:
        print("❌ ERROR: JSON should contain 'observations' and 'actions' keys!")
        return None, None

    # ✅ Convert to NumPy arrays
    obs_data = np.array(data["observations"], dtype=np.float32)
    action_data = np.array(data["actions"], dtype=np.float32)

    print(f"✅ Loaded {obs_data.shape[0]} samples!")
    return obs_data, action_data
