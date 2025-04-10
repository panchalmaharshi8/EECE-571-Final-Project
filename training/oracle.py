import numpy as np
from scipy.spatial import KDTree

class NearestNeighborOracle:
    def __init__(self, obs_data: np.ndarray, action_data: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray):
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_data = action_data
        normalized_obs = (obs_data - obs_mean) / (obs_std + 1e-8)
        self.tree = KDTree(normalized_obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        _, idx = self.tree.query(norm_obs)
        return self.action_data[idx]

