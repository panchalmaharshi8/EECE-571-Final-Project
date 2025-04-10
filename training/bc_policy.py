import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, act_dim),
            nn.Tanh() #clamping output -1,+1
        )

    def forward(self, x):
        return self.policy_net(x)


# class BCPolicy(nn.Module):
#     def __init__(self, obs_dim: int, act_dim: int):
#         super().__init__()
#         self.policy_net = nn.Sequential(
#             nn.Linear(obs_dim, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, act_dim),
#             nn.Tanh()  # Output range [-1, 1]
#         )

#     def forward(self, x):
#         return self.policy_net(x)
