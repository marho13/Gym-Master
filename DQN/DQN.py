import numpy as np
import random
import torch
import torch.hub
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple


class DQN(nn.Module):
    def __init__(self, img_height, img_width, outputSize, n_latent_var):
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=n_latent_var)
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var)
        self.out = nn.Linear(in_features=n_latent_var, out_features=outputSize)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
