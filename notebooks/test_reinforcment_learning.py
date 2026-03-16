# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn

# %%
from enum import Enum

class ConnectivityType(Enum):
    FOUR_CONNECTED = 4
    EIGHT_CONNECTED = 8

class PathfindingPolicy(nn.Module):
    def __init__(self, n_features, hidden_dim=256, connectivity=ConnectivityType.EIGHT_CONNECTED):
        super().__init__()

        # --- Encoder spatial ---
        self.conv = nn.Sequential(
            nn.Conv2d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # fixe la taille
        )

        # --- Fully connected ---
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 4, hidden_dim),
            nn.ReLU()
        )

        # --- Heads ---
        self.policy_head = nn.Linear(hidden_dim, connectivity.value)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, start, goal):
        """
        x      : (B, n_features, H, W)
        start  : (B, 2)
        goal   : (B, 2)
        """

        batch_size = x.size(0)

        # CNN
        feat = self.conv(x)
        feat = feat.view(batch_size, -1)

        # Concat positions
        pos = torch.cat([start, goal], dim=1)

        z = torch.cat([feat, pos], dim=1)
        z = self.fc(z)

        policy = self.policy_head(z)
        value = self.value_head(z)

        return policy, value


# %%
def path_cost(pred_pos, gt_path, goal, w_dist=1.0, w_progress=1.0):
    """
    pred_pos   : (T, 2)
    gt_path    : (T, 2)
    goal       : (2,)
    """

    cost = 0.0
    T = pred_pos.size(0)

    for t in range(T):
        p = pred_pos[t]

        # --- Distance to GT path ---
        dists = torch.norm(gt_path - p, dim=1)
        c_dist = torch.min(dists)

        # --- Progress ---
        if t > 0:
            prev = pred_pos[t - 1]
            prog = torch.norm(prev - goal) - torch.norm(p - goal)
        else:
            prog = 0.0

        cost += (
            w_dist * c_dist -
            w_progress * prog
        )

    return cost / T  # average cost per step


# %%
def construct_path(model, feature_maps, start, goal, max_steps=100):
    """
    Génère un chemin entre start et goal en utilisant le modèle.
    
    Args:
        model        : PathfindingPolicy PyTorch
        feature_maps : (C, H, W) tensor
        start        : (2,) tensor ou array
        goal         : (2,) tensor ou array
        max_steps    : nombre maximal d'étapes pour éviter boucle infinie
        
    Returns:
        path : list de positions (x, y)
    """

    device = next(model.parameters()).device
    model.eval()

    path = []
    pos = torch.tensor(start, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    feature_maps_t = feature_maps.unsqueeze(0).to(device)  # add batch dim

    for _ in range(max_steps):
        path.append(pos.squeeze(0).cpu().numpy())

        # Arrivé ?
        if torch.norm(pos - goal_t) < 1.0:  # tolérance 1 pixel
            break

        # Forward
        with torch.no_grad():
            policy_logits, _ = model(feature_maps_t, pos, goal_t)
            probs = torch.softmax(policy_logits, dim=-1)
            action = torch.argmax(probs, dim=-1)  # greedy

        # Déplacement discret
        dx, dy = 0, 0
        a = action.item()
        if a == 0:   dx, dy = -1, 0  # up
        elif a == 1: dx, dy = 1, 0   # down
        elif a == 2: dx, dy = 0, -1  # left
        elif a == 3: dx, dy = 0, 1   # right

        new_x = pos[0, 0] + dx
        new_y = pos[0, 1] + dy

        # Clamp pour rester dans la grille
        H, W = feature_maps.shape[1:]
        new_x = max(0, min(H - 1, new_x))
        new_y = max(0, min(W - 1, new_y))

        pos = torch.tensor([[new_x, new_y]], dtype=torch.float32, device=device)

    return path



# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PathfindingPolicy(n_features=16, hidden_dim=128, connectivity=ConnectivityType.EIGHT_CONNECTED).to(device)

# %%
import os

feature_maps_folder = "/home/morand/afs/EVAPORE/data/FIVES/feature_maps/"
feature_maps_names = os.listdir(feature_maps_folder)
feature_maps_paths = [os.path.join(feature_maps_folder, name) for name in feature_maps_names]

print(feature_maps_paths)

feature_map_test_path = feature_maps_paths[0]

feature_map_test = torch.load(feature_map_test_path)
print("Feature map shape:", feature_map_test.shape)

# %%
