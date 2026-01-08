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
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# %%
outmap_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/outmaps"
outmap_files = os.listdir(outmap_dir)
outmap_files.sort()
outmaps = []
for file in outmap_files:
    outmap = torch.load(os.path.join(outmap_dir, file), map_location='cpu', weights_only=False)
    outmaps.append(outmap)

# %%
test = outmaps[0]
print(test.shape)


# %%
plt.figure(figsize=(30, 30))
plt.imshow(test[1], cmap='gray')
plt.title('Outmap Channel 1')
plt.colorbar()
plt.show()

# %%
zone = test[1, 400:600, 200:400]
plt.figure(figsize=(20, 20))
plt.imshow(zone, cmap='gray')
plt.title('Zoomed Zone')
plt.colorbar()
plt.show()

# %%
import plotly.graph_objects as go

h,w = zone.shape
Z = zone.astype(np.float32)

fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis', showscale=True)])

fig.update_layout(
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="height",
        aspectratio=dict(x=1, y=1, z=0.1)
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    title="Grayscale image as 3D terrain"
)

fig.show()

# %%
p0_x, p0_y = 94, 100
p1_x, p1_y = 114, 3

# %%
import heapq

def dijkstra_heightmap(heightmap, 
                       start, # (y, x) coordinates of the start point
                       goal): # (y, x) coordinates of the goal point
    H, W = heightmap.shape

    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, dtype=int)
    visited = np.zeros((H, W), dtype=bool)

    dist[start] = 0.0

    pq = []
    heapq.heappush(pq, (0.0, start))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        current_dist, (y, x) = heapq.heappop(pq)

        if visited[y, x]:
            continue

        visited[y, x] = True

        if (y, x) == goal:
            break

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            if visited[ny, nx]:
                continue

            dh = heightmap[ny, nx] - heightmap[y, x]
            cost = 1.0 + max(0.0, dh)

            new_dist = current_dist + cost

            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                prev[ny, nx] = [y, x]
                heapq.heappush(pq, (new_dist, (ny, nx)))

    return dist, prev


# %%
def reconstruct_path(prev, start, goal):
    path = []
    current = goal

    while current != start:
        path.append(current)
        py, px = prev[current]
        if py == -1:
            return None
        current = (py, px)

    path.append(start)
    path.reverse()
    return path


# %%
heightmap = zone
start = (p0_y, p0_x)
goal = (p1_y, p1_x)

dist, prev = dijkstra_heightmap(heightmap, start, goal)
path = reconstruct_path(prev, start, goal)

print("Path from start to goal:", path)
print("Total cost:", dist[goal])


# %%
def display_path(heightmap, path):
    path_set = set(path)
    H, W = heightmap.shape
    display_map = np.zeros((H, W, 3), dtype=np.uint8)
    heightmap = heightmap / heightmap.max() * 255.0
    display_map[..., 0] = heightmap.astype(np.uint8)
    display_map[..., 1] = heightmap.astype(np.uint8)
    display_map[..., 2] = heightmap.astype(np.uint8)

    for y in range(H):
        for x in range(W):
            if (y, x) in path_set:
                display_map[y, x] = [255, 0, 0]

    plt.imshow(display_map)
    plt.title('Path on Heightmap')
    plt.show()


# %%
print(heightmap.min(), heightmap.max(), heightmap.dtype )
heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * 255.0
heightmap = heightmap.astype(np.uint8)

display_path(heightmap, path)


# %%
def reconstruct_path_with_radius(prev, start, goal, start_radius=1, goal_radius=1):
    path = []
    current = goal

    while current != start:
        path.append(current)
        py, px = prev[current]
        if py == -1:
            return None
        current = (py, px)

    path.append(start)
    path.reverse()

    radius_path = []
    n = len(path)
    for i, (y, x) in enumerate(path):
        ratio = i / n
        radius = int(start_radius * (1 - ratio) + goal_radius * ratio)
        radius_path.append(radius)

    return path, radius_path


# %%
path, radius_path = reconstruct_path_with_radius(prev, start, goal, start_radius=5, goal_radius=1)


# %%
def display_path_with_radius(heightmap, path, radius_path):
    path_set = set(path)
    H, W = heightmap.shape
    display_map = np.zeros((H, W, 3), dtype=np.uint8)
    heightmap = heightmap / heightmap.max() * 255.0
    display_map[..., 0] = heightmap.astype(np.uint8)
    display_map[..., 1] = heightmap.astype(np.uint8)
    display_map[..., 2] = heightmap.astype(np.uint8)

    for i, (y, x) in enumerate(path):
        radius = radius_path[i]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if dy * dy + dx * dx <= radius * radius:
                        display_map[ny, nx] = [255, 0, 0]

    plt.imshow(display_map)
    plt.title('Path on Heightmap')
    plt.show()


# %%
display_path_with_radius(heightmap, path, radius_path)

# %%
