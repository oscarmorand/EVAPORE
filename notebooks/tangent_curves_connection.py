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
import matplotlib.pyplot as plt
import json

# %%
path = "/home/morand/afs/EVAPORE/data/FIVES/centerlines/FIVES_051.json"
data = {}
with open(path, 'r') as f:
    data = json.load(f)
centerlines = data['train']["path_centerlines"]
print(len(centerlines))

# %%
'''
centerline = None
for c in centerlines:
    len_centerline = len(c)
    if len_centerline > 100:
        centerline = c
        break
'''
centerline = centerlines[157]
centerline = np.array(centerline)
centerline = centerline - np.min(centerline, axis=0)
print(centerline)

# %%
background = np.zeros((np.max(centerline, axis=0) + 10).astype(int))
background[centerline[:, 0]+5, centerline[:, 1]+5] = 1

plt.imshow(background, cmap='gray')
plt.show()


# %%
def estimate_tangent(points, n=10):
    pts = points[-n:]

    # center the points
    mean = pts.mean(axis=0)
    centered = pts - mean

    # PCA: direction of maximum variance
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]

    # ensure direction points "forward"
    if np.dot(direction, points[-1] - points[-2]) < 0:
        direction = -direction

    return direction / np.linalg.norm(direction)


# %%
def estimate_tangent(points, n=10, degree=2):
    pts = points[-n:]
    t = np.arange(n)

    tangent = np.zeros(pts.shape[1])
    for d in range(pts.shape[1]):
        coeffs = np.polyfit(t, pts[:, d], degree)
        tangent[d] = np.polyder(np.poly1d(coeffs))(t[-1])

    return tangent / np.linalg.norm(tangent)



# %%
plt.imshow(background, cmap='gray')
last_point = centerline[-1]
tangent = estimate_tangent(centerline, n=5)
plt.arrow(last_point[1]+5, last_point[0]+5, tangent[1]*20, tangent[0]*20, head_width=3, head_length=5, fc='red', ec='red')
plt.show()

# %%
inv_centerline = centerline[::-1]

plt.imshow(background, cmap='gray')
last_point = inv_centerline[-1]
tangent = estimate_tangent(inv_centerline, n=5)
plt.arrow(last_point[1]+5, last_point[0]+5, tangent[1]*20, tangent[0]*20, head_width=3, head_length=5, fc='red', ec='red')
plt.show()

# %%
import numpy as np

def hermite_curve(p0, p1, t0, t1, num_points=100):
    """
    Cubic Hermite curve connecting p0 to p1 with tangents t0 and t1
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    t0 = np.asarray(t0)
    t1 = np.asarray(t1)

    t = np.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 - t2

    curve = (
        h00[:, None] * p0 +
        h10[:, None] * t0 +
        h01[:, None] * p1 +
        h11[:, None] * t1
    )

    return curve


# %%
p0 = centerline[-1]
p1 = inv_centerline[-1]
t0 = estimate_tangent(centerline, n=5) * -1
t1 = estimate_tangent(inv_centerline, n=5) * 1
scale = np.linalg.norm(p1 - p0)
t0 *= scale
t1 *= scale
curve = hermite_curve(p0, p1, t0, t1, num_points=1000)


plt.imshow(background, cmap='gray')
plt.arrow(p0[1]+5, p0[0]+5, t0[1], t0[0], head_width=3, head_length=5, fc='red', ec='red')
plt.arrow(p1[1]+5, p1[0]+5, t1[1], t1[0], head_width=3, head_length=5, fc='red', ec='red')
plt.plot(curve[:, 1]+5, curve[:, 0]+5, 'blue')
plt.show()

# %%
import numpy as np

def rasterize_curve(curve, grid_spacing=1.0):
    """
    Converts a continuous curve into a list of grid coordinates
    that visually reconstruct the curve when drawn.
    
    Parameters
    ----------
    curve : (N, D) array
        Dense sampling of the curve (2D or 3D)
    grid_spacing : float
        Size of one grid cell

    Returns
    -------
    path : (M, D) array of integer grid coordinates
    """
    curve = np.asarray(curve)

    # map continuous coordinates to grid
    grid_coords = np.floor(curve / grid_spacing + 0.5).astype(int)

    # remove consecutive duplicates
    mask = np.any(np.diff(grid_coords, axis=0) != 0, axis=1)
    path = np.vstack([grid_coords[0], grid_coords[1:][mask]])

    return path



# %%
path = rasterize_curve(curve, grid_spacing=1.0)
print(path)

# %%
for p in path:
    background[p[0], p[1]] = 1

plt.imshow(background, cmap='gray')
plt.show()


# %% [markdown]
# # Geometric / curve fitting

# %%
def bresenham_line(x0, y0, x1, y1):
    """Return a list of integer pixel coordinates between (x0,y0) and (x1,y1) using Bresenham's algorithm."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


# %%
from scipy.interpolate import splprep, splev
import cv2
import matplotlib.pyplot as plt

path = centerline
y, x = centerline[:, 0], centerline[:, 1]

for s in [0, 1, 2, 5, 10, 20]:
    tck, u = splprep([x, y], s=s)  # s is smoothness parameter
    u_new = np.linspace(0, 1, len(x)*10)
    x_smooth, y_smooth = splev(u_new, tck)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y, '-', label='Original Path')
    plt.plot(x_smooth, y_smooth, '-', label='B-Spline Smoothed')
    plt.legend()
    plt.show()

    pixel_path = []
    for i in range(len(x_smooth)-1):
        line = bresenham_line(x_smooth[i], y_smooth[i], x_smooth[i+1], y_smooth[i+1])
        if pixel_path:
            # avoid duplicates
            line = line[1:]
        pixel_path.extend(line)

    pixel_path = np.array(pixel_path)

    print(pixel_path)

    plt.figure(figsize=(6,6))
    plt.plot(path[:,0], path[:,1], 'ro-', label='Original')
    plt.plot(pixel_path[:,1], pixel_path[:,0], 'g.-', label='Pixel-Connected Smooth Path')
    plt.legend()
    plt.axis('equal')
    plt.show()

    H, W = list((np.max(centerline, axis=0) + 10).astype(int))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(len(centerline)):
        img[centerline[i, 0]+5, centerline[i, 1]+5] = (255, 0, 0)  # original path in red
    for i in range(len(pixel_path)-1):
        cv2.line(img, (pixel_path[i, 0] + 5, pixel_path[i, 1] + 5), (pixel_path[i+1, 0] + 5, pixel_path[i+1, 1] + 5), color=(0, 255, 0), thickness=1)

    # Show rasterized image
    plt.imshow(img, cmap='gray')
    plt.title("Rasterized Smoothed Path")
    plt.show()

# %%
