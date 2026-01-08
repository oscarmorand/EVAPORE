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
from collections import defaultdict

# -----------------------------
# Union Find for H0
# -----------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.birth = {}

    def make_set(self, x, birth_time):
        self.parent[x] = x
        self.birth[x] = birth_time

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return None

        # Older component survives
        if self.birth[rx] >= self.birth[ry]:
            self.parent[ry] = rx
            return ry, rx
        else:
            self.parent[rx] = ry
            return rx, ry


# %%
# -----------------------------
# Cubical persistence
# -----------------------------

def cubical_persistence(image):
    H, W = image.shape

    cells = []

    # 2 cells
    for i in range(H):
        for j in range(W):
            cells.append(("square", (i, j), image[i, j]))

    # 1 cells horizontal edges
    for i in range(H):
        for j in range(W - 1):
            val = max(image[i, j], image[i, j + 1])
            cells.append(("edge_h", (i, j), val))

    # 1 cells vertical edges
    for i in range(H - 1):
        for j in range(W):
            val = max(image[i, j], image[i + 1, j])
            cells.append(("edge_v", (i, j), val))

    # 0 cells vertices
    for i in range(H + 1):
        for j in range(W + 1):
            vals = []
            for di in [-1, 0]:
                for dj in [-1, 0]:
                    pi = i + di
                    pj = j + dj
                    if 0 <= pi < H and 0 <= pj < W:
                        vals.append(image[pi, pj])
            if vals:
                cells.append(("vertex", (i, j), max(vals)))

    # Sort filtration
    cells.sort(key=lambda x: (-x[2], {"vertex": 0, "edge_h": 1, "edge_v": 1, "square": 2}[x[0]]))
    print(cells)

    uf = UnionFind()
    H0_pairs = []
    H1_pairs = []

    active_edges = {}
    cycle_birth = {}

    # Process filtration
    for cell_type, idx, val in cells:

        if cell_type == "vertex":
            uf.make_set(idx, val)

        elif cell_type == "edge_h" or cell_type == "edge_v":
            i, j = idx
            if cell_type == "edge_v":
                v1 = (i, j)
                v2 = (i, j + 1)
            else:
                v1 = (i, j)
                v2 = (i + 1, j)

            r1 = uf.find(v1)
            r2 = uf.find(v2)

            if r1 != r2:
                dead, alive = uf.union(v1, v2)
                H0_pairs.append((uf.birth[dead], val))
            else:
                active_edges[(cell_type, idx)] = val
                cycle_birth[(cell_type, idx)] = val

        elif cell_type == "square":
            i, j = idx
            boundary = [
                ("edge_v", (i, j)),
                ("edge_v", (i + 1, j)),
                ("edge_h", (i, j)),
                ("edge_h", (i, j + 1))
            ]

            cycle_edges = [e for e in boundary if e in active_edges]

            if cycle_edges:
                youngest = max(cycle_edges, key=lambda e: cycle_birth[e])
                birth = cycle_birth[youngest]
                H1_pairs.append((birth, val))

                for e in cycle_edges:
                    active_edges.pop(e, None)
                    cycle_birth.pop(e, None)

    # Infinite bars
    for v in uf.parent:
        if uf.find(v) == v:
            H0_pairs.append((uf.birth[v], np.inf))

    for e in cycle_birth:
        H1_pairs.append((cycle_birth[e], np.inf))

    return H0_pairs, H1_pairs


# %%
image = np.array([
        [0, 1, 4],
        [1, 1, 5],
        [2, 3, 5],
    ])

H0, H1 = cubical_persistence(image)

print("H0 persistence pairs")
for b, d in H0:
    print(b, d)

print("\nH1 persistence pairs")
for b, d in H1:
    print(b, d)


# %%
class TreeCC:
    def __init__(self):
        self.parent = {}
        self.birth = {}
        self.roots = set()

    def add_vertex(self, x, birth_time):
        self.parent[x] = x
        self.birth[x] = birth_time
        self.roots.add(x)

    def get_parent(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.get_parent(self.parent[x])
        return self.parent[x]

    def add_edge(self, x, y, active_val):
        rx = self.get_parent(x)
        ry = self.get_parent(y)
        if rx == ry:
            return False

        birth_x, birth_y = self.birth[rx], self.birth[ry]
        
        # Older component survives
        if birth_x >= birth_y:
            self.parent[ry] = rx
            self.roots.remove(ry)
        else:
            self.parent[rx] = ry
            self.roots.remove(rx)

        if birth_x == active_val or birth_y == active_val:
            return False
        return True


# %%
import matplotlib.pyplot as plt

class CubicalComplex:
    def __init__(self, image):
        H, W = image.shape
        cells = []

        # 2 cells
        for i in range(H):
            for j in range(W):
                cells.append(("square", (i, j), image[i, j]))

        # 1 cells horizontal edges
        for i in range(H):
            for j in range(W + 1):
                vals = []
                for dj in [-1, 0]:
                    pj = j + dj
                    if 0 <= pj < W:
                        vals.append(image[i, pj])
                cells.append(("edge_h", (i, j), max(vals)))

        # 1 cells vertical edges
        for i in range(H + 1):
            for j in range(W):
                vals = []
                for di in [-1, 0]:
                    pi = i + di
                    if 0 <= pi < H:
                        vals.append(image[pi, j])
                cells.append(("edge_v", (i, j), max(vals)))
            
        # 0 cells vertices
        for i in range(H + 1):
            for j in range(W + 1):
                vals = []
                for di in [-1, 0]:
                    for dj in [-1, 0]:
                        pi = i + di
                        pj = j + dj
                        if 0 <= pi < H and 0 <= pj < W:
                            vals.append(image[pi, pj])
                if vals:
                    cells.append(("vertex", (i, j), max(vals)))

        cells.sort(key=lambda x: (-x[2], {"vertex": 0, "edge_h": 1, "edge_v": 1, "square": 2}[x[0]]))

        self.image = image
        self.H, self.W = H, W
        self.cells = cells

        return
        self.cc_tree = TreeCC()
        self.h0_births = []
        self.h0_deaths = []
        current_val = None
        for cell_type, idx, val in cells:
            if current_val is None:
                current_val = val
            elif val != current_val:
                # New filtration value
                self.h0_births.append(self.cc_tree.roots.copy())
                current_val = val
            if cell_type == "vertex":
                self.cc_tree.add_vertex(idx, val)
            elif cell_type == "edge_h" or cell_type == "edge_v":
                i, j = idx
                if cell_type == "edge_h":
                    v1 = (i, j)
                    v2 = (i + 1, j)
                else:
                    v1 = (i, j)
                v2 = (i, j + 1)
                has_merged = self.cc_tree.add_edge(v1, v2, val)
                if has_merged:
                    self.h0_deaths.append(val)

    def display(self, square_size=10, gap_size=3, line_color='red'):
        img_W = self.W * square_size + (self.W + 1) * gap_size
        img_H = self.H * square_size + (self.H + 1) * gap_size
        img = np.zeros((img_H, img_W, 3), dtype=np.uint8) + 255
        max_val = np.max(self.image)
        sg_size = (square_size + gap_size)

        for cell_type, idx, val in self.cells:
            intensity = int((val / max_val) * 255)
            gray_level = [intensity]*3
            i, j = idx
            if cell_type == "vertex":
                top = i * sg_size
                left = j * sg_size
                img[top:top+gap_size, left:left+gap_size] = gray_level
            elif cell_type == "edge_h":
                top = i * sg_size + gap_size
                left = j * sg_size
                img[top:top+square_size, left:left+gap_size] = gray_level
            elif cell_type == "edge_v":
                top = i * sg_size
                left = j * sg_size + gap_size
                img[top:top+gap_size, left:left+square_size] = gray_level
            elif cell_type == "square":
                top = i * sg_size + gap_size
                left = j * sg_size + gap_size
                img[top:top+square_size, left:left+square_size] = gray_level
        plt.imshow(
            img,
            origin='upper',
            extent=[0, img.shape[1], 0, img.shape[0]]
        )

        for i in range(self.H + 1):
            up_line = i * sg_size
            bot_line = up_line + gap_size
            plt.plot([0, img_W], [up_line, up_line], color=line_color)
            plt.plot([0, img_W], [bot_line, bot_line], color=line_color)
        for j in range(self.W + 1):
            left_line = j * sg_size
            right_line = left_line + gap_size
            plt.plot([left_line, left_line], [0, img_H], color=line_color)
            plt.plot([right_line, right_line], [0, img_H], color=line_color)

        plt.axis('off')
        plt.show()


# %%
image = np.array([
    [0, 1, 4],
    [1, 2, 5],
    [2, 5, 4],
])

cc = CubicalComplex(image)
cc.display()
for cell in cc.cells:
    print(cell) 

print(cc.cc_tree.parent)
print(cc.cc_tree.birth)
print(cc.cc_tree.roots)

print(cc.h0_births)

# %%
