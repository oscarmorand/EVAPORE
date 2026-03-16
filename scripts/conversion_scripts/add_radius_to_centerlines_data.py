import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from graph.graph_creation import img_to_graph
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
import networkx as nx
from multiprocessing import Pool, cpu_count

data_folder = "/home/morand/afs/EVAPORE/data/FIVES/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")
img_folder = os.path.join(data_folder, "img")
prob_map_folder = os.path.join(data_folder, "probability_maps")
centerlines_folder = os.path.join(data_folder, "centerlines")

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

eval_transforms = {
    "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}

def process_case(i):
    # ========= Load data ========
    filename = gt_path_list[i]
    pred_path = os.path.join(pred_folder, filename)

    centerlines_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))
    with open(centerlines_path, 'r') as f:
        centerlines_data = json.load(f)
    centerlines_eval_data = centerlines_data["eval"]

    segmentation_mask_np = np.array(Image.open(pred_path).convert("L")) > 0

    # ======== Train data creation ========
    pred_graph: nx.Graph = img_to_graph(segmentation_mask_np, clean = True, closing_radius=1, return_pixel_graph=False)
    graph_wrapper = GraphWrapper(pred_graph)
    new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, eval_transforms).get_graph()

    nodes_radius_data = {}
    for id, n_data in new_nx_graph.nodes(data=True):
        nodes_radius_data[id] = float(n_data["radius"])

    # ======== Save data ========
    centerlines_eval_data["nodes_radius"] = nodes_radius_data
    centerlines_data["eval"] = centerlines_eval_data

    with open(centerlines_path, 'w') as f:
        json.dump(centerlines_data, f, indent=4)

def main():
    n_jobs = max(1, cpu_count() - 15)

    print(f"Launching multiprocessing with {n_jobs} workers")

    with Pool(processes=n_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_case, range(len(gt_path_list))),
                total=len(gt_path_list),
                desc="Processing cases"
            )
        )

def test():
    process_case(0)


if __name__ == "__main__":
    main()