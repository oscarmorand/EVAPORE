import networkx as nx
import torch
import numpy as np
from skimage.morphology import dilation, disk
from skimage.measure import label


def get_query_edges(graph: nx.Graph, n_closest: int = 1, max_dist: float = None) -> torch.Tensor:
    G = graph.copy()

    connected_components = list(nx.connected_components(graph))

    cc_extrimities = {}
    for cc_i, cc in enumerate(connected_components):
        extrimities = []
        for n in cc:
            if G.degree(n) == 1:
                extrimities.append(n)
        cc_extrimities[cc_i] = extrimities

    cc_n_count = [len(cc) for cc in connected_components]
    main_cc = np.argmax(cc_n_count)
    small_ccs = {i: cc for i, cc in enumerate(connected_components) if i != main_cc}

    virtual_edges_to_add = []
    for i, cc in small_ccs.items():
        extremities = cc_extrimities[i]
        other_ccs = {j: other_cc for j, other_cc in enumerate(connected_components) if j != i}
        other_ccs_all_nodes = []
        for other_cc in other_ccs.values():
            other_ccs_all_nodes.extend(other_cc)

        for extrimity in extremities:
            other_nodes_distances = []
            extrimity_pos = np.array(G.nodes[extrimity]['pos'])
            for other_cc_node in other_ccs_all_nodes:
                other_cc_node_pos = np.array(G.nodes[other_cc_node]['pos'])
                dist = np.linalg.norm(extrimity_pos - other_cc_node_pos)
                other_nodes_distances.append(dist)
            other_nodes_distances = np.array(other_nodes_distances)
            closest_indices = np.argsort(other_nodes_distances)[:n_closest]
            if max_dist is not None:
                closest_indices = closest_indices[other_nodes_distances[closest_indices] <= max_dist]
            closest_nodes = [other_ccs_all_nodes[idx] for idx in closest_indices]

            for extremity_closest_other_cc_node in closest_nodes:
                if (extrimity, extremity_closest_other_cc_node) not in virtual_edges_to_add and (extremity_closest_other_cc_node, extrimity) not in virtual_edges_to_add:
                    virtual_edges_to_add.append([extrimity, extremity_closest_other_cc_node])
    
    virtual_edges_index_tensor = torch.tensor(virtual_edges_to_add, dtype=torch.long).t().contiguous()
    return virtual_edges_index_tensor


# ======== Cleaning paths functions ========

def cut_mask_from_negative_edges(centerline, mask):
    centerline_mask = np.zeros_like(mask, dtype=bool)
    for (x, y) in centerline:
        centerline_mask[x, y] = True

    combined_mask = np.logical_and(centerline_mask, np.logical_not(mask))
    labeled_mask, num_labels = label(combined_mask, return_num=True, connectivity=2)
    centerlines = [[] for _ in range(num_labels)]
    for (x, y) in centerline:
        label_id = labeled_mask[x, y]
        if label_id > 0:
            centerlines[label_id - 1].append([int(x), int(y)])
    
    return centerlines

def cut_mask_from_negative_edges_for_all(centerlines, mask, classes=None, edges=None, return_old_centerlines=False):
    new_centerlines = []
    new_classes = []
    new_edges = []
    old_centerlines = []
    for i, centerline in enumerate(centerlines):
        new_centerline = cut_mask_from_negative_edges(centerline, mask)
        n_new_centerline = len(new_centerline)
        if n_new_centerline > 0:
            new_centerlines.extend(new_centerline)
            if return_old_centerlines:
                old_centerlines.extend([centerline for _ in range(n_new_centerline)])
            if classes is not None:
                cls = classes[i]
                new_classes.extend([cls for _ in range(len(new_centerline))])
            if edges is not None:
                edge = edges[i]
                new_edges.extend([edge for _ in range(len(new_centerline))])

    res = {"new_centerlines": new_centerlines}
    if return_old_centerlines:
        res["old_centerlines"] = old_centerlines
    if classes is not None:
        res["classes"] = new_classes
    if edges is not None:
        res["edges"] = new_edges
    return res

def is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold):
    sum_min_distances = 0
    for (xr, yr) in true_path_reconstructed_centerline:
        r = np.array([xr, yr])
        min_dist = float('inf')
        for (xe, ye) in true_path_existing_centerline:
            e = np.array([xe, ye])
            dist = np.linalg.norm(r - e)
            if dist < min_dist:
                min_dist = dist
        sum_min_distances += min_dist
    sum_min_distances /= len(true_path_reconstructed_centerline)
    return sum_min_distances <= distance_ratio_threshold

def remove_too_far_reconstructed_paths_for_all(true_path_existing_centerlines, true_path_reconstructed_centerlines, distance_ratio_threshold):
    new_reconstructed_classes = []
    for i, true_path_reconstructed_centerline in enumerate(true_path_reconstructed_centerlines):
        true_path_existing_centerline = true_path_existing_centerlines[i]
        condition = is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold)
        new_reconstructed_classes.append(int(condition))
    return new_reconstructed_classes

def clean_paths_on_surface_of_mask(centerlines, mask, kernel_size=1, threshold=0.5, classes=None):
    new_centerlines = []
    if classes is not None:
        new_classes = []
        
    labeled_mask = label(mask)
    dilated_mask = dilation(labeled_mask, disk(kernel_size))
    
    for i, centerline in enumerate(centerlines):
        centerline_mask = np.zeros_like(mask, dtype=np.uint8)
        for (x, y) in centerline:
            centerline_mask[x, y] = 1

        combined_mask = centerline_mask * dilated_mask
        values_on_mask = np.count_nonzero(combined_mask)
        n_diff_cc = len(np.unique(combined_mask)) - (1 if 0 in combined_mask else 0)
        ratio_on_mask = values_on_mask / len(centerline)

        if ratio_on_mask <= threshold or n_diff_cc > 1:
            new_centerlines.append(centerline)
            if classes is not None:
                new_classes.append(classes[i])

    if classes is not None:
        return new_centerlines, new_classes
    return new_centerlines
