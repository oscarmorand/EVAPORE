from typing import Any
import hydra
import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader as ImageDataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from graph_neural_networks.data.dataset.graph_dataset import GraphDataset
from graph.graph_visualization import get_graph_overlay_img, get_virtual_centerline
from graph_neural_networks.reconstruction.reconstruction_method import ReconstructionMethod

from graph_neural_networks.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    pre_hydra_routine,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

OmegaConf.register_new_resolver("eval", eval)

def get_graph_ids(dataloader: GraphDataLoader | ImageDataLoader) -> list[int]:
    graph_ids = []
    for batch in dataloader:
        if isinstance(dataloader, GraphDataLoader):
            id = batch.graph_id.item()
        else:
            id, _ = batch
            id = id.item()
        graph_ids.append(id)
    return graph_ids

def compute_reconstructed_masks(dataset: GraphDataset, 
                                graph_ids: list[int], 
                                predictions: list[torch.Tensor], 
                                reconstruction_method: ReconstructionMethod
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    reconstructed_imgs = []
    reconstructed_masks = []
    for graph_id, pred in tqdm(zip(graph_ids, predictions), total=len(predictions), desc="Computing reconstructed masks and images"):
        all_data = dataset.get_all_from_keys(graph_id, ["nx", "probability_map", "pred"], apply_foreground_mask=True)
        raw_graph = all_data["nx"]
        probability_map = all_data["probability_map"]
        pred_mask = (torch.tensor(all_data["pred"]) > 0).type(torch.uint8)

        paths, radius_paths = reconstruction_method.reconstruct(
            map=probability_map,
            graph=raw_graph,
            new_edges=pred
        )
        reconstructed_img, reconstructed_mask = reconstruction_method.draw_reconstruction(
            mask=pred_mask,
            paths=paths,
            radius_paths=radius_paths,
            old_edges_color=np.array([255, 255, 255]),
            new_edges_color=np.array([255, 0, 0])
        )

        reconstructed_imgs.append(reconstructed_img)
        reconstructed_masks.append(reconstructed_mask)

    return reconstructed_imgs, reconstructed_masks

def log_inference_images(dataset: GraphDataset, 
                         graph_ids: list[int], 
                         predictions: list[torch.Tensor], 
                         edge_scores: list[torch.Tensor], 
                         reconstructed_imgs: list[torch.Tensor], 
                         wandb_logger: WandbLogger
) -> None:
    if predictions is None:
        predictions = [None] * len(reconstructed_imgs)
    if edge_scores is None:
        edge_scores = [None] * len(reconstructed_imgs)
    full_images = []
    for graph_id, pred, edge_score, reconstructed_img in tqdm(zip(graph_ids, predictions, edge_scores, reconstructed_imgs), total=len(predictions), desc="Logging inference images"):
        all_data = dataset.get_all_from_keys(graph_id, ["nx", "img", "probability_map", "pred", "gt"], apply_foreground_mask=True)
        raw_graph = all_data["nx"]
        probability_map = all_data["probability_map"]
        img = all_data["img"]
        pred_mask = (torch.tensor(all_data["pred"]) > 0).type(torch.uint8)
        gt = (torch.tensor(all_data["gt"]) > 0).type(torch.uint8)

        background = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        img_data = get_graph_overlay_img(background, raw_graph, show_edges=True)
        if pred is not None and edge_score is not None:
            for i, (u, v) in enumerate(pred.t().tolist()):
                u_pos, v_pos = raw_graph.nodes[u]['pos'], raw_graph.nodes[v]['pos']
                x0, y0 = int(u_pos[1]), int(u_pos[0])
                x1, y1 = int(v_pos[1]), int(v_pos[0])
                virtual_centerline = get_virtual_centerline(x0, y0, x1, y1)
                cmap = plt.get_cmap('hot')
                if edge_score is not None:
                    color = np.array(cmap(edge_score[i].item())[:3])
                else:
                    color = np.array([1.0, 0.0, 0.0])
                for x, y in virtual_centerline:
                    img_data[y, x, :] = np.floor(color * 255).astype(np.uint8)

        pred_img = np.zeros((dataset.height, dataset.width, 3), dtype=np.uint8)
        pred_img[pred_mask.bool().numpy()] = np.array([255, 255, 255], dtype=np.uint8)

        gt_img = np.zeros((dataset.height, dataset.width, 3), dtype=np.uint8)
        gt_img[gt.bool().numpy()] = np.array([255, 255, 255], dtype=np.uint8)

        prob_img = (probability_map - probability_map.min()) / (probability_map.max() - probability_map.min())
        prob_img = (np.stack([prob_img.numpy()]*3, axis=-1) * 255.0).astype(np.uint8)

        w, h = dataset.width, dataset.height
        full_img = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        full_img[0:h, 0:w, :] = img
        full_img[0:h, w:2*w, :] = gt_img
        full_img[0:h, 2*w:3*w, :] = pred_img
        full_img[h:2*h, 0:w, :] = prob_img
        full_img[h:2*h, w:2*w, :] = img_data
        full_img[h:2*h, 2*w:3*w, :] = reconstructed_img.cpu().numpy()
        full_images.append(full_img)

    wandb_logger.log_image(key="inference_full_images", images=full_images)

@task_wrapper
def infer(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Performs inference given checkpoint on a datamodule test set.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A dictionary containing all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info(f'Instantiating reconstruction method <{cfg.reconstruction._target_}>')
    reconstruction_method: ReconstructionMethod = hydra.utils.instantiate(cfg.reconstruction)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting inference!")
    res = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    predictions = [r[0] for r in res]
    edge_scores = [r[1] for r in res]

    wandb_logger: WandbLogger = logger[0]
    datamodule.setup(stage="predict")
    dataset: GraphDataset = datamodule.dataset
    dataloader = datamodule.predict_dataloader() 
    
    graph_ids = get_graph_ids(dataloader)

    reconstructed_imgs, reconstructed_masks = compute_reconstructed_masks(
        dataset,
        graph_ids,
        predictions,
        reconstruction_method
    )

    log_inference_images(dataset, graph_ids, predictions, edge_scores, reconstructed_imgs, wandb_logger)

    return predictions, object_dict


@hydra.main(version_base=None, config_path="configs", config_name="infer.yaml")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point for inference.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    infer(cfg)


def main() -> None:
    """Main entry point for inference, before Hydra is called.

    This is a workaround for issues with Python packaging tools requiring a function to target for script entrypoints.
    It provides a target for entrypoints that comes before Hydra is called, allowing for pre-Hydra routines to be run
    (e.g. setting up environment variables, registering custom OmegaConf resolvers etc.)
    """
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
