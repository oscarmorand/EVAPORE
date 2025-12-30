from typing import Any
import hydra
import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from torch_geometric.loader import DataLoader
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

    reconstructed_imgs = []
    link_pred_images = []
    for batch, pred, edge_score in tqdm(zip(dataloader, predictions, edge_scores), total=len(predictions), desc="Evaluation"):
        pyg_graph = batch
        graph_id = pyg_graph.graph_id.item()

        all_data = dataset.get_all_from_keys(graph_id, ["nx", "probability_map", "pred"])
        raw_graph = all_data["nx"]
        probability_map: torch.Tensor = all_data["probability_map"][1]
        pred_img: np.array = all_data["pred"]
        pred_img = torch.tensor(pred_img)

        background = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        img_data = get_graph_overlay_img(background, raw_graph, show_edges=True)

        for i, (u, v) in enumerate(pred.t().tolist()):
            u_pos, v_pos = raw_graph.nodes[u]['pos'], raw_graph.nodes[v]['pos']
            x0, y0 = int(u_pos[1]), int(u_pos[0])
            x1, y1 = int(v_pos[1]), int(v_pos[0])
            virtual_centerline = get_virtual_centerline(x0, y0, x1, y1)
            cmap = plt.get_cmap('hot')
            color = np.array(cmap(edge_score[i].item())[:3])
            for x, y in virtual_centerline:
                img_data[y, x, :] = np.floor(color * 255).astype(np.uint8)

        link_pred_images.append(img_data)

        paths, radius_paths = reconstruction_method.reconstruct(
            map=probability_map,
            graph=raw_graph,
            new_edges=pred
        )
        reconstructed_img, _ = reconstruction_method.draw_reconstruction(
            mask=pred_img,
            paths=paths,
            radius_paths=radius_paths,
            old_edges_color=np.array([255, 255, 255]),
            new_edges_color=np.array([255, 0, 0])
        )
        reconstructed_img = reconstructed_img.numpy()
        reconstructed_imgs.append(reconstructed_img)

    wandb_logger.log_image(key="inference_link_prediction", images=link_pred_images)
    wandb_logger.log_image(key="inference_reconstruction", images=reconstructed_imgs)

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
