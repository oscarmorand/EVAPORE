from typing import Any

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import torch
from torchmetrics import Metric, MetricCollection
from tqdm import tqdm
from lightning.pytorch.loggers.wandb import WandbLogger
import numpy as np
import wandb
import matplotlib.pyplot as plt

from graph.graph_visualization import get_graph_overlay_img, get_virtual_centerline
from graph_neural_networks.data.dataset.graph_dataset import GraphDataset
from graph_neural_networks.reconstruction.reconstruction_method import ReconstructionMethod
from graph_neural_networks.models import LinkPredictionBaselineWithoutLearning
from graph_neural_networks.infer import compute_reconstructed_masks, log_inference_images, get_graph_ids
from graph_neural_networks.eval import compute_image_based_metrics, log_metrics

from graph_neural_networks.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    pre_hydra_routine,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates algorithm with closing on a datamodule test set.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A pair of dictionaries containing metrics and all instantiated objects, respectively.
    """

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger", []))

    log.info(f"Instantiating image metrics...")
    image_metrics: MetricCollection = hydra.utils.instantiate(cfg.get("image_metrics", []))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "image_metrics": image_metrics,
    }

    wandb_logger: WandbLogger = logger[0]

    log.info("Starting inference!")
    datamodule.setup(stage="predict")
    dataset: GraphDataset = datamodule.dataset
    dataloader = datamodule.predict_dataloader() 
    graph_ids = get_graph_ids(dataloader)

    reconstructed_imgs = []
    reconstructed_masks = []
    for batch in tqdm(dataloader, desc="Predicting"):
        reconstructed_img, reconstructed_mask = model.predict_step(batch)
        reconstructed_imgs.append(reconstructed_img)
        reconstructed_masks.append(reconstructed_mask)

    log_inference_images(dataset, graph_ids, None, None, reconstructed_imgs, wandb_logger)

    before_reconstruction_scores, after_reconstruction_scores = compute_image_based_metrics(dataset, graph_ids, image_metrics, reconstructed_masks)

    log_metrics(before_reconstruction_scores, after_reconstruction_scores, image_metrics, wandb_logger)    

    metric_dict = {}

    return metric_dict, object_dict


@hydra.main(version_base=None, config_path="configs", config_name="eval_algo_with_closing.yaml")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


def main() -> None:
    """Main entry point for training, before Hydra is called.

    This is a workaround for issues with Python packaging tools requiring a function to target for script entrypoints.
    It provides a target for entrypoints that comes before Hydra is called, allowing for pre-Hydra routines to be run
    (e.g. setting up environment variables, registering custom OmegaConf resolvers etc.)
    """
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
