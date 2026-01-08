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
from torch_geometric.loader import DataLoader

from graph_neural_networks.data.dataset.graph_dataset import GraphDataset
from graph_neural_networks.reconstruction.reconstruction_method import ReconstructionMethod
from graph_neural_networks.infer import compute_reconstructed_masks, log_inference_images, get_graph_ids

from graph_neural_networks.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    pre_hydra_routine,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def compute_image_based_metrics(dataset: GraphDataset, 
                                graph_ids: list[int],
                                image_metrics: dict[str, Metric],
                                reconstructed_masks: list[torch.Tensor]
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    before_reconstruction_scores = {metric_name: [] for metric_name in image_metrics}
    after_reconstruction_scores = {metric_name: [] for metric_name in image_metrics}
    for graph_id, reconstructed_mask in tqdm(zip(graph_ids, reconstructed_masks), total=len(reconstructed_masks), desc="Computing image-based metrics"):
        all_data = dataset.get_all_from_keys(graph_id, ["pred", "gt"])
        pred_mask = (torch.tensor(all_data["pred"]) > 0).type(torch.uint8)
        gt = (torch.tensor(all_data["gt"]) > 0).type(torch.uint8)
 
        # Metrics computation
        for metric_name, metric in image_metrics.items():
            score = metric.get_score(pred=pred_mask, target=gt)
            before_reconstruction_scores[metric_name].append(score)

        for metric_name, metric in image_metrics.items():
            score = metric.get_score(pred=reconstructed_mask, target=gt)
            after_reconstruction_scores[metric_name].append(score)

    return before_reconstruction_scores, after_reconstruction_scores

def log_metrics(before_reconstruction_scores: dict[str, list[float]],
                 after_reconstruction_scores: dict[str, list[float]], 
                 image_metrics: dict[str, Metric], 
                 wandb_logger: WandbLogger
) -> None:
    boxplot_images = []
    for metric_name in tqdm(image_metrics.keys(), desc="Logging image-based metrics"):
        before_scores = before_reconstruction_scores[metric_name]
        after_scores = after_reconstruction_scores[metric_name]
        relative_differences = [(a - b) / (b + 1e-8) for a, b in zip(after_scores, before_scores)]
        relative_differences = [rd for rd in relative_differences if (not np.isnan(rd) and not np.isinf(rd) and rd < 1e2)]

        wandb_logger.log_table(key=f"inference_{metric_name}_scores_before_reconstruction", 
                             data=[[d] for d in before_scores],
                             columns=["data"])
        wandb_logger.log_table(key=f"inference_{metric_name}_scores_after_reconstruction",
                                data=[[d] for d in after_scores],
                                columns=["data"])
        wandb_logger.log_table(key=f"inference_{metric_name}_scores_relative_difference",
                               data=[[d] for d in relative_differences],
                               columns=["data"])
        
        y = ([before_scores, after_scores], [relative_differences])
        labels = (["before", "after"], ["relative difference"], ["relative_difference (without outliers)"])
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
        ax[0].boxplot(y[0], tick_labels=labels[0], showfliers=True)
        ax[0].set_ylabel('metric value')
        ax[0].set_title(metric_name)

        ax[1].boxplot(y[0], tick_labels=labels[0], showfliers=False)
        ax[1].set_title(f"{metric_name} (without outliers)")

        ax[2].boxplot(y[1], tick_labels=labels[1], showfliers=True)
        ax[3].boxplot(y[1], tick_labels=labels[1], showfliers=False)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
        plt.close(fig)
        boxplot_images.append(img)

    wandb_logger.log_image(key=f"boxplots", images=boxplot_images)

@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule test set.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A pair of dictionaries containing metrics and all instantiated objects, respectively.
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

    log.info(f"Instantiating image metrics...")
    image_metrics: MetricCollection = hydra.utils.instantiate(cfg.get("image_metrics", []))

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

    wandb_logger: WandbLogger = logger[0]

    log.info("Starting testing!")
    res = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    predictions = [r[0] for r in res]
    edge_scores = [r[1] for r in res]

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

    before_reconstruction_scores, after_reconstruction_scores = compute_image_based_metrics(
        dataset,
        graph_ids,
        image_metrics,
        reconstructed_masks
    )

    log_metrics(before_reconstruction_scores, after_reconstruction_scores, image_metrics, wandb_logger)

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
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
