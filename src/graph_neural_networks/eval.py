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

    full_images = []
    before_reconstruction_scores = {metric_name: [] for metric_name in image_metrics}
    after_reconstruction_scores = {metric_name: [] for metric_name in image_metrics}
    for batch, pred, edge_score in tqdm(zip(dataloader, predictions, edge_scores), total=len(predictions), desc="Evaluation"):
        pyg_graph = batch
        graph_id = pyg_graph.graph_id.item()

        all_data = dataset.get_all_from_keys(graph_id, ["nx", "img", "probability_map", "pred", "gt"])
        raw_graph = all_data["nx"]
        probability_map = all_data["probability_map"][1]
        img = all_data["img"]
        pred_mask = (torch.tensor(all_data["pred"]) > 0).type(torch.uint8)
        gt = (torch.tensor(all_data["gt"]) > 0).type(torch.uint8)

        background = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        img_data = get_graph_overlay_img(background, raw_graph, show_edges=True)

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
        reconstructed_img = reconstructed_img.numpy()

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
        full_img[h:2*h, 2*w:3*w, :] = reconstructed_img
        full_images.append(full_img)
 
        # Metrics computation
        for metric_name, metric in image_metrics.items():
            score = metric.get_score(pred=pred_mask, target=gt)
            before_reconstruction_scores[metric_name].append(score)

        for metric_name, metric in image_metrics.items():
            score = metric.get_score(pred=reconstructed_mask, target=gt)
            after_reconstruction_scores[metric_name].append(score)

    wandb_logger.log_image(key="inference_full_images", images=full_images)

    # Log metrics
    boxplot_images = []
    for metric_name, _ in image_metrics.items():
        before_scores = before_reconstruction_scores[metric_name]
        after_scores = after_reconstruction_scores[metric_name]
        relative_differences = [(a - b) / (b + 1e-8) for a, b in zip(after_scores, before_scores)]

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
        labels = (["before", "after"], ["relative difference"])
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].boxplot(y[0], tick_labels=labels[0])
        ax[0].set_ylabel('metric value')
        ax[0].set_title(metric_name)
        ax[1].boxplot(y[1], tick_labels=labels[1])
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
        plt.close(fig)
        boxplot_images.append(img)

    wandb_logger.log_image(key=f"boxplots", images=boxplot_images)

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
