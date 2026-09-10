import pytorch_lightning as pl
import json
import os
import matplotlib.pyplot as plt

class SaveConfigCallback(pl.Callback):
    def __init__(self, config: dict, save_dir: str, filename: str = "config.json"):
        self.config = config
        self.save_dir = save_dir
        self.filename = filename

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)

        with open(os.path.join(self.save_dir, self.filename), "w") as f:
            json.dump(self.config, f, indent=2)

        arch_path = os.path.join(self.save_dir, "model_architecture.txt")
        with open(arch_path, "w") as f:
            f.write(str(pl_module))


class PlotMetricsCallback(pl.Callback):
    def __init__(self, save_dir: str, metrics: list[str] = None, every_n_epochs: int = 1):
        self.save_dir = save_dir
        self.metrics = metrics or ["loss", "dice"]
        self.every_n_epochs = every_n_epochs
        self.history = {}  # {"train_loss": [...], "val_loss": [...], ...}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        for metric in self.metrics:
            for stage in ["train", "val"]:
                key = f"{stage}_{metric}"
                if key in trainer.callback_metrics:
                    value = trainer.callback_metrics[key].item()
                    self.history.setdefault(key, []).append(value)

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._plot()

    def _plot(self):
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(6 * len(self.metrics), 4))
        if len(self.metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, self.metrics):
            for stage in ["train", "val"]:
                key = f"{stage}_{metric}"
                if key in self.history:
                    ax.plot(self.history[key], label=stage)
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "curves.png"))
        plt.close(fig)