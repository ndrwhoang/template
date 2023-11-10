# ruff: noqa: E402
import abc
import configparser
import typing as t
import copy
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch.callbacks as callbacks
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, MinMetric, MaxMetric

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.trainer.optimizers import get_optimizer


class LightningTrainerInterface(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, config: configparser.ConfigParser):
        super().__init__()
        self.config = config

        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        # To be used for callbacks
        self.best_val_loss = MinMetric()
        self.best_val_accuracy = MaxMetric()

    def configure_optimizers(self):
        return get_optimizer(self.config, self.trainer.model.parameters())

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    @abc.abstractmethod
    def step(self, batch: t.Dict) -> t.Tuple[torch.Tensor, torch.Tensor]:
        pass

    def training_step(self, batch: t.Dict, batch_idx: int) -> torch.Tensor:
        logits, loss = self.step(batch)
        preds = torch.argmax(logits, dim=-1)

        self.train_accuracy(preds, batch["labels"])
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "train_accuracy": self.train_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: t.Dict, batch_idx: int) -> torch.Tensor:
        logits, loss = self.step(batch)
        preds = torch.argmax(logits, dim=-1)

        self.val_accuracy(preds, batch["labels"])
        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "val_accuracy": self.val_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.logged_metrics("val-loss_epoch")
        self.best_val_loss.update(val_loss)
        self.best_val_accuracy.update(self.val_accuracy.compute())

        self.log_dict(
            {
                "val_best_loss": self.best_val_loss.compute(),
                "val_best_accuracy": self.best_val_accuracy.compute(),
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


class LightningModelWrapper(LightningTrainerInterface):
    def __init__(self, config: configparser.ConfigParser, model: nn.Module):
        super().__init__(config)

        self.save_hyperparameters(ignore=["model"])
        self.model = copy.deepcopy(model)

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def step(self, batch: t.Dict) -> t.Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(**batch)
        loss = self.loss_fn(logits, batch["labels"])

        return logits, loss


def get_callbacks(config: configparser.ConfigParser) -> t.List[callbacks.Callback]:
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=config.get("training", "training_out"),
        filename="epoch{epoch:02d}-val_accuracy{val_accuracy:.2f}",
        save_last=True,
        save_top_k=3,
        mode="max",
        auto_insert_metric_name=False,
    )
    model_summary_callback = callbacks.RichModelSummary()
    lr_monitor_callback = callbacks.LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )
    gpu_monitor_callback = callbacks.DeviceStatsMonitor()

    trainer_callbacks = [
        checkpoint_callback,
        model_summary_callback,
        lr_monitor_callback,
        gpu_monitor_callback,
    ]

    return trainer_callbacks


def get_logger(config: configparser.ConfigParser):
    logger = WandbLogger(
        name=config.get("training", "run_name"),
        save_dir=config.get("training", "training_out"),
    )

    return logger


def get_strategy(config: configparser.ConfigParser):
    strategy = DeepSpeedStrategy(
        stage=3,
        offload_optimizer=config.getboolean("training", "offload_optimizer"),
        offload_parameters=config.getboolean("training", "offload_parameters"),
        cpu_checkpoint=config.getboolean("training", "cpu_checkpointing"),
    )

    return strategy


def get_trainer(config: configparser.ConfigParser):
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=config.get("training", "precision"),
        max_epochs=config.getint("training", "n_epochs"),
        accumulate_grad_batches=config.getint("training", "n_accumulate_steps"),
        deterministic=True,
        inference_mode=True,
        profiler="simple",
        logger=get_logger(config),
        strategy=get_strategy(config),
        callbacks=get_callbacks(config),
        default_root_dir=config.get("training", "output_dir"),
        fast_dev_run=config.getboolean("training", "test_run"),
    )

    return trainer