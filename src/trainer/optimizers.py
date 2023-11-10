import configparser
import torch
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


def get_optimizer(config: configparser.ConfigParser, parameters: nn.Parameter):
    if config.getboolean("training", "cpu_checkpointing"):
        optimizer = DeepSpeedCPUAdam(parameters, lr=config.getfloat("training", "lr"))
    else:
        optimizer = FusedAdam(parameters, lr=config.getfloat("training", "lr"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.getint("training", "scheduler_patience")
    )

    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
