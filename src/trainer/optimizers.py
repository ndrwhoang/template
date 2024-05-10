import typing as t

import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def get_optimizer(config: t.Dict, parameters: nn.Parameter):
    if config["training"]["deepspeed"]["cpu_checkpointing"]:
        optimizer = DeepSpeedCPUAdam(
            parameters, lr=config["training"]["optimizer"]["lr"]
        )
    else:
        optimizer = FusedAdam(parameters, lr=config["training"]["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config["training"]["scheduler"]["patience"]
    )

    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
