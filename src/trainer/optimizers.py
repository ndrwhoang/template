import configparser
import logging

from torch import nn
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_optimizer_and_scheduler(
    config: configparser.ConfigParser, model: PreTrainedModel
):
    config = config["optimizer"]

    if config["alg"] == "adam_bnb":
        import bitsandbytes as bnb

        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": config.getfloat("weight_decay"),
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(config.getfloat("adam_beta1"), config.getfloat("adam_beta2")),
            eps=config.getfloat("adam_epsilon"),
            lr=config.getfloat("lr"),
        )
        scheduler = None

    elif config["alg"] == "adafactor":
        from transformers.optimization import Adafactor, AdafactorSchedule

        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )
        scheduler = AdafactorSchedule(optimizer)

    elif config["alg"] == "adamw":
        from torch.optim import AdamW
        from transformers import get_scheduler

        optimizer = AdamW(
            model.parameters(),
            lr=config.getfloat("lr"),
            betas=(config.getfloat("adam_beta1"), config.getfloat("adam_beta2")),
            eps=config.getfloat("adam_epsilon"),
        )
        scheduler = get_scheduler(config["optimizer"]["scheduler"], optimizer)

    return optimizer, scheduler
