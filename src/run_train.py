# ruff: noqa: E402

import argparse
import configparser
import logging
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.loader.torch_loader import TextDataModule
from src.models.classifier import Classifier
from src.trainer.pl_trainer import get_trainer, LightningModelWrapper

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_setup(config: configparser.ConfigParser):
    pretrained_encoder = AutoModel.from_pretrained(
        config.get("model", "pretrained_name")
    )
    tokenizer = AutoTokenizer.from_pretrained(config.get("model", "pretrained_name"))
    if config.getboolean("training", "cpu_checkpointing"):
        try:
            pretrained_encoder.gradient_checkpointing_enable()
        except ValueError:
            pass

    model = Classifier(pretrained_encoder, config)
    lightning_model = LightningModelWrapper(config, model)

    lightning_data_module = TextDataModule(config, tokenizer)

    lightning_trainer = get_trainer(config)
    # Hack to put deepspeed on any optimizer (this is mainly to suppress the warning)
    lightning_trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

    return lightning_model, lightning_data_module, lightning_trainer


def main(config_path: str):
    logger.info("hello world")
    config = configparser.ConfigParser()
    config.read(Path(config_path))

    (model, data_module, trainer) = training_setup(config)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
