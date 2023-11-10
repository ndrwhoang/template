# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path
import typing as t

from sklearn.model_selection import KFold
import numpy as np
import yaml
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.loader.torch_loader import TextDataModule
from src.models.classifier import Classifier
from src.trainer.pl_trainer import get_trainer, LightningModelWrapper

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_setup(config: t.Dict):
    pretrained_encoder = AutoModel.from_pretrained(config["model"]["pretrained_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    if config["training"]["deepspeed"]["cpu_checkpointing"]:
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


def train(config: t.Dict):
    logger.info("Started training")

    (model, data_module, trainer) = training_setup(config)
    trainer.fit(model, datamodule=data_module)


def train_kfold(config: t.Dict):
    # this whole thing is a hack
    logger.info("Started training cross validation")

    pretrained_encoder = AutoModel.from_pretrained(config["model"]["pretrained_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    if config["training"]["deepspeed"]["cpu_checkpointing"]:
        try:
            pretrained_encoder.gradient_checkpointing_enable()
        except ValueError:
            pass

    lightning_data_module = TextDataModule(config, tokenizer)
    lightning_data_module.prepare_data("kfold")
    # temp array of the same shpae opf the dataset so that KFold can split it for ids
    temp_array = np.zeros((len(lightning_data_module.train_val_dataset), 1))
    # The split here controls the number of validation samples in each run 
    # so we don't use more samples for validation than necessary, not the number of folds we run
    kfold = KFold(n_splits=1 / config["training"]["kfold"]["p_val_samples"])

    lightning_trainer = get_trainer(config)

    for i_fold, (train_ids, val_ids) in enumerate(kfold.split(temp_array)):
        if i_fold > config["training"]["kfold"]["n_folds"]:
            break
        logger.info(f" Started training on fold {i_fold}")

        config["training"]["run_name"] += f"_fold_{i_fold}"

        model = Classifier(pretrained_encoder, config)
        lightning_model = LightningModelWrapper(config, model)
        lightning_trainer = get_trainer(config)

        train_dataloader = lightning_data_module.train_dataloader_kfold(train_ids)
        val_dataloader = lightning_data_module.val_dataloader_kfold(val_ids)

        lightning_trainer.fit(lightning_model, train_dataloader, val_dataloader)


def main(config_path: str):
    with open(Path(config_path), "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
