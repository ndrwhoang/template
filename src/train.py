import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import configparser
import logging
import os

import torch
from transformers import AutoTokenizer, AutoConfig

from src.loader.base_loader import import_datasets
from src.models.encoder import ScoringModel
from src.trainer import (
    metrics as trainer_metrics,
    optimizers as trainer_optimizers,
    utils as trainer_utils,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_setup(config: configparser.ConfigParser):
    pretrained_name = config["model"]["pretrained_name"]
    logger.info(f" Using {pretrained_name} backbone")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    datasets = import_datasets(config, tokenizer)
    autoconfig = AutoConfig.from_pretrained(pretrained_name)
    model = ScoringModel(autoconfig)
    training_args = trainer_utils.get_training_args(config["training"])
    optimizer, scheduler = trainer_optimizers.get_optimizer_and_scheduler(
        training_args, model
    )

    return model, datasets, training_args, optimizer, scheduler


def basic_train(config):
    from transformers import Trainer

    (model, datasets, training_args, optimizer, scheduler) = training_setup(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=trainer_metrics.compute_metrics,
        optimizers=(optimizer, scheduler),
    )
    # print(trainer.state.best_metric)
    trainer.train()


def hyperparameter_search(config):
    from src.trainer import hyperparameter_search as hp_search
    from src.trainer.custom_trainer import CustomTrainer

    (_, datasets, training_args, _, _) = training_setup(config)
    model_init = hp_search.get_model_init(config)

    trainer = CustomTrainer(
        model=None,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=trainer_metrics.compute_metrics,
        model_init=model_init,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_search.hyperparameter_space,
        n_trials=config.getint("training", "n_trials"),
    )

    logger.info(best_trial)


def main():
    config = configparser.ConfigParser()
    config.read(Path("configs/config.ini"))

    trainer_utils.save_config(config)
    basic_train(config)


if __name__ == "__main__":
    main()
