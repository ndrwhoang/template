# trainer utils

"""
Helper functions for trainer
"""

from pathlib import Path
import configparser
import logging

from transformers import TrainingArguments


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_config(config: configparser.ConfigParser):
    """
    Save config info that is not in HF's config
    """
    outfile = Path(
        config["training"]["output_dir"], "run_config", config["training"]["run_name"]
    )
    Path(outfile).mkdir(parents=True, exist_ok=True)

    with open(outfile / "config.ini", "w") as f:
        config.write(f)


def get_training_args(config: configparser.ConfigParser):
    training_args = TrainingArguments(
        # general
        run_name=config["run_name"],
        seed=config.getint("seed"),
        output_dir=str(Path(config["output_dir"]) / config["run_name"]),
        per_device_train_batch_size=config.getint("train_bs"),
        per_device_eval_batch_size=config.getint("eval_bs"),
        num_train_epochs=config.getint("n_epochs"),
        save_total_limit=config.getint("n_ckpt_limit"),
        dataloader_num_workers=config.getint("n_workers"),
        evaluation_strategy="epoch",
        # misc
        dataloader_drop_last=True,
        remove_unused_columns=False,
        do_train=True,
        # optimization
        gradient_accumulation_steps=config.getint("gradient_accumulation_steps"),
        fp16=config.getboolean("fp16"),
        gradient_checkpointing=config.getboolean("gradient_checkpointing"),
        fp16_full_eval=config.getboolean("fp16_eval"),
        # logging
        logging_strategy="epoch",
        log_level="info",
        save_strategy="epoch",
        report_to=["tensorboard"],
        # optimizer settings
        optim=config["optimizer"],
        warmup_steps=config.getint("warmup_steps"),
        max_steps=config.getint("max_steps"),
        adam_beta1=config.getfloat("adam_beta1"),
        adam_beta2=config.getfloat("adam_beta2"),
        adam_epsilon=config.getfloat("adam_epsilon"),
        learning_rate=config.getfloat("lr"),
    )

    return training_args
