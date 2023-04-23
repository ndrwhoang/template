import configparser
import logging
from pathlib import Path
import typing as t

import torch
from datasets import load_dataset, disable_caching, IterableDatasetDict
from transformers import AutoTokenizer


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
disable_caching()


def import_datasets(
    config: configparser.ConfigParser, tokenizer: AutoTokenizer
) -> IterableDatasetDict:
    train_path = str(Path(config["path"]["train"]))
    val_path = str(Path(config["path"]["val"]))
    test_path = str(Path(config["path"]["test"]))

    datasets = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path, "test": test_path},
        # streaming=True,
    )
    collate_fn = lambda x: convert_to_samples(x, tokenizer, config)
    datasets_out = datasets.map(collate_fn, batched=True)
    datasets_out.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return datasets_out


def convert_to_samples(
    sample: t.Dict, tokenizer: AutoTokenizer, config: configparser.ConfigParser
) -> t.Dict:
    labels = torch.tensor([sample["labels"]], dtype=torch.long)
    labels = labels.permute(1, 0)

    tokenizer_out = tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        max_length=config.getint("model", "max_length"),
    )

    return {
        "input_ids": tokenizer_out["input_ids"],
        "attention_mask": tokenizer_out["attention_mask"],
        "labels": labels,
    }
