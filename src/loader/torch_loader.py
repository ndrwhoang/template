import logging
import typing as t
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import AutoTokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextDataset(Dataset):
    """
    Dataset class
    """

    def __init__(self, config: t.Dict, mode: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["pretrained_name"]
        )
        self.sample_ids, self.samples, self.labels = self._load_dataset(mode)

    def _load_dataset(self, mode: str) -> t.Tuple[t.List, t.List, t.List]:
        if mode == "train":
            data_path = Path(self.config["training"]["paths"]["train"])
        elif mode == "validate":
            data_path = Path(self.config["training"]["paths"]["val"])
        elif mode == "test":
            data_path = Path(self.config["training"]["paths"]["test"])
        elif mode == "predict":
            data_path = Path(self.config["training"]["paths"]["predict"])
        elif mode == "kfold":
            data_path = Path(self.config["training"]["paths"]["train_val"])

        dataset = pd.read_csv(Path(data_path))

        return dataset["sample_id"], dataset["text"], dataset["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> t.Tuple:
        return self.sample_ids[index], self.samples[index], self.labels[index]

    @staticmethod
    def collate_fn(self, batch: t.Tuple) -> t.Dict:
        (sample_ids, samples, labels) = zip(*batch)

        tokenizer_out = self.tokenizer(
            samples,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=self.config["training"]["max_len"],
        )

        labels = torch.tensor(labels)

        return {
            "sample_ids": sample_ids,
            "input_ids": tokenizer_out["input_ids"],
            "attention_mask": tokenizer_out["attention_mask"],
            "labels": labels,
        }


class TextDataModule(pl.LightningDataModule):
    """
    lightning dataset interface
    """

    def __init__(self, config: t.Dict):
        super().__init__()
        self.config = config

    def setup(self, stage):
        assert stage in ["fit", "validate", "test", "predict", "kfold"]

        if stage == "fit":
            self.train_dataset = TextDataset(self.config, "train")
            self.val_dataset = TextDataset(self.config, "validate")
        elif stage == "validate":
            self.val_dataset = TextDataset(self.config, "validate")
        elif stage == "test":
            self.test_dataset = TextDataset(self.config, "test")
        elif stage == "predict":
            self.predict_dataset = TextDataset(self.config, "predict")
        elif stage == "kfold":
            self.train_val_dataset = TextDataset(self.config, "train_val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["train_bs"],
            pin_memory=True,
            shuffle=True,
            collate_fn=TextDataset.collate_fn,
            num_workers=self.config["training"]["n_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["val_bs"],
            pin_memory=True,
            shuffle=False,
            collate_fn=TextDataset.collate_fn,
            num_workers=self.config["training"]["n_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["test_bs"],
            pin_memory=True,
            shuffle=False,
            collate_fn=TextDataset.collate_fn,
            num_workers=self.config["training"]["n_workers"],
        )

    def train_dataloader_kfold(self, train_ids: t.List):
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.config["training"]["train_bs"],
            pin_memory=True,
            shuffle=True,
            collate_fn=TextDataset.collate_fn,
            num_workers=self.config["training"]["n_workers"],
            sampler=SubsetRandomSampler(train_ids),
        )

    def val_dataloader_kfold(self, val_ids: t.List):
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.config["training"]["val_bs"],
            pin_memory=True,
            shuffle=False,
            collate_fn=TextDataset.collate_fn,
            num_workers=self.config["training"]["n_workers"],
            sampler=SubsetRandomSampler(val_ids),
        )
