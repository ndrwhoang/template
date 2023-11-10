import logging
from pathlib import Path
import typing as t

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer
import lightning.pytorch as pl


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextDataset(Dataset):
    def __init__(self, dataset: t.Dict[str, t.List]):
        self.input_texts = dataset["input_texts"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.input_texts[index], self.labels[index]

    @staticmethod
    def convert_label_dtype(df: pd.DataFrame) -> pd.DataFrame:
        df["label"] = df["label"].astype(int)

        return df

    @staticmethod
    def placeholder(df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: t.Dict, tokenizer: object, max_length: int):
        (input_texts, labels) = zip(*batch)

        tokenizer_out = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=max_length,
        )

        labels = torch.tensor(labels)

        return {
            "input_ids": tokenizer_out["input_ids"],
            "attention_mask": tokenizer_out["attention_mask"],
            "labels": labels,
        }


class TextDataModule(pl.LightningDataModule):
    def __init__(self, config: t.Dict, tokenizer: AutoTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def _process_dataset(self, path: Path) -> t.List[t.Dict]:
        dataset = pd.read_json(path, lines=True)
        dataset = dataset.pipe(TextDataset.convert_label_dtype).pipe(
            TextDataset.placeholder
        )
        dataset = dataset.to_dict("list")

        return dataset

    def setup(self, stage):
        assert stage in ["fit", "validate", "test", "predict", "kfold"]

        if stage == "fit":
            train_path = Path(self.config["training"]["paths"]["train"])
            val_path = Path(self.config["training"]["paths"]["val"])

            self.train_dataset = TextDataset(self._process_dataset(train_path))
            self.val_dataset = TextDataset(self._process_dataset(val_path))

        if stage == "validate":
            val_path = Path(self.config["training"]["paths"]["val"])
            self.val_dataset = TextDataset(self._process_dataset(val_path))

        if stage == "test":
            test_path = Path(self.config["training"]["paths"]["test"])
            self.test_dataset = TextDataset(self._process_dataset(test_path))

        if stage == "predict":
            predict_path = Path(self.config["inference"]["paths"]["inference"])
            self.predict_dataset = TextDataset(self._process_dataset(predict_path))

        if stage == "kfold":
            train_val_path = Path(self.config["training"]["paths"]["train_val"])
            self.train_val_dataset = TextDataset(self._process_dataset(train_val_path))

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
