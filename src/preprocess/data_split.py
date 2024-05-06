# ruff: noqa: E402
# flake8: noqa: E402
import logging
import typing as t
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_dev_test_split(df: pd.DataFrame, config: t.Dict) -> t.List[t.Dict]:
    df = df.sample(frac=1, random_state=config["general"]["seed"]).reset_index(
        drop=True
    )
    df_train, df_dev_test = train_test_split(
        df, train_size=config.get("dataset", "train_size")
    )
    df_dev, df_test = train_test_split(df_dev_test, train_size=0.5)

    df_train.to_json(config.get("path", "train"), orient="records", lines=True)
    df_dev.to_json(config.get("path", "dev"), orient="records", lines=True)
    df_test.to_json(config.get("path", "test"), orient="records", lines=True)

    logger.info(
        f' dumped data to: \n{config.get("path", "train")} \n{config.get("path", "dev")} \n{config.get("path", "test")}'
    )


def main():
    with open(Path("configs", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_json("data/raw/data.jsonl", lines=True)
    train_dev_test_split(df, config)


if __name__ == "__main__":
    main()
