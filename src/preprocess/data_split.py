# ruff: noqa: E402

import typing as t
from pathlib import Path
import configparser
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocess.utils import dump_jsonl


def train_dev_test_split(
    df: pd.DataFrame, config: configparser.ConfigParser
) -> t.List[t.Dict]:
    df = df.sample(frac=1, random_state=config.getint("general", "seed")).reset_index(
        drop=True
    )
    df_train, df_dev_test = train_test_split(
        df, train_size=config.getfloat("dataset", "train_size")
    )
    df_dev, df_test = train_test_split(df_dev_test, train_size=0.5)

    dump_jsonl(df_train.to_dict("records"), config.get("path", "train"))
    dump_jsonl(df_dev.to_dict("records"), config.get("path", "dev"))
    dump_jsonl(df_test.to_dict("records"), config.get("path", "test"))


def main():
    config = configparser.ConfigParser()
    config.read(Path("configs", "config.ini"))

    df = pd.read_json("data/raw/data.jsonl", lines=True)
    train_dev_test_split(df, config)


if __name__ == "__main__":
    main()
