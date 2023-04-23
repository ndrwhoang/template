import configparser
import json
import logging
import random
import typing as t
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_to_jsonl(output: t.List, path_out: Path):
    with open(path_out, "w", encoding="utf-8") as f:
        for line in output:
            json.dump(line, ensure_ascii=False)
            f.write("\n")


def train_val_test_split(
    config: configparser.ConfigParser,
    train_filename: str,
    val_filename: str,
    test_filename: str,
):
    """
    Read, shuffle, split, dump
    """

    def _load_all_samples():
        raise NotImplementedError

    all_samples = _load_all_samples()

    # Shuffle
    random.shuffle(all_samples)
    cutoff = int(len(all_samples) * config.getfloat("preprocess", "train_p"))
    train_samples = all_samples[:cutoff]
    val_test_samples = all_samples[cutoff:]
    val_samples = val_test_samples[: len(val_test_samples) // 2]
    test_samples = val_test_samples[len(val_test_samples) // 2 :]

    # Dump
    train_out = Path(config["path"]["train"]) / f"{train_filename}.jsonl"
    val_out = Path(config["path"]["val"]) / f"{val_filename}.jsonl"
    test_out = Path(config["path"]["test"]) / f"{test_filename}.jsonl"

    write_to_jsonl(train_samples, train_out)
    write_to_jsonl(val_samples, val_out)
    write_to_jsonl(test_samples, test_out)

    logger.info(f" Dumped {len(train_samples)} samples to {str(train_out)}")


def main():
    config = configparser.ConfigParser()
    config.read(Path("configs/config.ini"))
    random.seed(config.getint("general", "seed"))

    train_val_test_split(config, "train", "val", "test")
