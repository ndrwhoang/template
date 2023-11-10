import json
import typing as t
import logging
from pathlib import Path


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def dump_jsonl(object: t.List[t.Dict], path: str):
    with open(Path(path), "w", encoding="utf-8") as f:
        for line in object:
            json.dump(line, f)
            f.write("\n")

    logger.info(f" Dump {len(object)} samples to {path}")
