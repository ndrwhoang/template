"""
A custom trainer to work around specific issues
(e.g. doing hyperparam search with custom optimizer at the same time 
https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/trainer.py#L2462)
"""

from torch import nn
from transformers import Trainer

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.trainer.optimizers import get_optimizer_and_scheduler


class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer, self.lr_scheduler = get_optimizer_and_scheduler(
            self.args, self.model
        )
