import configparser
import copy
import typing as t

import torch
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, config: configparser.ConfigParser):
        super(ClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=config.getfloat("model", "dropout")),
            nn.Linear(
                # NOTE: this valuie depends on the encoder used
                config.getint("model", "pretrained_hidden_dim"),
                config.getint("model", "hidden_size"),
            ),
            nn.Dropout(p=config.getfloat("model", "dropout")),
            nn.Tanh(),
            nn.Linear(
                config.getint("model", "hidden_size"),
                # NOTE: this value depends the number of classes
                config.getint("model", "n_classes"),
            ),
        )

    def forward(self, logits: torch.Tensor):
        logits = self.head(logits)

        return logits


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, config: configparser.ConfigParser):
        super(Classifier, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        self.config = config
        self.prediction_head = ClassifierHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: t.Optional[torch.Tensor] = None,
        **kwargs
    ):
        text_encoding = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = torch.mean(text_encoding.last_hidden_state, dim=1)
        logits = self.prediction_head(logits)

        return logits
