# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassifierHead(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()

    def forward(self, logits: torch.Tensor):
        raise NotImplementedError

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError


class RegressionHead(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()

    def forward(self, logits: torch.Tensor):
        raise NotImplementedError

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError


class ScoringModel(PreTrainedModel):
    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.pretrained_encoder = AutoModel.from_pretrained(config._name_or_path)
        self.prediction_head = ClassifierHead(config)

        # for param in self.pretrained_encoder.parameters():
        #     param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ):
        encoder_output = self.pretrained_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = torch.mean(encoder_output.last_hidden_state, dim=1)
        logits = self.prediction_head(logits)

        loss = self.prediction_head(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)
