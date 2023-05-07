import logging
import configparser

from transformers import AutoConfig

from src.models.encoder import ScoringModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def hyperparameter_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
    }


def get_model_init(config: configparser.ConfigParser):
    pretrained_name = config["model"]["pretrained_name"]

    def model_init(trial):
        autoconfig = AutoConfig.from_pretrained(pretrained_name)
        model = ScoringModel(autoconfig)

        return model

    return model_init
