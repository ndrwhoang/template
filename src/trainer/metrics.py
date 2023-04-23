import logging

import numpy as np
from sklearn.metrics import f1_score, classification_report


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    logger.info(classification_report(labels, predictions, zero_division=0))

    return {"f1": f1}
