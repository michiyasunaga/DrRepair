import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, config, meta):
        super().__init__()

    def initialize(self, config, meta):
        """
        Initialize GloVe or whatever.
        """
        pass

    def forward(self, batch):
        """
        Return "logit" that can be read by get_loss and get_pred.
        """
        raise NotImplementedError

    def get_loss(self, logit, batch):
        """
        Args:
            logit: Output from forward(batch)
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        raise NotImplementedError

    def get_pred(self, logit, batch):
        """
        Args:
            logit: Output from forward(batch)
            batch: list[Example]
        Returns:
            predictions to be read by dataset.evaluate

            Note that you can look at the correct answer in `batch`,
            which is useful for an oracle model.
        """
        raise NotImplementedError
