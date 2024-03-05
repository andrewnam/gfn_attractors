from abc import ABC, abstractmethod
import torch.nn as nn


class ScoreModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, z_hat, x=None, z0=None):
        """
        Returns s(x, z_hat).

        Can use z0 instead of x depending on the model.

        z_hat: (batch_size, z_dim)
        x: (batch_size, ...)
        z0: (batch_size, z_dim)
        returns:
            score: (batch_size, )
            metrics: dict
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, z_hat):
        """
        Returns a sample from p(x | z_hat).
        z_hat: (batch_size, z_dim)
        """
        raise NotImplementedError
    
    def get_loss(self, z_hat, x=None, z0=None):
        """
        Returns the training loss for the model.
        Usually, this would be -score, but may be different for some models.

        Can use z0 instead of x depending on the model.

        z_hat: (batch_size, z_dim)
        x: (batch_size, ...)
        z0: (batch_size, z_dim)
        returns: 
            loss: (batch_size, )
            metrics: dict
        """
        return -self.forward(z_hat, x=x, z0=z0), {}
