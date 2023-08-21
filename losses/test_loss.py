from losses.base_loss import BaseLoss
import torch
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TestLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reduction = cfg.reduction

    def _get_loss(self, pred, dest):
        """Returns the loss function without the coefficient.
        :param pred: the predicted value. tensor (*, out_dim)
        :param dest: the destination value. tensor (*, out_dim)
        """
        # L2 loss
        loss = torch.linalg.norm(pred - dest, dim=-1) ** 2
        # reduction
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        raise ValueError(f'Invalid reduction: {self.reduction}, should be sum or mean')
