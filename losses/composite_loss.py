from losses.base_loss import BaseLoss
from losses.build_loss_func import build_loss


class CompositeLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_funcs = [build_loss(val) for key, val in cfg.losses if key not in ['type', 'coef']]

    def _get_loss(self, *args, **kwargs):
        """Returns the loss function without the coefficient."""
        return sum([loss_func(*args, **kwargs) for loss_func in self.loss_funcs])
