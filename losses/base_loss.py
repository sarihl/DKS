from typing import Optional, Callable


class BaseLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.coef = cfg.coef
        self.log = cfg.log
        if self.log:
            self.logger_params = cfg.logger_params if 'logger_params' in cfg and cfg.logger_params is not None else {}
            self.name = self.logger_params.pop('name')

    def _get_loss(self, *args, **kwargs):
        """Returns the loss function WITHOUT the coefficient."""
        raise NotImplementedError

    def __call__(self, *args, stage: Optional[str] = None, logger: Optional[Callable], **kwargs):
        loss = self.coef * self._get_loss(*args, **kwargs)
        if self.log:
            assert logger is not None
            name = self.name if stage is None else f'{stage}_{self.name}'
            logger(name, loss, **self.logger_params)

        return loss
