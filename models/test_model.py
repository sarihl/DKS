import pytorch_lightning as pl
from typing import Union, Dict
from omegaconf import DictConfig
from networks import build_network
from losses import build_loss
import torch
from utils.registry import MODEL_REGISTRY
from utils.misc import build_optimizer, build_scheduler


@MODEL_REGISTRY.register()
class TestModel(pl.LightningModule):
    def __init__(self, cfg: Union[Dict, DictConfig]):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_1 = build_network(cfg.net_1)
        self.loss = build_loss(cfg.loss_1)

    def forward(self, x):
        return self.model_1(x)

    def training_step(self, batch, batch_idx):
        src, dest = batch
        out = self(src)
        loss = self.loss(out, dest, stage='train', logger=self.log)
        return loss

    def validation_step(self, batch, batch_idx):
        src, dest = batch
        out = self(src)
        loss = self.loss(out, dest, stage='val', logger=self.log)
        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg.optimizer, self.parameters())
        if 'scheduler' in self.cfg:
            scheduler = build_scheduler(self.cfg.scheduler, optimizer)
            return [optimizer], [scheduler]
        return optimizer
