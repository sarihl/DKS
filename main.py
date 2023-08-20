# A readme.txt file is provided in the root directory of the project.
# It provides a brief explanation of the code structure.
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import hydra
from omegaconf import DictConfig
from utils import print_cfg, init_callbacks
import torch
from models import init_model_and_cfg
from datamodules import build_datamodule


@hydra.main(version_base=None, config_path='options', config_name='base')
def main(cfg: DictConfig):
    # printing configuration
    print_cfg(cfg)

    # initializations
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision(cfg.precision)

    # initializing logger
    resume = cfg.logger.run_name is not None
    logger = None
    if cfg.logger.enabled:
        wandb.login()
        # if the run should be resumed, we need to pass the resume flag to the logger
        kw = {'resume': 'must'} if resume else {}
        # initializing logger
        logger = WandbLogger(project=cfg.logger.project_name, log_model=cfg.logger.log_model, id=cfg.logger.run_name,
                             **kw)
        # uploading config to wandb
        wandb.config.update(cfg, allow_val_change=True)

    # initializing callbacks
    callbacks = init_callbacks(cfg.trainer.callbacks)

    # initializing model
    model, ckpt_path, cfg = init_model_and_cfg(cfg, resume=resume)

    # initializing data module
    dm = build_datamodule(cfg.data)

    # initializing trainer
    trainer = pl.Trainer(devices=cfg.trainer.devices, accelerator=cfg.trainer.accelerator,
                         log_every_n_steps=cfg.trainier.log_every_n_steps, max_epochs=cfg.trainer.max_epochs,
                         callbacks=callbacks, logger=logger)

    # training
    trainer.fit(model, train_dataloaders=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
