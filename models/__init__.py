# Adapted from https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching/blob/main/models/__init__.py
import importlib
import os.path as osp
import sys
from utils import scandir
from utils.registry import MODEL_REGISTRY
import wandb
from omegaconf import DictConfig
import pytorch_lightning as pl
import os
import ast

__all__ = ['build_model', 'init_model_and_cfg']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """
    Build model from options
    Args:
        opt (dict): Configuration dict. It must contain:
            type (str): Model type.

    returns:
        model (BaseModel): model built by opt.
    """
    model = MODEL_REGISTRY.get(opt['type'])(opt)
    print(f'Model [{model.__class__.__name__}] is created.', file=sys.stderr)  # logging

    return model


def init_model_and_cfg(cfg: DictConfig, resume: bool) -> (pl.LightningModule, str, DictConfig):
    """
    Initializes model and configuration.
    if resume is True, the model & cfg are loaded from the run_name provided in the config.
    if resume is False, the model & cfg are initialized from scratch using the config.
    :param cfg: configuration, must contain the skeleton provided in options/resume_skeleton.yaml and
                options/run_skeleton.yaml
    :param resume: whether to resume from a previous run or not.
    :return: the model, the path to the checkpoint, and the configuration.
    note: if resume is False, the path to the checkpoint is None, and the configuration is the same as the input cfg.
    """
    if resume:
        wandb.login()
        # the reference to the checkpoint
        checkpoint_reference = (cfg.logger.username + '/' + cfg.logger.project_name + '/' + 'model-'
                                + cfg.logger.run_name + ':' + cfg.logger.alias)

        # download checkpoint locally (if not already cached)
        api = wandb.Api()
        artifact = api.artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()
        chkpt_path = os.path.join(artifact_dir, "model.ckpt")

        # get the config from the run
        run = api.run(cfg.logger.user_name + '/' + cfg.logger.project_name + '/' + cfg.logger.run_name)
        config = {key: ast.literal_eval(run.config['_content'][key]) for key, val in run.config['_content'].items()}
        config = DictConfig(config)

        # build the model
        model = MODEL_REGISTRY.get(cfg['model']['type']).load_from_checkpoint(chkpt_path)
        print(f'Model [{model.__class__.__name__}] is created.', file=sys.stderr)  # logging
        return model, chkpt_path, config
    else:
        return build_model(cfg['model']), None, cfg
