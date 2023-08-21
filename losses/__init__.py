# Adapted from https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching/blob/main/networks/__init__.py
import importlib
import os.path as osp
from utils import scandir
from losses.build_loss_func import build_loss

__all__ = ['build_loss']

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with
# '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the loss modules
_loss_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]
