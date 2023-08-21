# Adapted from https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching/blob/main/datasets/__init__.py
import importlib
from copy import deepcopy
from os import path as osp
import sys
from utils import scandir
from utils.registry import DATAMODULE_REGISTRY

__all__ = ['build_datamodule']

# automatically scan and import datamodule modules for registry
# scan all the files under the data folder with '_datamodule' in file names
data_folder = osp.dirname(osp.abspath(__file__))
datamodule_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder, recursive=True) if
                        v.endswith('_datamodule.py')]
# import all the datamodule modules
_datamodule_modules = [importlib.import_module(f'datamodules.{file_name}') for file_name in datamodule_filenames]


def build_datamodule(datamodule_opt):
    """Build datamodule from options.

    Args:
        datamodule_opt (dict): Configuration for datamodule. It must contain:
            type (str): DataModule type.
    Return:
        datamodule (pytorch_lightning.DataModule): datamodule built by opt.
    """
    datamodule_opt = deepcopy(datamodule_opt)
    datamodule_type = datamodule_opt.pop('type')
    # build datamodule
    datamodule = DATAMODULE_REGISTRY.get(datamodule_type)(datamodule_opt)
    print(f'datamodule [{datamodule.__class__.__name__}] is built.', file=sys.stderr)  # logging
    return datamodule
