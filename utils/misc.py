from typing import Optional
from omegaconf import OmegaConf, DictConfig
import sys
import os
import importlib


def scandir(dir_path: str, suffix: Optional[str] = None, recursive: bool = False, full_path: bool = False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(inner_dir_path: str, inner_suffix: Optional[str], inner_recursive: bool):
        for entry in os.scandir(inner_dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if inner_suffix is None:
                    yield return_path
                elif return_path.endswith(inner_suffix):
                    yield return_path
            else:
                if inner_recursive:
                    yield from _scandir(entry.path, inner_suffix=inner_suffix, inner_recursive=inner_recursive)
                else:
                    continue

    return _scandir(dir_path, inner_suffix=suffix, inner_recursive=recursive)


def print_cfg(cfg: DictConfig, file=sys.stderr):
    hash_str = '#' * 10
    print(hash_str + ' Printing Configuration ' + hash_str, file=file)
    print(OmegaConf.to_yaml(cfg), file=file)
    print(hash_str + ' Configuration over ' + hash_str, file=file)


def init_callbacks(cfg: Optional[DictConfig]):
    # if no callbacks are provided, return None
    if cfg is None:
        return None

    # import the callbacks module
    module = importlib.import_module('pytorch_lightning.callbacks')

    # initialize the callbacks
    callback_list = []
    for callback_name, params in cfg.items():
        callback = getattr(module, callback_name)
        callback_list.append(callback(**params))

    return callback_list
